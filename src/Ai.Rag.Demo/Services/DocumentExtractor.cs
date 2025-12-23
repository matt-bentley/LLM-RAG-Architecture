using Ai.Rag.Demo.Models;
using Microsoft.Extensions.Options;
using Microsoft.ML.Tokenizers;

namespace Ai.Rag.Demo.Services;

/// <summary>
/// Service for processing documents by using PDF bookmarks (outlines) to identify sections
/// and chunking text accordingly using token-based sizing
/// </summary>
public class DocumentExtractor : IDocumentExtractor
{
    private readonly int _maxChunkTokens;
    private readonly int _chunkOverlapTokens;
    private readonly Tokenizer _tokenizer;
    private static readonly char[] _sentenceEndings = ['.', '!', '?'];

    public DocumentExtractor(IOptions<EmbeddingSettings> embeddingSettings)
    {
        _maxChunkTokens = embeddingSettings.Value.MaxChunkTokens;
        _chunkOverlapTokens = embeddingSettings.Value.ChunkOverlapTokens;
        if (_maxChunkTokens <= 0)
        {
            throw new ArgumentException("Max chunk tokens must be greater than 0", nameof(_maxChunkTokens));
        }

        if (_chunkOverlapTokens < 0 || _chunkOverlapTokens >= _maxChunkTokens)
        {
            throw new ArgumentException("Chunk overlap tokens must be between 0 and max chunk tokens", nameof(_chunkOverlapTokens));
        }

        _tokenizer = TiktokenTokenizer.CreateForModel("gpt-4");
    }

    /// <summary>
    /// Process a document file: extract text and create chunks based on bookmarks
    /// </summary>
    public async Task<List<DocumentChunk>> ProcessDocumentAsync(string filePath, IExtractionStrategy extractionStrategy, CancellationToken cancellationToken = default)
    {
        var fileName = Path.GetFileName(filePath);
        var sections = await ExtractSectionsAsync(filePath, extractionStrategy, cancellationToken);
        return CreateChunksFromSections(sections, fileName);
    }

    /// <summary>
    /// Process multiple document files: extract text and create chunks for each
    /// </summary>
    public async Task<Dictionary<string, List<DocumentChunk>>> ProcessDocumentsAsync(
        IEnumerable<string> filePaths,
        IExtractionStrategy extractionStrategy,
        CancellationToken cancellationToken = default)
    {
        var results = new Dictionary<string, List<DocumentChunk>>();

        foreach (var filePath in filePaths)
        {
            var fileName = Path.GetFileName(filePath);
            var chunks = await ProcessDocumentAsync(filePath, extractionStrategy, cancellationToken);
            results[fileName] = chunks;
        }

        return results;
    }

    private Task<List<DocumentSection>> ExtractSectionsAsync(string filePath, IExtractionStrategy extractionStrategy, CancellationToken cancellationToken)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"PDF file not found: {filePath}");
        }
        using var file = File.OpenRead(filePath);
        return extractionStrategy.ExtractAsync(file);
    }

    private List<DocumentChunk> CreateChunksFromSections(List<DocumentSection> sections, string sourceDocument)
    {
        var chunks = new List<DocumentChunk>();

        foreach (var section in sections)
        {
            var sectionChunks = CreateChunksFromSection(section, sourceDocument);
            chunks.AddRange(sectionChunks);
        }

        return chunks;
    }

    private List<DocumentChunk> CreateChunksFromSection(DocumentSection section, string sourceDocument)
    {
        var chunks = new List<DocumentChunk>();

        if (section.PageTexts.Count == 0)
        {
            return chunks;
        }

        var fullText = section.Text;
        if (string.IsNullOrWhiteSpace(fullText))
        {
            return chunks;
        }

        // If section fits in one chunk, create it directly with all pages
        var tokenCount = _tokenizer.CountTokens(fullText);
        if (tokenCount <= _maxChunkTokens)
        {
            var pages = section.PageTexts.Select(p => p.PageNumber).ToList();
            chunks.Add(CreateChunk(fullText, section, pages, sourceDocument, 0, 0, 1));
            return chunks;
        }

        // Section is too large - split while tracking which pages each chunk comes from
        var subChunks = SplitLargeSectionWithPages(section);
        var subChunkCount = subChunks.Count;

        for (int i = 0; i < subChunkCount; i++)
        {
            var (text, pages) = subChunks[i];
            chunks.Add(CreateChunk(text, section, pages, sourceDocument, i, i, subChunkCount));
        }

        return chunks;
    }

    private List<(string Text, List<int> Pages)> SplitLargeSectionWithPages(DocumentSection section)
    {
        var result = new List<(string Text, List<int> Pages)>();
        var currentChunkText = new System.Text.StringBuilder();
        var currentChunkPages = new HashSet<int>();
        var currentTokenCount = 0;

        foreach (var pageText in section.PageTexts)
        {
            var pageContent = pageText.Text;
            var pageContentIndex = 0;

            while (pageContentIndex < pageContent.Length)
            {
                var remainingPageContent = pageContent.Substring(pageContentIndex);
                var spaceNeeded = currentChunkText.Length > 0 ? 1 : 0; // space separator if not empty
                var availableTokens = _maxChunkTokens - currentTokenCount - spaceNeeded;

                if (availableTokens <= 0)
                {
                    // No space left - flush current chunk
                    if (currentChunkText.Length > 0)
                    {
                        result.Add((currentChunkText.ToString().Trim(), currentChunkPages.OrderBy(p => p).ToList()));

                        var overlapText = GetOverlapText(currentChunkText.ToString());
                        var lastPage = currentChunkPages.Count > 0 ? currentChunkPages.Max() : (int?)null;

                        currentChunkText.Clear();
                        currentChunkPages.Clear();

                        if (!string.IsNullOrEmpty(overlapText))
                        {
                            currentChunkText.Append(overlapText);
                            currentTokenCount = _tokenizer.CountTokens(overlapText);
                            if (lastPage.HasValue)
                            {
                                currentChunkPages.Add(lastPage.Value);
                            }
                        }
                        else
                        {
                            currentTokenCount = 0;
                        }
                    }
                    continue;
                }

                var remainingTokenCount = _tokenizer.CountTokens(remainingPageContent);

                if (remainingTokenCount <= availableTokens)
                {
                    // Remaining page content fits entirely
                    if (currentChunkText.Length > 0)
                    {
                        currentChunkText.Append(' ');
                        currentTokenCount += 1;
                    }
                    currentChunkText.Append(remainingPageContent);
                    currentTokenCount += remainingTokenCount;
                    currentChunkPages.Add(pageText.PageNumber);
                    break; // Done with this page
                }
                else
                {
                    // Take as much as fits, trying to break at sentence boundary
                    var textToTake = GetTextWithinTokenLimit(remainingPageContent, availableTokens);

                    if (currentChunkText.Length > 0)
                    {
                        currentChunkText.Append(' ');
                    }
                    currentChunkText.Append(textToTake);
                    currentChunkPages.Add(pageText.PageNumber);

                    // Flush the chunk
                    result.Add((currentChunkText.ToString().Trim(), currentChunkPages.OrderBy(p => p).ToList()));

                    var overlapText = GetOverlapText(currentChunkText.ToString());
                    var lastPageNum = currentChunkPages.Count > 0 ? currentChunkPages.Max() : (int?)null;

                    currentChunkText.Clear();
                    currentChunkPages.Clear();

                    if (!string.IsNullOrEmpty(overlapText))
                    {
                        currentChunkText.Append(overlapText);
                        currentTokenCount = _tokenizer.CountTokens(overlapText);
                        if (lastPageNum.HasValue)
                        {
                            currentChunkPages.Add(lastPageNum.Value);
                        }
                    }
                    else
                    {
                        currentTokenCount = 0;
                    }

                    // Move past the content we took
                    pageContentIndex += textToTake.Length;
                }
            }
        }

        // Don't forget the last chunk
        if (currentChunkText.Length > 0)
        {
            result.Add((currentChunkText.ToString().Trim(), currentChunkPages.OrderBy(p => p).ToList()));
        }

        return result;
    }

    /// <summary>
    /// Gets text from the start of the input that fits within the token limit,
    /// preferring to break at sentence boundaries
    /// </summary>
    private string GetTextWithinTokenLimit(string text, int maxTokens)
    {
        // Binary search to find approximate character position for token limit
        var low = 0;
        var high = text.Length;
        var bestFit = 0;

        while (low <= high)
        {
            var mid = (low + high) / 2;
            var substring = text.Substring(0, mid);
            var tokens = _tokenizer.CountTokens(substring);

            if (tokens <= maxTokens)
            {
                bestFit = mid;
                low = mid + 1;
            }
            else
            {
                high = mid - 1;
            }
        }

        if (bestFit == 0)
        {
            // Even one character exceeds limit, take minimum
            return text.Length > 0 ? text.Substring(0, 1) : string.Empty;
        }

        var textToSearch = text.Substring(0, bestFit);

        // Try to find a sentence boundary within the last quarter of the text
        var searchStart = Math.Max(0, bestFit - bestFit / 4);
        var lastSentenceEnd = textToSearch.LastIndexOfAny(_sentenceEndings, bestFit - 1, bestFit - searchStart);

        if (lastSentenceEnd > 0)
        {
            return text.Substring(0, lastSentenceEnd + 1);
        }

        return textToSearch;
    }

    private string GetOverlapText(string text)
    {
        if (_chunkOverlapTokens <= 0)
        {
            return string.Empty;
        }

        var totalTokens = _tokenizer.CountTokens(text);
        if (totalTokens <= _chunkOverlapTokens)
        {
            return string.Empty;
        }

        // Find text from the end that contains approximately _chunkOverlapTokens tokens
        // Use binary search to find the start position
        var low = 0;
        var high = text.Length;
        var bestStart = text.Length;

        while (low <= high)
        {
            var mid = (low + high) / 2;
            var substring = text.Substring(mid);
            var tokens = _tokenizer.CountTokens(substring);

            if (tokens >= _chunkOverlapTokens)
            {
                bestStart = mid;
                low = mid + 1;
            }
            else
            {
                high = mid - 1;
            }
        }

        if (bestStart >= text.Length)
        {
            return string.Empty;
        }

        var overlapRegion = text.Substring(bestStart);

        // Try to find a sentence boundary to start the overlap at a clean point
        var sentenceStart = -1;
        for (int i = 0; i < overlapRegion.Length - 1; i++)
        {
            var c = overlapRegion[i];
            if (_sentenceEndings.Contains(c) && char.IsWhiteSpace(overlapRegion[i + 1]))
            {
                // Start after the whitespace following the punctuation
                sentenceStart = i + 2;
            }
        }

        if (sentenceStart > 0 && sentenceStart < overlapRegion.Length)
        {
            return overlapRegion.Substring(sentenceStart);
        }

        // No sentence boundary found - return the raw overlap
        return overlapRegion;
    }

    private static DocumentChunk CreateChunk(
        string text,
        DocumentSection section,
        List<int> pages,
        string sourceDocument,
        int chunkIndex,
        int subChunkIndex,
        int totalSubChunks)
    {
        var startPage = pages.Count > 0 ? pages.Min() : section.StartPage;
        var endPage = pages.Count > 0 ? pages.Max() : section.EndPage;

        var metadata = new Dictionary<string, string>
        {
            ["ChunkType"] = "Bookmark",
            ["BookmarkLevel"] = section.Level.ToString()
        };

        if (pages.Count > 0)
        {
            metadata["Pages"] = string.Join(",", pages);
        }

        // Create a section-safe ID using the full heading path hash for uniqueness
        var sectionHash = string.IsNullOrEmpty(section.FullHeadingPath) 
            ? "nosection" 
            : Math.Abs(section.FullHeadingPath.GetHashCode()).ToString();

        return new DocumentChunk
        {
            Id = $"{sourceDocument}_bookmark_{sectionHash}_chunk_{chunkIndex}",
            Text = text,
            Embedding = [],
            SourceDocument = sourceDocument,
            ChunkIndex = chunkIndex,
            ChunkTotal = totalSubChunks,
            Section = section.HeadingText ?? sourceDocument,
            StartPage = startPage,
            EndPage = endPage,
            SectionPath = section.FullHeadingPath ?? section.HeadingText ?? sourceDocument,
            Metadata = metadata
        };
    }
}

using Ai.Rag.Demo.Models;
using UglyToad.PdfPig;
using UglyToad.PdfPig.Outline;

namespace Ai.Rag.Demo.DocumentExtraction.ExtractionStrategies
{
    public sealed class PdfBookmarkExtractor : IExtractionStrategy
    {
        private readonly int _maxBookmarkDepth;
        private readonly int _skipPages;

        public PdfBookmarkExtractor(int maxBookmarkDepth = 1, int skipPages = 0)
        {
            if (maxBookmarkDepth < 1)
            {
                throw new ArgumentException("Max bookmark depth must be at least 1", nameof(maxBookmarkDepth));
            }
            if (skipPages < 0)
            {
                throw new ArgumentException("Skip pages cannot be negative", nameof(skipPages));
            }

            _maxBookmarkDepth = maxBookmarkDepth;
            _skipPages = skipPages;
        }

        public Task<List<DocumentSection>> ExtractAsync(Stream file)
        {
            using var document = PdfDocument.Open(file);

            if (!document.TryGetBookmarks(out var bookmarks))
            {
                throw new InvalidOperationException("Failed to read bookmarks from PDF document.");
            }

            var pageTexts = ExtractPageTexts(document);
            var flattenedBookmarks = FlattenBookmarks(bookmarks.Roots, 1);

            var sections = CreateSectionsFromBookmarks(flattenedBookmarks, pageTexts, document.NumberOfPages);
            return Task.FromResult(sections);
        }

        private Dictionary<int, PageWordInfo> ExtractPageTexts(PdfDocument document)
        {
            var pageTexts = new Dictionary<int, PageWordInfo>();

            foreach (var page in document.GetPages())
            {
                var pageIndex = page.Number - 1;
                if (pageIndex < _skipPages)
                {
                    continue;
                }
                var wordPositions = PdfTextUtilities.GetWordPositions(page.GetWords());

                pageTexts[page.Number] = new PageWordInfo
                {
                    PageNumber = page.Number,
                    Words = wordPositions,
                    FullText = PdfTextUtilities.BuildTextWithLineBreaks(wordPositions)
                };
            }

            return pageTexts;
        }

        private List<FlattenedBookmark> FlattenBookmarks(IEnumerable<BookmarkNode> nodes, int depth, List<string> parentTitles = null)
        {
            parentTitles ??= [];
            var result = new List<FlattenedBookmark>();

            foreach (var node in nodes)
            {
                // Only include bookmarks within the depth limit
                if (depth <= _maxBookmarkDepth)
                {
                    int? pageNumber = null;
                    double? yPosition = null;

                    // Try to get the destination page number and position
                    if (node is DocumentBookmarkNode docBookmark)
                    {
                        pageNumber = docBookmark.PageNumber;
                        // Get Y position if available from the destination coordinates
                        var dest = docBookmark.Destination;
                        if (dest != null && dest.Coordinates.Top.HasValue)
                        {
                            yPosition = dest.Coordinates.Top.Value;
                        }
                    }

                    // Build full title with parent hierarchy
                    var fullTitle = parentTitles.Count > 0
                        ? string.Join(" > ", parentTitles) + " > " + node.Title
                        : node.Title;

                    var page = pageNumber ?? 1;
                    if (page <= _skipPages)
                    {
                        continue;
                    }
                    result.Add(new FlattenedBookmark
                    {
                        Title = node.Title,
                        FullTitle = fullTitle,
                        PageNumber = page,
                        Level = depth,
                        YPosition = yPosition
                    });
                }

                // Recursively process children with updated parent titles
                if (node.Children != null && node.Children.Any())
                {
                    var childParentTitles = new List<string>(parentTitles) { node.Title };
                    result.AddRange(FlattenBookmarks(node.Children, depth + 1, childParentTitles));
                }
            }

            return result.OrderBy(b => b.PageNumber).ThenBy(b => b.Level).ToList();
        }

        private List<DocumentSection> CreateSectionsFromBookmarks(
            List<FlattenedBookmark> bookmarks,
            Dictionary<int, PageWordInfo> pageWordInfos,
            int totalPages)
        {
            var sections = new List<DocumentSection>();

            // Build a full text index per page for searching (use space-joined for searching)
            var pageFullTexts = pageWordInfos.ToDictionary(
                kvp => kvp.Key,
                kvp => string.Join(" ", kvp.Value.Words.Select(w => w.Text))
            );

            // Keep the line-break version for output
            var pageTextsWithLineBreaks = pageWordInfos.ToDictionary(
                kvp => kvp.Key,
                kvp => kvp.Value.FullText
            );

            for (int i = 0; i < bookmarks.Count; i++)
            {
                var bookmark = bookmarks[i];
                var bookmarkPage = bookmark.PageNumber;

                // Find next bookmark to determine end boundary
                FlattenedBookmark nextBookmark = i + 1 < bookmarks.Count ? bookmarks[i + 1] : null;
                var nextBookmarkPage = nextBookmark?.PageNumber ?? totalPages + 1;

                // Find the actual page and position where this bookmark's heading appears
                var (actualStartPage, headingStartIndex) = FindHeadingLocation(
                    pageFullTexts, bookmark.Title, bookmarkPage, Math.Min(bookmarkPage + 2, totalPages));

                if (actualStartPage == -1)
                {
                    // Heading not found, use bookmark page as fallback
                    actualStartPage = bookmarkPage;
                    headingStartIndex = 0;
                }

                // Calculate the index after the heading text to skip it in the output
                var contentStartIndex = headingStartIndex;
                if (headingStartIndex >= 0 && pageFullTexts.TryGetValue(actualStartPage, out var startPageText))
                {
                    // Find where the heading ends in the text
                    var headingEndIndex = FindHeadingEndIndex(startPageText, bookmark.Title, headingStartIndex);
                    if (headingEndIndex > headingStartIndex)
                    {
                        contentStartIndex = headingEndIndex;
                    }
                }

                // Find where the next bookmark's heading appears to determine end boundary
                var (nextHeadingPage, nextHeadingIndex) = nextBookmark != null
                    ? FindHeadingLocation(pageFullTexts, nextBookmark.Title, nextBookmarkPage, Math.Min(nextBookmarkPage + 2, totalPages))
                    : (-1, -1);

                if (nextHeadingPage == -1 && nextBookmark != null)
                {
                    nextHeadingPage = nextBookmarkPage;
                    nextHeadingIndex = 0;
                }

                // Collect text from pages in this section
                var pageTextsList = new List<PageText>();
                var endPage = nextHeadingPage > 0 ? nextHeadingPage : totalPages;

                for (int pageNum = actualStartPage; pageNum <= endPage; pageNum++)
                {
                    if (!pageFullTexts.TryGetValue(pageNum, out var pageFullText) || string.IsNullOrEmpty(pageFullText))
                    {
                        continue;
                    }

                    // Use the line-break version for output, but calculate indices from space-joined version
                    var textWithLineBreaks = pageTextsWithLineBreaks.GetValueOrDefault(pageNum, pageFullText);
                    var textToUse = textWithLineBreaks;

                    // For the first page, start from after the heading
                    if (pageNum == actualStartPage && contentStartIndex > 0)
                    {
                        // Find corresponding position in line-break text
                        var adjustedIndex = MapIndexToLineBreakText(pageFullText, textWithLineBreaks, contentStartIndex);
                        if (adjustedIndex >= 0 && adjustedIndex < textWithLineBreaks.Length)
                        {
                            textToUse = textWithLineBreaks.Substring(adjustedIndex).TrimStart();
                        }
                    }

                    // For the last page (if next heading is on this page), stop before the next heading
                    if (nextHeadingPage > 0 && pageNum == nextHeadingPage && nextHeadingIndex > 0)
                    {
                        var adjustedNextIndex = MapIndexToLineBreakText(pageFullText, textWithLineBreaks, nextHeadingIndex);
                        
                        // If this is also the start page, we've already trimmed the beginning
                        if (pageNum == actualStartPage && contentStartIndex > 0)
                        {
                            var adjustedStartIndex = MapIndexToLineBreakText(pageFullText, textWithLineBreaks, contentStartIndex);
                            var relativeNextIndex = adjustedNextIndex - adjustedStartIndex;
                            if (relativeNextIndex > 0 && relativeNextIndex < textToUse.Length)
                            {
                                textToUse = textToUse.Substring(0, relativeNextIndex).Trim();
                            }
                            else if (relativeNextIndex <= 0)
                            {
                                // Next section starts before or at our start, skip this page
                                continue;
                            }
                        }
                        else
                        {
                            if (adjustedNextIndex > 0 && adjustedNextIndex < textWithLineBreaks.Length)
                            {
                                textToUse = textWithLineBreaks.Substring(0, adjustedNextIndex).Trim();
                            }
                        }
                    }
                    else if (nextHeadingPage > 0 && pageNum == nextHeadingPage && nextHeadingIndex == 0)
                    {
                        // Next section starts at the very beginning of this page, don't include it
                        continue;
                    }

                    if (!string.IsNullOrWhiteSpace(textToUse))
                    {
                        pageTextsList.Add(new PageText
                        {
                            PageNumber = pageNum,
                            Text = textToUse
                        });
                    }
                }

                if (pageTextsList.Count > 0)
                {
                    sections.Add(new DocumentSection
                    {
                        HeadingText = bookmark.Title,
                        FullHeadingPath = bookmark.FullTitle,
                        Level = bookmark.Level,
                        StartPage = actualStartPage,
                        EndPage = pageTextsList.Max(p => p.PageNumber),
                        PageTexts = pageTextsList
                    });
                }
            }

            return sections;
        }

        /// <summary>
        /// Finds the end index of a heading in the text, accounting for whitespace variations.
        /// </summary>
        private static int FindHeadingEndIndex(string pageText, string headingTitle, int headingStartIndex)
        {
            if (string.IsNullOrEmpty(pageText) || string.IsNullOrEmpty(headingTitle) || headingStartIndex < 0)
            {
                return headingStartIndex;
            }

            // Try exact match first
            if (headingStartIndex + headingTitle.Length <= pageText.Length)
            {
                var potentialMatch = pageText.Substring(headingStartIndex, Math.Min(headingTitle.Length, pageText.Length - headingStartIndex));
                if (potentialMatch.Equals(headingTitle, StringComparison.OrdinalIgnoreCase))
                {
                    var endIndex = headingStartIndex + headingTitle.Length;
                    // Skip any trailing whitespace after the heading
                    while (endIndex < pageText.Length && char.IsWhiteSpace(pageText[endIndex]))
                    {
                        endIndex++;
                    }
                    return endIndex;
                }
            }

            // If exact match failed, use word-by-word matching (same as FindHeadingInText)
            var normalizedHeading = NormalizeForComparison(headingTitle);
            var headingWords = normalizedHeading.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            
            if (headingWords.Length == 0)
            {
                return headingStartIndex;
            }

            var searchIndex = headingStartIndex;
            var lastWordEndIndex = headingStartIndex;

            foreach (var word in headingWords)
            {
                var foundIndex = pageText.IndexOf(word, searchIndex, StringComparison.OrdinalIgnoreCase);
                if (foundIndex < 0)
                {
                    break;
                }
                lastWordEndIndex = foundIndex + word.Length;
                searchIndex = lastWordEndIndex;
            }

            // Skip any trailing whitespace after the heading
            while (lastWordEndIndex < pageText.Length && char.IsWhiteSpace(pageText[lastWordEndIndex]))
            {
                lastWordEndIndex++;
            }

            return lastWordEndIndex;
        }

        /// <summary>
        /// Maps an index from space-joined text to the corresponding position in line-break text.
        /// This is needed because heading search uses space-joined text, but output uses line-break text.
        /// </summary>
        private static int MapIndexToLineBreakText(string spaceJoinedText, string lineBreakText, int spaceJoinedIndex)
        {
            if (spaceJoinedIndex <= 0)
            {
                return 0;
            }

            if (string.IsNullOrEmpty(spaceJoinedText) || string.IsNullOrEmpty(lineBreakText))
            {
                return spaceJoinedIndex;
            }

            // Count non-whitespace characters up to the index in space-joined text
            var nonWhitespaceCount = 0;
            
            for (int i = 0; i < spaceJoinedIndex && i < spaceJoinedText.Length; i++)
            {
                if (!char.IsWhiteSpace(spaceJoinedText[i]))
                {
                    nonWhitespaceCount++;
                }
            }

            // Find the same position in line-break text by matching non-whitespace character count
            var targetNonWhitespace = nonWhitespaceCount;
            var currentNonWhitespace = 0;
            
            for (int i = 0; i < lineBreakText.Length; i++)
            {
                if (!char.IsWhiteSpace(lineBreakText[i]))
                {
                    currentNonWhitespace++;
                }
                
                if (currentNonWhitespace >= targetNonWhitespace)
                {
                    // Return position after this character, or adjust to start of next word
                    return i + 1;
                }
            }

            return lineBreakText.Length;
        }

        /// <summary>
        /// Find the page and character index where a heading appears.
        /// Searches starting from the expected page and a few pages after.
        /// </summary>
        private static (int Page, int Index) FindHeadingLocation(
            Dictionary<int, string> pageFullTexts,
            string headingTitle,
            int expectedPage,
            int maxPage)
        {
            if (string.IsNullOrEmpty(headingTitle))
            {
                return (-1, -1);
            }

            // Search starting from expected page
            for (int pageNum = expectedPage; pageNum <= maxPage; pageNum++)
            {
                if (!pageFullTexts.TryGetValue(pageNum, out var pageText) || string.IsNullOrEmpty(pageText))
                {
                    continue;
                }

                var index = FindHeadingInText(pageText, headingTitle);
                if (index >= 0)
                {
                    return (pageNum, index);
                }
            }

            // Also search one page before in case bookmark page is slightly off
            if (expectedPage > 1 && pageFullTexts.TryGetValue(expectedPage - 1, out var prevPageText))
            {
                var index = FindHeadingInText(prevPageText, headingTitle);
                if (index >= 0)
                {
                    return (expectedPage - 1, index);
                }
            }

            return (-1, -1);
        }

        /// <summary>
        /// Find the starting index of a heading in the page text.
        /// Uses fuzzy matching to handle minor differences in whitespace/formatting.
        /// </summary>
        private static int FindHeadingInText(string pageText, string headingTitle)
        {
            if (string.IsNullOrEmpty(pageText) || string.IsNullOrEmpty(headingTitle))
            {
                return -1;
            }

            // Normalize the heading title - remove extra whitespace and convert to comparable form
            var normalizedHeading = NormalizeForComparison(headingTitle);

            if (string.IsNullOrEmpty(normalizedHeading))
            {
                return -1;
            }

            // Try exact match first (case-insensitive)
            var exactIndex = pageText.IndexOf(headingTitle, StringComparison.OrdinalIgnoreCase);
            if (exactIndex >= 0)
            {
                return exactIndex;
            }

            // Try matching with normalized whitespace
            // Build a pattern that matches the heading words with flexible whitespace
            var headingWords = normalizedHeading.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (headingWords.Length == 0)
            {
                return -1;
            }

            // Search for the sequence of words in the page text
            var searchIndex = 0;
            var firstWordIndex = -1;
            var wordIndex = 0;

            while (searchIndex < pageText.Length && wordIndex < headingWords.Length)
            {
                var word = headingWords[wordIndex];
                var foundIndex = pageText.IndexOf(word, searchIndex, StringComparison.OrdinalIgnoreCase);

                if (foundIndex < 0)
                {
                    // Word not found, heading not present
                    return -1;
                }

                if (wordIndex == 0)
                {
                    firstWordIndex = foundIndex;
                }
                else
                {
                    // Check that this word follows reasonably close to the previous match
                    // Allow for some whitespace and punctuation between words
                    var gap = foundIndex - searchIndex;
                    if (gap > word.Length + 10) // Allow reasonable gap
                    {
                        // Gap too large, restart search from after the first word
                        searchIndex = firstWordIndex + headingWords[0].Length;
                        wordIndex = 0;
                        firstWordIndex = -1;
                        continue;
                    }
                }

                searchIndex = foundIndex + word.Length;
                wordIndex++;
            }

            return firstWordIndex;
        }

        private static string NormalizeForComparison(string text)
        {
            if (string.IsNullOrEmpty(text))
            {
                return string.Empty;
            }

            // Replace multiple whitespace with single space and trim
            return System.Text.RegularExpressions.Regex.Replace(text.Trim(), @"\s+", " ");
        }

        private sealed class FlattenedBookmark
        {
            public string Title { get; init; } = string.Empty;
            public string FullTitle { get; init; } = string.Empty;
            public int PageNumber { get; init; }
            public int Level { get; init; }
            public double? YPosition { get; init; }
        }

        private sealed class PageWordInfo
        {
            public int PageNumber { get; init; }
            public List<PdfTextUtilities.WordPosition> Words { get; init; } = [];
            public string FullText { get; init; } = string.Empty;
        }
    }
}

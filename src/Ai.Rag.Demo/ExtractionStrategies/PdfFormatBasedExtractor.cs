using Ai.Rag.Demo.Models;
using Ai.Rag.Demo.Services;
using UglyToad.PdfPig;
using UglyToad.PdfPig.Graphics.Colors;

namespace Ai.Rag.Demo.ExtractionStrategies;

/// <summary>
/// Extraction strategy for processing PDFs by detecting sections based on font characteristics
/// (size, style, color) and creating sections accordingly
/// </summary>
public sealed class PdfFormatBasedExtractor : IExtractionStrategy
{
    private readonly int _maxHeadingDepth;
    private readonly double? _maxHeadingFontSize;
    private readonly int _skipPages;
    private readonly string? _headingColorKey;

    /// <summary>
    /// Creates a new format-based PDF extractor
    /// </summary>
    /// <param name="maxHeadingDepth">Maximum heading depth to consider for section breaks (1 = only top-level headings, 2 = top + second level, etc.)</param>
    /// <param name="maxHeadingFontSize">Maximum font size to consider as a heading (larger text is ignored). Null means no upper limit.</param>
    /// <param name="skipPages">Number of pages to skip from the beginning of the document</param>
    /// <param name="headingColorKey">Specific color key to look for in headings (e.g., "0.00_0.00_1.00" for blue RGB). Null means any color.</param>
    public PdfFormatBasedExtractor(
        int maxHeadingDepth = 1,
        double? maxHeadingFontSize = null,
        int skipPages = 0,
        string? headingColorKey = null)
    {
        if (maxHeadingDepth < 1)
        {
            throw new ArgumentException("Max heading depth must be at least 1", nameof(maxHeadingDepth));
        }

        if (maxHeadingFontSize.HasValue && maxHeadingFontSize.Value <= 0)
        {
            throw new ArgumentException("Max heading font size must be greater than 0", nameof(maxHeadingFontSize));
        }

        if (skipPages < 0)
        {
            throw new ArgumentException("Skip pages cannot be negative", nameof(skipPages));
        }

        _maxHeadingDepth = maxHeadingDepth;
        _maxHeadingFontSize = maxHeadingFontSize;
        _skipPages = skipPages;
        _headingColorKey = headingColorKey;
    }

    /// <summary>
    /// Extract document sections from a PDF stream based on font characteristics
    /// </summary>
    public Task<List<DocumentSection>> ExtractAsync(Stream file)
    {
        using var document = PdfDocument.Open(file);
        var sections = ExtractSections(document);
        return Task.FromResult(sections);
    }

    private List<DocumentSection> ExtractSections(PdfDocument document)
    {
        var textBlocks = new List<TextBlock>();

        // First pass: collect all text blocks with font information
        foreach (var page in document.GetPages())
        {
            var pageNumber = page.Number;
            if (pageNumber <= _skipPages)
            {
                continue;
            }

            var words = page.GetWords().ToList();

            foreach (var word in words)
            {
                var letters = word.Letters.ToList();
                if (letters.Count == 0) continue;

                // Get font characteristics from the first letter (representative)
                var firstLetter = letters[0];
                var fontSize = firstLetter.PointSize;
                if(_maxHeadingFontSize.HasValue && fontSize > _maxHeadingFontSize.Value)
                {
                    continue;
                }
                var fontName = firstLetter.FontName ?? string.Empty;
                var color = GetColorKey(firstLetter.Color);
                var isBold = IsBoldFont(fontName);
                var isItalic = IsItalicFont(fontName);

                textBlocks.Add(new TextBlock
                {
                    Text = word.Text,
                    FontSize = fontSize,
                    FontName = fontName,
                    IsBold = isBold,
                    IsItalic = isItalic,
                    ColorKey = color,
                    PageNumber = pageNumber,
                    Top = word.BoundingBox.Top,
                    Bottom = word.BoundingBox.Bottom
                });
            }
        }

        if (textBlocks.Count == 0)
        {
            return [];
        }

        // Calculate font statistics to determine what constitutes a heading
        var fontStats = CalculateFontStatistics(textBlocks);


        // Second pass: identify section boundaries and group text
        return IdentifySections(textBlocks, fontStats);
    }

    private static string GetColorKey(IColor color)
    {
        if (color == null)
        {
            return "default";
        }

        // Convert color to RGB if possible and create a key
        return color switch
        {
            RGBColor rgb => $"{rgb.R:F2}_{rgb.G:F2}_{rgb.B:F2}",
            GrayColor gray => $"gray_{gray.Gray:F2}",
            CMYKColor cmyk => $"cmyk_{cmyk.C:F2}_{cmyk.M:F2}_{cmyk.Y:F2}_{cmyk.K:F2}",
            _ => color.ToString() ?? "default"
        };
    }

    private static bool IsBoldFont(string fontName)
    {
        var lowerName = fontName.ToLowerInvariant();
        return lowerName.Contains("bold") ||
               lowerName.Contains("heavy") ||
               lowerName.Contains("black") ||
               lowerName.Contains("demi");
    }

    private static bool IsItalicFont(string fontName)
    {
        var lowerName = fontName.ToLowerInvariant();
        return lowerName.Contains("italic") ||
               lowerName.Contains("oblique") ||
               lowerName.Contains("slant");
    }

    private FontStatistics CalculateFontStatistics(List<TextBlock> textBlocks)
    {
        var fontSizes = textBlocks.Select(b => b.FontSize).ToList();
        var avgFontSize = fontSizes.Average();
        var maxFontSize = fontSizes.Max();
        var minFontSize = fontSizes.Min();

        // Find the most common font size (body text)
        var bodyFontSize = fontSizes
            .GroupBy(s => Math.Round(s, 1))
            .OrderByDescending(g => g.Count())
            .First()
            .Key;

        // Find distinct heading font sizes (larger than body)
        // If a specific color is required for headings, only consider blocks with that color
        var headingFontSizesQuery = textBlocks
            .Where(b => b.FontSize > bodyFontSize * 1.1);

        // Filter by color if specified
        if (!string.IsNullOrEmpty(_headingColorKey))
        {
            headingFontSizesQuery = headingFontSizesQuery.Where(b => b.ColorKey == _headingColorKey);
        }

        // Apply max heading font size filter if specified
        if (_maxHeadingFontSize.HasValue)
        {
            headingFontSizesQuery = headingFontSizesQuery.Where(b => b.FontSize <= _maxHeadingFontSize.Value);
        }

        var headingFontSizes = headingFontSizesQuery
            .Select(b => Math.Round(b.FontSize, 1))
            .Distinct()
            .OrderByDescending(s => s)
            .ToList();

        return new FontStatistics
        {
            AverageFontSize = avgFontSize,
            MaxFontSize = maxFontSize,
            MinFontSize = minFontSize,
            BodyFontSize = bodyFontSize,
            HeadingFontSizes = headingFontSizes
        };
    }

    private List<DocumentSection> IdentifySections(List<TextBlock> textBlocks, FontStatistics fontStats)
    {
        var sections = new List<DocumentSection>();
        var currentSection = new DocumentSection();
        var currentPageTexts = new Dictionary<int, System.Text.StringBuilder>();
        var currentHeadingText = new System.Text.StringBuilder();
        var lastPageNumber = 0;
        var isCollectingHeading = false;
        var headingFontSize = 0.0;
        var headingIsBold = false;
        var headingStartPage = 0;
        var headingLevel = 0;
        TextBlock? lastBodyBlock = null;
        
        // Track parent headings by level for building hierarchical paths
        var headingsByLevel = new Dictionary<int, string>();

        foreach (var block in textBlocks)
        {
            var blockHeadingLevel = GetHeadingLevel(block, fontStats);
            var isHeading = blockHeadingLevel > 0 && blockHeadingLevel <= _maxHeadingDepth;

            if (isHeading)
            {
                // Check if this is a continuation of the current heading (same font characteristics)
                var isContinuationOfHeading = isCollectingHeading && 
                                              Math.Abs(block.FontSize - headingFontSize) < 0.5 &&
                                              block.PageNumber == lastPageNumber;

                if (isContinuationOfHeading)
                {
                    // Continue collecting the heading (add space for new line or between words)
                    currentHeadingText.Append(' ');
                    currentHeadingText.Append(block.Text);
                }
                else
                {
                    // New heading detected - finalize previous section if we have content
                    if (currentPageTexts.Count > 0)
                    {
                        currentSection.PageTexts = currentPageTexts
                            .OrderBy(kvp => kvp.Key)
                            .Select(kvp => new PageText { PageNumber = kvp.Key, Text = kvp.Value.ToString().Trim() })
                            .Where(pt => !string.IsNullOrWhiteSpace(pt.Text))
                            .ToList();
                        
                        if (currentSection.PageTexts.Count > 0)
                        {
                            currentSection.EndPage = currentSection.PageTexts.Max(p => p.PageNumber);
                            sections.Add(currentSection);
                        }
                        currentPageTexts.Clear();
                    }

                    // Start collecting new heading
                    isCollectingHeading = true;
                    currentHeadingText.Clear();
                    currentHeadingText.Append(block.Text);
                    headingFontSize = block.FontSize;
                    headingIsBold = block.IsBold;
                    headingStartPage = block.PageNumber;
                    headingLevel = blockHeadingLevel;

                    // Start new section
                    currentSection = new DocumentSection
                    {
                        StartPage = block.PageNumber,
                        Level = headingLevel
                    };
                }
                
                // Reset lastBodyBlock when starting/continuing a heading
                lastBodyBlock = null;
            }
            else
            {
                // Non-heading text - finalize heading if we were collecting one
                if (isCollectingHeading)
                {
                    var headingText = currentHeadingText.ToString().Trim();
                    currentSection.HeadingText = headingText;
                    currentSection.HeadingFontSize = headingFontSize;
                    currentSection.HeadingIsBold = headingIsBold;
                    currentSection.StartPage = headingStartPage;
                    currentSection.Level = headingLevel;
                    
                    // Update heading hierarchy - clear all levels deeper than current
                    headingsByLevel[headingLevel] = headingText;
                    var levelsToRemove = headingsByLevel.Keys.Where(l => l > headingLevel).ToList();
                    foreach (var level in levelsToRemove)
                    {
                        headingsByLevel.Remove(level);
                    }
                    
                    // Build full heading path from parent levels
                    currentSection.FullHeadingPath = BuildFullHeadingPath(headingsByLevel, headingLevel);
                    
                    isCollectingHeading = false;
                    lastBodyBlock = null;
                }

                // Ensure page entry exists
                if (!currentPageTexts.ContainsKey(block.PageNumber))
                {
                    currentPageTexts[block.PageNumber] = new System.Text.StringBuilder();
                }

                // Add body text to appropriate page with line break detection
                if (currentPageTexts[block.PageNumber].Length > 0)
                {
                    // Determine if we should insert a line break or a space
                    if (lastBodyBlock != null && lastBodyBlock.PageNumber == block.PageNumber)
                    {
                        var previousWordPosition = new PdfTextUtilities.WordPosition
                        {
                            Text = lastBodyBlock.Text,
                            Top = lastBodyBlock.Top,
                            Bottom = lastBodyBlock.Bottom
                        };
                        var currentWordPosition = new PdfTextUtilities.WordPosition
                        {
                            Text = block.Text,
                            Top = block.Top,
                            Bottom = block.Bottom
                        };

                        if (PdfTextUtilities.ShouldInsertLineBreak(previousWordPosition, currentWordPosition))
                        {
                            currentPageTexts[block.PageNumber].AppendLine();
                        }
                        else
                        {
                            currentPageTexts[block.PageNumber].Append(' ');
                        }
                    }
                    else
                    {
                        // Different page or first block after heading, use space
                        currentPageTexts[block.PageNumber].Append(' ');
                    }
                }
                currentPageTexts[block.PageNumber].Append(block.Text);
                lastBodyBlock = block;
            }

            lastPageNumber = block.PageNumber;
        }

        // Finalize last heading if still collecting
        if (isCollectingHeading)
        {
            var headingText = currentHeadingText.ToString().Trim();
            currentSection.HeadingText = headingText;
            currentSection.HeadingFontSize = headingFontSize;
            currentSection.HeadingIsBold = headingIsBold;
            currentSection.StartPage = headingStartPage;
            currentSection.Level = headingLevel;
            
            // Update heading hierarchy
            headingsByLevel[headingLevel] = headingText;
            var levelsToRemove = headingsByLevel.Keys.Where(l => l > headingLevel).ToList();
            foreach (var level in levelsToRemove)
            {
                headingsByLevel.Remove(level);
            }
            
            // Build full heading path from parent levels
            currentSection.FullHeadingPath = BuildFullHeadingPath(headingsByLevel, headingLevel);
        }

        // Don't forget the last section
        if (currentPageTexts.Count > 0)
        {
            currentSection.PageTexts = currentPageTexts
                .OrderBy(kvp => kvp.Key)
                .Select(kvp => new PageText { PageNumber = kvp.Key, Text = kvp.Value.ToString().Trim() })
                .Where(pt => !string.IsNullOrWhiteSpace(pt.Text))
                .ToList();
            
            if (currentSection.PageTexts.Count > 0)
            {
                currentSection.EndPage = currentSection.PageTexts.Max(p => p.PageNumber);
                sections.Add(currentSection);
            }
        }

        return sections;
    }

    private static string BuildFullHeadingPath(Dictionary<int, string> headingsByLevel, int currentLevel)
    {
        var pathParts = headingsByLevel
            .Where(kvp => kvp.Key <= currentLevel)
            .OrderBy(kvp => kvp.Key)
            .Select(kvp => kvp.Value)
            .ToList();
        
        return string.Join(" > ", pathParts);
    }

    private int GetHeadingLevel(TextBlock block, FontStatistics fontStats)
    {
        // If font size exceeds max heading font size, it's not a heading (e.g., title page)
        if (_maxHeadingFontSize.HasValue && block.FontSize > _maxHeadingFontSize.Value)
        {
            return 0;
        }

        // If a specific heading color is required, check if this block matches
        if (!string.IsNullOrEmpty(_headingColorKey) && block.ColorKey != _headingColorKey)
        {
            return 0;
        }

        var roundedSize = Math.Round(block.FontSize, 1);
        
        // Check if this font size is in our heading font sizes list (sorted largest to smallest)
        // Filter out sizes above maxHeadingFontSize when determining index
        var eligibleHeadingSizes = _maxHeadingFontSize.HasValue
            ? fontStats.HeadingFontSizes.Where(s => s <= _maxHeadingFontSize.Value).ToList()
            : fontStats.HeadingFontSizes;

        var index = eligibleHeadingSizes.IndexOf(roundedSize);
        if (index >= 0)
        {
            // Level 1 is the largest heading, level 2 is second largest, etc.
            return index + 1;
        }

        // Check if it's bold with body font size (treat as lowest level heading)
        if (block.IsBold && block.FontSize >= fontStats.BodyFontSize)
        {
            // Bold body text is the lowest priority heading level
            return eligibleHeadingSizes.Count + 1;
        }

        // Not a heading
        return 0;
    }

    private sealed class TextBlock
    {
        public string Text { get; init; } = string.Empty;
        public double FontSize { get; init; }
        public string FontName { get; init; } = string.Empty;
        public bool IsBold { get; init; }
        public bool IsItalic { get; init; }
        public string ColorKey { get; init; } = string.Empty;
        public int PageNumber { get; init; }
        public double Top { get; init; }
        public double Bottom { get; init; }
    }

    private sealed class FontStatistics
    {
        public double AverageFontSize { get; init; }
        public double MaxFontSize { get; init; }
        public double MinFontSize { get; init; }
        public double BodyFontSize { get; init; }
        public List<double> HeadingFontSizes { get; init; } = [];
    }
}

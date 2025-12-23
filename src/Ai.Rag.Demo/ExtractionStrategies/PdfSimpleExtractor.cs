using Ai.Rag.Demo.Models;
using Ai.Rag.Demo.Services;
using UglyToad.PdfPig;

namespace Ai.Rag.Demo.ExtractionStrategies;

/// <summary>
/// A simple PDF extraction strategy that extracts all content into a single section.
/// </summary>
public sealed class PdfSimpleExtractor : IExtractionStrategy
{
    private readonly int _skipPages;

    /// <summary>
    /// Creates a new instance of PdfSimpleExtractor.
    /// </summary>
    /// <param name="skipPages">Number of pages to skip from the beginning of the document.</param>
    public PdfSimpleExtractor(int skipPages = 0)
    {
        if (skipPages < 0)
        {
            throw new ArgumentException("Skip pages cannot be negative", nameof(skipPages));
        }

        _skipPages = skipPages;
    }

    public Task<List<DocumentSection>> ExtractAsync(Stream file)
    {
        using var document = PdfDocument.Open(file);

        var pageTexts = new List<PageText>();
        var startPage = 0;
        var endPage = document.NumberOfPages;

        foreach (var page in document.GetPages())
        {
            if (page.Number <= _skipPages)
            {
                continue;
            }

            if (startPage == 0)
            {
                startPage = page.Number;
            }

            var wordPositions = PdfTextUtilities.GetWordPositions(page.GetWords());
            var pageText = PdfTextUtilities.BuildTextWithLineBreaks(wordPositions);

            if (!string.IsNullOrWhiteSpace(pageText))
            {
                pageTexts.Add(new PageText
                {
                    PageNumber = page.Number,
                    Text = pageText
                });
            }
        }

        var sections = new List<DocumentSection>();

        if (pageTexts.Count > 0)
        {
            sections.Add(new DocumentSection
            {
                Level = 1,
                StartPage = startPage,
                EndPage = endPage,
                PageTexts = pageTexts
            });
        }

        return Task.FromResult(sections);
    }
}

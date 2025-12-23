namespace Ai.Rag.Demo.Models;

/// <summary>
/// Represents a section of a document with its heading, pages, and text content
/// </summary>
public sealed class DocumentSection
{
    /// <summary>
    /// The section heading or title
    /// </summary>
    public string HeadingText { get; set; } = string.Empty;
    
    /// <summary>
    /// The full hierarchical heading path (e.g., "Chapter 1 > Section 1.1")
    /// Only populated when different from HeadingText
    /// </summary>
    public string FullHeadingPath { get; set; } = string.Empty;
    
    /// <summary>
    /// Font size of the heading (for font-based extraction)
    /// </summary>
    public double HeadingFontSize { get; set; }
    
    /// <summary>
    /// Whether the heading is bold (for font-based extraction)
    /// </summary>
    public bool HeadingIsBold { get; set; }
    
    /// <summary>
    /// The hierarchical level of this section (1 = top level, 2 = second level, etc.)
    /// </summary>
    public int Level { get; set; } = 1;
    
    /// <summary>
    /// The first page where this section starts
    /// </summary>
    public int StartPage { get; set; } = 1;
    
    /// <summary>
    /// The last page where this section ends
    /// </summary>
    public int EndPage { get; set; } = 1;
    
    /// <summary>
    /// Text content organized by page
    /// </summary>
    public List<PageText> PageTexts { get; set; } = [];
    
    /// <summary>
    /// The full text of this section (all pages combined)
    /// </summary>
    public string Text => string.Join(" ", PageTexts.Select(p => p.Text));
}

/// <summary>
/// Represents text content from a specific page
/// </summary>
public sealed class PageText
{
    /// <summary>
    /// The page number
    /// </summary>
    public int PageNumber { get; init; }
    
    /// <summary>
    /// The text content from this page
    /// </summary>
    public string Text { get; set; } = string.Empty;
}

using System.Text;

namespace Ai.Rag.Demo.Models;

/// <summary>
/// Represents a chunk of text with its embedding and metadata
/// </summary>
public class DocumentChunk
{
    public required string Id { get; set; }
    public required string Text { get; set; }
    public float[] Embedding { get; set; }
    public required string SourceDocument { get; set; }
    public int StartPage { get; set; }
    public int EndPage { get; set; }
    public int ChunkIndex { get; set; }
    public int ChunkTotal { get; set; }

    /// <summary>
    /// The section title or heading this chunk belongs to
    /// </summary>
    public string Section { get; set; }

    /// <summary>
    /// The full hierarchical section path (e.g., "Chapter 1 > Section 1.1 > Subsection 1.1.1")
    /// </summary>
    public string SectionPath { get; set; }

    public Dictionary<string, string> Metadata { get; set; } = new();

    public string GenerateEmbeddingText()
    {
        var header = BuildPathHeader();
        var embeddingText = new StringBuilder(header);
        if(header.Length > 0)
        {
            embeddingText.AppendLine();
        }
        embeddingText.Append(Text);
        return embeddingText.ToString();
    }

    private string BuildPathHeader()
    {
        var sb = new StringBuilder();
        if (!string.IsNullOrEmpty(SourceDocument))
        {
            sb.Append("Document: ");
            sb.Append(SourceDocument);
            sb.AppendLine();
        }
        if (!string.IsNullOrEmpty(SectionPath))
        {
            sb.Append("Section: ");
            sb.Append(SectionPath);
            sb.AppendLine();
        }
        else if (!string.IsNullOrEmpty(Section))
        {
            sb.Append("Section: ");
            sb.Append(Section);
            sb.AppendLine();
        }
        
        return sb.ToString();
    }
}

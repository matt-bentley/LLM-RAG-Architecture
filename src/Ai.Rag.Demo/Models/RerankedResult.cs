namespace Ai.Rag.Demo.Models;

/// <summary>
/// Represents a search result that has been re-ranked by the LLM
/// </summary>
public class RerankedResult
{
    /// <summary>
    /// The original chunk metadata
    /// </summary>
    public required DocumentChunk Chunk { get; set; }

    /// <summary>
    /// The relevance score assigned by the reranker (0-10)
    /// </summary>
    public double RelevanceScore { get; set; }

    /// <summary>
    /// The original position in the search results before re-ranking
    /// </summary>
    public int OriginalRank { get; set; }

    /// <summary>
    /// The new position after re-ranking
    /// </summary>
    public int NewRank { get; set; }
}

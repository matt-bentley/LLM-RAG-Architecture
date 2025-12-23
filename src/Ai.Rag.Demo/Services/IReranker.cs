using Ai.Rag.Demo.Models;

namespace Ai.Rag.Demo.Services;

/// <summary>
/// Interface for re-ranking search results using an LLM
/// </summary>
public interface IReranker
{
    /// <summary>
    /// Re-ranks the given search results based on relevance to the query using an LLM
    /// </summary>
    /// <param name="query">The original search query</param>
    /// <param name="results">The search results from the embedding store</param>
    /// <param name="topK">Number of top results to return after re-ranking</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Re-ranked results with relevance scores</returns>
    Task<List<RerankedResult>> RerankAsync(
        string query,
        List<DocumentChunk> results,
        int topK = 5,
        CancellationToken cancellationToken = default);
}

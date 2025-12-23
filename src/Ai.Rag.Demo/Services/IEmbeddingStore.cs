using Ai.Rag.Demo.Models;

namespace Ai.Rag.Demo.Services;

/// <summary>
/// Interface for storing and retrieving embedding indexes
/// </summary>
public interface IEmbeddingStore
{
    
    /// <summary>
    /// Store multiple chunks with their embeddings
    /// </summary>
    Task StoreChunksAsync(IEnumerable<DocumentChunk> chunks, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Search for similar chunks using cosine similarity
    /// </summary>
    Task<List<DocumentChunk>> SearchAsync(string query, float[] queryEmbedding, int topK = 5, CancellationToken cancellationToken = default);

    /// <summary>
    /// Search for similar chunks and include adjacent chunks for better context
    /// </summary>
    /// <param name="query">The search query text</param>
    /// <param name="queryEmbedding">The query embedding vector</param>
    /// <param name="topK">Number of top results to return</param>
    /// <param name="adjacentChunkCount">Number of chunks before and after each result to include</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>List of matching chunks including adjacent chunks</returns>
    Task<List<DocumentChunk>> SearchWithAdjacentChunksAsync(
        string query,
        float[] queryEmbedding,
        int topK = 5,
        int adjacentChunkCount = 1,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Retrieves specific chunks by document name and chunk indices
    /// </summary>
    /// <param name="documentName">The source document name</param>
    /// <param name="chunkIndices">The chunk indices to retrieve</param>
    /// <param name="sectionPath">The section path to filter by (optional)</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>List of matching chunks</returns>
    Task<List<DocumentChunk>> GetChunksByDocumentAndIndicesAsync(
        string documentName,
        IEnumerable<int> chunkIndices,
        string sectionPath = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Delete all chunks from a specific document
    /// </summary>
    Task DeleteDocumentAsync(string documentName, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Clear all stored chunks
    /// </summary>
    Task ClearAsync(CancellationToken cancellationToken = default);
}

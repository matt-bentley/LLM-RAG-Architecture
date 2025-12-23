using Ai.Rag.Demo.Models;

namespace Ai.Rag.Demo.EmbeddingGenerators;

/// <summary>
/// Interface for generating text embeddings
/// </summary>
public interface IEmbeddingGenerator
{
    /// <summary>
    /// Generate an embedding vector for the given text
    /// </summary>
    Task<float[]> GenerateEmbeddingAsync(string text, CancellationToken cancellationToken = default);

    /// <summary>
    /// Generate embedding vectors for multiple texts at once
    /// </summary>
    Task<IReadOnlyList<float[]>> GenerateEmbeddingsAsync(List<DocumentChunk> chunks, CancellationToken cancellationToken = default);
}

using Ai.Rag.Demo.Models;

namespace Ai.Rag.Demo.DocumentExtraction;

/// <summary>
/// Interface for document processing services that extract text and create chunks
/// </summary>
public interface IDocumentExtractor
{
    /// <summary>
    /// Process a document file: extract text and create chunks
    /// </summary>
    Task<List<DocumentChunk>> ProcessDocumentAsync(string filePath, IExtractionStrategy extractionStrategy, CancellationToken cancellationToken = default);

    /// <summary>
    /// Process multiple document files: extract text and create chunks for each
    /// </summary>
    Task<Dictionary<string, List<DocumentChunk>>> ProcessDocumentsAsync(
        IEnumerable<string> filePaths,
        IExtractionStrategy extractionStrategy,
        CancellationToken cancellationToken = default);
}

public interface IExtractionStrategy
{
    Task<List<DocumentSection>> ExtractAsync(Stream file);
}

using System.Numerics.Tensors;
using System.Text.Json;
using Ai.Rag.Demo.Models;
using Ai.Rag.Demo.Services;

namespace Ai.Rag.Demo.EmbeddingStores;

/// <summary>
/// File system-based implementation of embedding storage
/// </summary>
public class FileSystemEmbeddingStore : IEmbeddingStore
{
    private readonly string _storageDirectory;
    private readonly string _indexFilePath;
    private readonly JsonSerializerOptions _jsonOptions;

    public FileSystemEmbeddingStore(string storageDirectory)
    {
        _storageDirectory = storageDirectory;
        _indexFilePath = Path.Combine(_storageDirectory, "index.json");
        _jsonOptions = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        Directory.CreateDirectory(_storageDirectory);
    }

    public async Task StoreChunksAsync(IEnumerable<DocumentChunk> chunks, CancellationToken cancellationToken = default)
    {
        var existingChunks = await LoadAllChunksAsync(cancellationToken);
        
        foreach (var chunk in chunks)
        {
            // Remove existing chunk with same ID if present
            existingChunks.RemoveAll(c => c.Id == chunk.Id);
            existingChunks.Add(chunk);
        }

        await SaveAllChunksAsync(existingChunks, cancellationToken);
    }

    public async Task<List<DocumentChunk>> SearchAsync(string query, float[] queryEmbedding, int topK = 5, CancellationToken cancellationToken = default)
    {
        var chunks = await LoadAllChunksAsync(cancellationToken);
        
        if (chunks.Count == 0)
        {
            return new List<DocumentChunk>();
        }

        // Calculate cosine similarity for each chunk
        var similarities = chunks.Select(chunk => new
        {
            Chunk = chunk,
            Similarity = CosineSimilarity(queryEmbedding, chunk.Embedding)
        })
        .OrderByDescending(x => x.Similarity)
        .Take(topK)
        .Select(x => x.Chunk)
        .ToList();

        return similarities;
    }

    /// <inheritdoc />
    public async Task<List<DocumentChunk>> SearchWithAdjacentChunksAsync(
        string query,
        float[] queryEmbedding,
        int topK = 5,
        int adjacentChunkCount = 1,
        CancellationToken cancellationToken = default)
    {
        var results = await SearchAsync(query, queryEmbedding, topK, cancellationToken);

        if (adjacentChunkCount <= 0 || results.Count == 0)
        {
            return results;
        }

        return await ExpandWithAdjacentChunksAsync(results, adjacentChunkCount, cancellationToken);
    }

    public async Task DeleteDocumentAsync(string documentName, CancellationToken cancellationToken = default)
    {
        var chunks = await LoadAllChunksAsync(cancellationToken);
        chunks.RemoveAll(c => c.SourceDocument == documentName);
        await SaveAllChunksAsync(chunks, cancellationToken);
    }

    public async Task ClearAsync(CancellationToken cancellationToken = default)
    {
        await SaveAllChunksAsync(new List<DocumentChunk>(), cancellationToken);
    }

    /// <inheritdoc />
    public async Task<List<DocumentChunk>> GetChunksByDocumentAndIndicesAsync(
        string documentName,
        IEnumerable<int> chunkIndices,
        string? sectionPath = null,
        CancellationToken cancellationToken = default)
    {
        var indicesSet = chunkIndices.ToHashSet();
        if (indicesSet.Count == 0)
        {
            return [];
        }

        var chunks = await LoadAllChunksAsync(cancellationToken);

        return chunks
            .Where(c => c.SourceDocument == documentName && indicesSet.Contains(c.ChunkIndex))
            .ToList();
    }

    private async Task<List<DocumentChunk>> ExpandWithAdjacentChunksAsync(
        List<DocumentChunk> results,
        int adjacentChunkCount,
        CancellationToken cancellationToken)
    {
        var chunksToFetch = new HashSet<ChunkIdentifier>();

        foreach (var result in results)
        {
            for (int offset = -adjacentChunkCount; offset <= adjacentChunkCount; offset++)
            {
                if (offset == 0) continue; // Skip the chunk itself

                var adjacentIndex = result.ChunkIndex + offset;
                if (adjacentIndex < 0
                    || (adjacentIndex + 1) > result.ChunkTotal)
                {
                    // skip outside bounds of chunk indices
                    continue;
                }

                var chunkAlreadyInResults = results.Any(e => e.SourceDocument == result.SourceDocument
                                                            && e.ChunkIndex == adjacentIndex
                                                            && e.SectionPath == result.SectionPath);
                if (!chunkAlreadyInResults)
                {
                    var chunkIdentifier = new ChunkIdentifier(result.SourceDocument, result.SectionPath, adjacentIndex);
                    chunksToFetch.Add(chunkIdentifier);
                }
            }
        }

        if (chunksToFetch.Count == 0)
        {
            return results;
        }

        var expandedResults = new List<DocumentChunk>(results);

        foreach (var chunks in chunksToFetch.GroupBy(e => new { e.SourceDocument, e.SectionPath }))
        {
            var adjacentChunks = await GetChunksByDocumentAndIndicesAsync(chunks.Key.SourceDocument, chunks.Select(c => c.ChunkIndex), chunks.Key.SectionPath, cancellationToken);
            expandedResults.AddRange(adjacentChunks);
        }

        return expandedResults;
    }

    private async Task<List<DocumentChunk>> LoadAllChunksAsync(CancellationToken cancellationToken)
    {
        if (!File.Exists(_indexFilePath))
        {
            return [];
        }

        var json = await File.ReadAllTextAsync(_indexFilePath, cancellationToken);
        return JsonSerializer.Deserialize<List<DocumentChunk>>(json, _jsonOptions) ?? [];
    }

    private async Task SaveAllChunksAsync(List<DocumentChunk> chunks, CancellationToken cancellationToken)
    {
        var json = JsonSerializer.Serialize(chunks, _jsonOptions);
        await File.WriteAllTextAsync(_indexFilePath, json, cancellationToken);
    }

    private static float CosineSimilarity(float[] vector1, float[] vector2)
    {
        if (vector1.Length != vector2.Length)
        {
            throw new ArgumentException("Vectors must have the same length");
        }

        var dotProduct = TensorPrimitives.Dot(vector1.AsSpan(), vector2.AsSpan());
        var magnitude1 = TensorPrimitives.Norm(vector1.AsSpan());
        var magnitude2 = TensorPrimitives.Norm(vector2.AsSpan());

        if (magnitude1 == 0 || magnitude2 == 0)
        {
            return 0;
        }

        return dotProduct / (magnitude1 * magnitude2);
    }
}

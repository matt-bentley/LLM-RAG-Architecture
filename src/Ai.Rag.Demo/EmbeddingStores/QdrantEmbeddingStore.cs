using Ai.Rag.Demo.Models;
using Ai.Rag.Demo.Services;
using Qdrant.Client;
using Qdrant.Client.Grpc;

namespace Ai.Rag.Demo.EmbeddingStores;

/// <summary>
/// Qdrant-based implementation of embedding storage for vector search
/// </summary>
public class QdrantEmbeddingStore : IEmbeddingStore
{
    private readonly QdrantClient _client;
    private readonly string _collectionName;
    private readonly int _vectorSize;

    public QdrantEmbeddingStore(string host = "localhost", int port = 6334, string collectionName = "documents", int vectorSize = 1536)
    {
        _client = new QdrantClient(host, port);
        _collectionName = collectionName;
        _vectorSize = vectorSize;

        InitializeCollectionAsync().GetAwaiter().GetResult();
    }

    public QdrantEmbeddingStore(QdrantClient client, string collectionName = "documents", int vectorSize = 1536)
    {
        _client = client;
        _collectionName = collectionName;
        _vectorSize = vectorSize;

        InitializeCollectionAsync().GetAwaiter().GetResult();
    }

    private async Task InitializeCollectionAsync()
    {
        try
        {
            // Check if collection exists
            var collections = await _client.ListCollectionsAsync();
            var collectionExists = collections.Contains(_collectionName);

            if (!collectionExists)
            {
                // Create collection with cosine similarity
                await _client.CreateCollectionAsync(
                    collectionName: _collectionName,
                    vectorsConfig: new VectorParams
                    {
                        Size = (ulong)_vectorSize,
                        Distance = Distance.Cosine
                    });

                // Create payload indexes for efficient filtering
                await _client.CreatePayloadIndexAsync(
                    collectionName: _collectionName,
                    fieldName: "sourceDocument",
                    schemaType: PayloadSchemaType.Keyword);

                await _client.CreatePayloadIndexAsync(
                    collectionName: _collectionName,
                    fieldName: "sectionPath",
                    schemaType: PayloadSchemaType.Keyword);

                await _client.CreatePayloadIndexAsync(
                    collectionName: _collectionName,
                    fieldName: "chunkIndex",
                    schemaType: PayloadSchemaType.Integer);
            }
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to initialize Qdrant collection '{_collectionName}': {ex.Message}", ex);
        }
    }

    public async Task StoreChunksAsync(IEnumerable<DocumentChunk> chunks, CancellationToken cancellationToken = default)
    {
        var chunkList = chunks.ToList();
        if (chunkList.Count == 0)
        {
            return;
        }

        var points = chunkList.Select(chunk => chunk.ToPointStruct()).ToList();

        await _client.UpsertAsync(
            collectionName: _collectionName,
            points: points,
            cancellationToken: cancellationToken);
    }

    public async Task<List<DocumentChunk>> SearchAsync(string query, float[] queryEmbedding, int topK = 5, CancellationToken cancellationToken = default)
    {
        var searchResults = await _client.SearchAsync(
            collectionName: _collectionName,
            vector: queryEmbedding,
            limit: (ulong)topK,
            //vectorsSelector: new WithVectorsSelector { Enable = true }, // by default the embeddings are not returned
            cancellationToken: cancellationToken);

        var chunks = searchResults.Select(result => result.Payload.ToDocumentChunk())
                                  .ToList();

        return chunks;
    }

    public async Task DeleteDocumentAsync(string documentName, CancellationToken cancellationToken = default)
    {
        // Delete all points where sourceDocument matches
        await _client.DeleteAsync(
            collectionName: _collectionName,
            filter: new Filter
            {
                Must =
                {
                    new Condition
                    {
                        Field = new FieldCondition
                        {
                            Key = "sourceDocument",
                            Match = new Match { Keyword = documentName }
                        }
                    }
                }
            },
            cancellationToken: cancellationToken);
    }

    public async Task ClearAsync(CancellationToken cancellationToken = default)
    {
        // Delete the collection and recreate it
        await _client.DeleteCollectionAsync(_collectionName, cancellationToken: cancellationToken);
        await InitializeCollectionAsync();
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

    /// <summary>
    /// Retrieves specific chunks by document name and chunk indices
    /// </summary>
    public async Task<List<DocumentChunk>> GetChunksByDocumentAndIndicesAsync(
        string documentName,
        IEnumerable<int> chunkIndices,
        string sectionPath = null,
        CancellationToken cancellationToken = default)
    {
        var indicesList = chunkIndices.ToList();
        if (indicesList.Count == 0)
        {
            return [];
        }

        var allChunks = new List<DocumentChunk>();

        // Query for each chunk index separately (simpler and more reliable)
        foreach (var chunkIndex in indicesList)
        {
            var filter = new Filter
            {
                Must =
                {
                    new Condition
                    {
                        Field = new FieldCondition
                        {
                            Key = "sourceDocument",
                            Match = new Match { Keyword = documentName }
                        }
                    },
                    new Condition
                    {
                        Field = new FieldCondition
                        {
                            Key = "chunkIndex",
                            Match = new Match { Integer = chunkIndex }
                        }
                    }
                }
            };
            if (!string.IsNullOrEmpty(sectionPath))
            {
                filter.Must.Add(new Condition
                {
                    Field = new FieldCondition
                    {
                        Key = "sectionPath",
                        Match = new Match { Keyword = sectionPath }
                    }
                });
            }

            var scrollResponse = await _client.ScrollAsync(
                collectionName: _collectionName,
                filter: filter,
                limit: 1,
                cancellationToken: cancellationToken);

            foreach (var result in scrollResponse.Result)
            {
                allChunks.Add(result.Payload.ToDocumentChunk());
            }
        }

        return allChunks;
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
}

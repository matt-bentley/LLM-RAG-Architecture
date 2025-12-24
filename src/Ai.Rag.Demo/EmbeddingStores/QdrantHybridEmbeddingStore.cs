using Ai.Rag.Demo.Models;
using Ai.Rag.Demo.Services;
using Ai.Rag.Demo.Settings;
using Microsoft.Extensions.Options;
using Qdrant.Client;
using Qdrant.Client.Grpc;

namespace Ai.Rag.Demo.EmbeddingStores;

/// <summary>
/// Qdrant-based implementation of embedding storage with hybrid search using dense vectors and BM25 sparse vectors with RRF fusion
/// </summary>
public class QdrantHybridEmbeddingStore : IEmbeddingStore
{
    private readonly QdrantClient _client;
    private readonly string _collectionName;
    private readonly int _denseVectorSize;
    private readonly Bm25SparseVectorizer _bm25Vectorizer;
    private readonly float _denseWeight;
    private readonly float _sparseWeight;
    private const double _k1 = 1.2;
    private const double _b = 0.75;

    private const string DenseVectorName = "dense";
    private const string SparseVectorName = "sparse";

    public QdrantHybridEmbeddingStore(IOptions<EmbeddingSettings> embeddingOptions, IOptions<QdrantSettings> qdrantOptions)
    {
        var qdrantSettings = qdrantOptions.Value;
        _client = new QdrantClient(
            host: qdrantSettings.Host,
            port: qdrantSettings.Port,
            https: qdrantSettings.UseTls,
            apiKey: qdrantSettings.ApiKey);
        _collectionName = qdrantSettings.CollectionName;
        _denseVectorSize = embeddingOptions.Value.VectorSize;
        _denseWeight = embeddingOptions.Value.DenseVectorWeight;
        _sparseWeight = 1 - embeddingOptions.Value.DenseVectorWeight;
        _bm25Vectorizer = new Bm25SparseVectorizer(
            storagePath: $"./{_collectionName}_df.json",
            k1: _k1,
            b: _b);

        InitializeCollectionAsync().GetAwaiter().GetResult();
    }

    public QdrantHybridEmbeddingStore(
        QdrantClient client,
        string collectionName = "documents_hybrid",
        int vectorSize = 1536,
        double k1 = 1.2,
        double b = 0.75,
        float denseWeight = 0.7f,
        float sparseWeight = 0.3f,
        string dfStoragePath = null)
    {
        _client = client;
        _collectionName = collectionName;
        _denseVectorSize = vectorSize;
        _denseWeight = denseWeight;
        _sparseWeight = sparseWeight;
        _bm25Vectorizer = new Bm25SparseVectorizer(
            storagePath: dfStoragePath ?? $"./{collectionName}_df.json",
            k1: k1,
            b: b);

        InitializeCollectionAsync().GetAwaiter().GetResult();
    }

    public QdrantHybridEmbeddingStore(
        QdrantClient client,
        Bm25SparseVectorizer bm25Vectorizer,
        string collectionName = "documents_hybrid",
        int vectorSize = 1536,
        float denseWeight = 0.7f,
        float sparseWeight = 0.3f)
    {
        _client = client;
        _collectionName = collectionName;
        _denseVectorSize = vectorSize;
        _denseWeight = denseWeight;
        _sparseWeight = sparseWeight;
        _bm25Vectorizer = bm25Vectorizer;

        InitializeCollectionAsync().GetAwaiter().GetResult();
    }

    private async Task InitializeCollectionAsync()
    {
        try
        {
            var collections = await _client.ListCollectionsAsync();
            var collectionExists = collections.Contains(_collectionName);

            if (!collectionExists)
            {
                // Create collection with both dense and sparse vectors
                await _client.CreateCollectionAsync(
                    collectionName: _collectionName,
                    vectorsConfig: new VectorParamsMap
                    {
                        Map =
                        {
                            [DenseVectorName] = new VectorParams
                            {
                                Size = (ulong)_denseVectorSize,
                                Distance = Distance.Cosine
                            }
                        }
                    },
                    sparseVectorsConfig: new SparseVectorConfig
                    {
                        Map =
                        {
                            [SparseVectorName] = new SparseVectorParams()
                        }
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
            throw new InvalidOperationException($"Failed to initialize Qdrant hybrid collection '{_collectionName}': {ex.Message}", ex);
        }
    }

    public async Task StoreChunksAsync(IEnumerable<DocumentChunk> chunks, CancellationToken cancellationToken = default)
    {
        var chunkList = chunks.ToList();
        if (chunkList.Count == 0)
        {
            return;
        }

        // Update document frequencies for all chunks
        _bm25Vectorizer.AddDocuments(chunkList.Select(c => c.Text));

        // Compute sparse vectors with updated IDF values
        var points = chunkList.Select(chunk =>
        {
            var sparseVector = _bm25Vectorizer.ComputeSparseVector(chunk.Text);

            var point = chunk.ToPointStruct();

            // Add dense and sparse vectors
            point.Vectors = new Vectors
            {
                Vectors_ = new NamedVectors
                {
                    Vectors =
                    {
                        [DenseVectorName] = new Vector { Data = { chunk.Embedding } },
                        [SparseVectorName] = new Vector
                        {
                            Indices = new SparseIndices { Data = { sparseVector.Indices } },
                            Data = { sparseVector.Values }
                        }
                    }
                }
            };

            return point;
        }).ToList();

        await _client.UpsertAsync(
            collectionName: _collectionName,
            points: points,
            cancellationToken: cancellationToken);
    }

    /// <summary>
    /// Search for similar chunks using hybrid search (dense + sparse with RRF fusion)
    /// </summary>
    public async Task<List<DocumentChunk>> SearchAsync(string query, float[] queryEmbedding, int topK = 5, CancellationToken cancellationToken = default)
    {
        return await SearchHybridAsync(queryEmbedding, query, topK, cancellationToken);
    }

    /// <summary>
    /// Performs hybrid search using dense vectors and BM25 sparse vectors with weighted RRF fusion.
    /// Weights are applied via prefetch limits - higher weight = more candidates considered.
    /// </summary>
    public async Task<List<DocumentChunk>> SearchHybridAsync(float[] queryEmbedding, string queryText, int topK = 5, CancellationToken cancellationToken = default)
    {
        // Compute sparse vector for the query
        var querySparseVector = _bm25Vectorizer.ComputeSparseVector(queryText);

        // Calculate prefetch limits based on weights
        // Higher weight = more candidates, giving that source more influence in RRF
        var totalWeight = _denseWeight + _sparseWeight;
        var densePrefetch = Math.Max(topK, (int)(topK * 4 * (_denseWeight / totalWeight)));
        var sparsePrefetch = Math.Max(topK, (int)(topK * 4 * (_sparseWeight / totalWeight)));

        // Use Query API with Prefetch for hybrid search with RRF fusion
        var searchResults = await _client.QueryAsync(
            collectionName: _collectionName,
            prefetch:
            [
                // Dense vector search (semantic)
                new PrefetchQuery
                {
                    Query = new Query { Nearest = new VectorInput { Dense = new DenseVector { Data = { queryEmbedding } } } },
                    Using = DenseVectorName,
                    Limit = (ulong)densePrefetch
                },
                // Sparse vector search (BM25 keyword)
                new PrefetchQuery
                {
                    Query = new Query
                    {
                        Nearest = new VectorInput
                        {
                            Sparse = new SparseVector
                            {
                                Indices = { querySparseVector.Indices },
                                Values = { querySparseVector.Values }
                            }
                        }
                    },
                    Using = SparseVectorName,
                    Limit = (ulong)sparsePrefetch
                }
            ],
            query: new Query { Fusion = Fusion.Rrf },
            limit: (ulong)topK,
            cancellationToken: cancellationToken);

        var chunks = searchResults.Select(result => result.Payload.ToDocumentChunk())
                                  .ToList();

        return chunks;
    }

    public async Task DeleteDocumentAsync(string documentName, CancellationToken cancellationToken = default)
    {
        // Note: This doesn't update document frequencies - would need to track which terms
        // were in deleted documents for perfect accuracy.
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
                            Match = new Qdrant.Client.Grpc.Match { Keyword = documentName }
                        }
                    }
                }
            },
            cancellationToken: cancellationToken);
    }

    public async Task ClearAsync(CancellationToken cancellationToken = default)
    {
        await _client.DeleteCollectionAsync(_collectionName, cancellationToken: cancellationToken);
        _bm25Vectorizer.Clear();
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
                            Match = new Qdrant.Client.Grpc.Match { Keyword = documentName }
                        }
                    },
                    new Condition
                    {
                        Field = new FieldCondition
                        {
                            Key = "chunkIndex",
                            Match = new Qdrant.Client.Grpc.Match { Integer = chunkIndex }
                        }
                    }
                }
            };

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

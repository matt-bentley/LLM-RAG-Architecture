using Ai.Rag.Demo.Models;
using Ai.Rag.Demo.Services;
using Microsoft.Extensions.Options;
using Qdrant.Client;
using Qdrant.Client.Grpc;

namespace Ai.Rag.Demo.EmbeddingStores;

/// <summary>
/// Qdrant-based implementation of embedding storage with hybrid search using dense vectors 
/// and BM25 sparse vectors with RRF fusion. Uses Qdrant's built-in IDF modifier instead of
/// tracking document frequencies locally.
/// </summary>
/// <remarks>
/// This is a simpler implementation that delegates IDF calculation to Qdrant.
/// Trade-offs vs QdrantHybridEmbeddingStore:
/// - Pros: No local state to maintain, no DF file to sync, always consistent
/// - Cons: Less accurate BM25 (Qdrant computes IDF from sparse index occupancy, not true document frequency)
/// </remarks>
public class QdrantHybridIdfEmbeddingStore : IEmbeddingStore
{
    private readonly QdrantClient _client;
    private readonly string _collectionName;
    private readonly int _denseVectorSize;
    private readonly float _denseWeight;
    private readonly float _sparseWeight;
    private readonly TextTokenizer _tokenizer;

    // BM25 TF parameters (IDF is handled by Qdrant)
    private const double _k1 = 1.2;
    private const double _b = 0.75;
    private readonly double _avgDocLength;

    private const string DenseVectorName = "dense";
    private const string SparseVectorName = "sparse";

    public QdrantHybridIdfEmbeddingStore(IOptions<EmbeddingSettings> options)
    {
        _client = new QdrantClient("localhost", 6334);
        _collectionName = "documents_hybrid_qdrant_idf";
        _avgDocLength = options.Value.MaxChunkTokens;
        _denseVectorSize = options.Value.VectorSize;
        _denseWeight = options.Value.DenseVectorWeight;
        _sparseWeight = 1 - options.Value.DenseVectorWeight;
        _tokenizer = new TextTokenizer();

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
                // Sparse vectors use Qdrant's IDF modifier for BM25-like scoring
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
                            [SparseVectorName] = new SparseVectorParams
                            {
                                // Qdrant applies IDF at query time based on index statistics
                                Modifier = Modifier.Idf
                            }
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

        var points = chunkList.Select(chunk =>
        {
            // Compute TF-only sparse vector (Qdrant will apply IDF)
            var sparseVector = ComputeTfSparseVector(chunk.Text);

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
        // Compute TF-only sparse vector for the query
        var querySparseVector = ComputeTfSparseVector(query);

        // Calculate prefetch limits based on weights
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
                // Sparse vector search (BM25 keyword) - Qdrant applies IDF
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

    public async Task DeleteDocumentAsync(string documentName, CancellationToken cancellationToken = default)
    {
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
    /// Computes a TF-only (term frequency) sparse vector for BM25.
    /// IDF is applied by Qdrant at query time using the Modifier.Idf setting.
    /// </summary>
    private (uint[] Indices, float[] Values) ComputeTfSparseVector(string text)
    {
        var (termFrequencies, tokenCount) = _tokenizer.GetTermFrequencies(text);
        if (tokenCount == 0)
        {
            return ([], []);
        }

        // Calculate BM25 TF weights (without IDF - Qdrant handles that)
        var sparseEntries = new Dictionary<uint, float>();

        foreach (var (index, tf) in termFrequencies)
        {
            // BM25 term frequency saturation formula
            // TF component: (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (docLength / avgDocLength)))
            var tfNormalized = (tf * (_k1 + 1)) / (tf + _k1 * (1 - _b + _b * (tokenCount / _avgDocLength)));

            sparseEntries[index] = (float)tfNormalized;
        }

        var sortedEntries = sparseEntries.OrderBy(e => e.Key).ToList();
        return (sortedEntries.Select(e => e.Key).ToArray(), sortedEntries.Select(e => e.Value).ToArray());
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

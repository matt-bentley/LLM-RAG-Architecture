using Ai.Rag.Demo.EmbeddingStores;
using Ai.Rag.Demo.ExtractionStrategies;
using Ai.Rag.Demo.Models;

namespace Ai.Rag.Demo.Services;

/// <summary>
/// Service for managing the RAG pipeline (indexing and retrieval)
/// </summary>
public class RagService
{
    private readonly IEmbeddingStore _embeddingStore;
    private readonly IEmbeddingGenerator _embeddingGenerator;
    private readonly IReranker _rerankerService;
    private readonly IDocumentExtractor _documentExtractor;
    private readonly FileSystemEmbeddingStore _fileSystemEmbeddingStore;
    private readonly QdrantEmbeddingStore _qdrantEmbeddingStore;
    private readonly QdrantHybridEmbeddingStore _hybridEmbeddingStore;
    private readonly QdrantHybridEmbeddingStoreWithQdrantIdf _hybridIdfEmbeddingStore;
    private readonly int _adjacentChunkCount;

    public RagService(
        IEmbeddingStore embeddingStore,
        IEmbeddingGenerator embeddingGenerator,
        IReranker rerankerService,
        FileSystemEmbeddingStore fileSystemEmbeddingStore,
        QdrantEmbeddingStore qdrantEmbeddingStore,
        QdrantHybridEmbeddingStore hybridEmbeddingStore,
        QdrantHybridEmbeddingStoreWithQdrantIdf hybridIdfEmbeddingStore,
        IDocumentExtractor documentExtractor,
        int adjacentChunkCount = 1)
    {
        _embeddingStore = embeddingStore;
        _embeddingGenerator = embeddingGenerator;
        _rerankerService = rerankerService;
        _fileSystemEmbeddingStore = fileSystemEmbeddingStore;
        _qdrantEmbeddingStore = qdrantEmbeddingStore;
        _hybridEmbeddingStore = hybridEmbeddingStore;
        _hybridIdfEmbeddingStore = hybridIdfEmbeddingStore;
        _documentExtractor = documentExtractor;
        _adjacentChunkCount = adjacentChunkCount;
    }

    /// <summary>
    /// Index a PDF document by extracting text, chunking, and generating embeddings
    /// </summary>
    public async Task<IndexResult> IndexPdfAsync(string pdfFileName, CancellationToken cancellationToken = default)
    {
        var pdfFilePath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "Data", pdfFileName));

        if (!File.Exists(pdfFilePath))
        {
            throw new FileNotFoundException($"PDF not found: {pdfFilePath}");
        }

        // Process document: extract text and create chunks
        //var extractionStrategy = new PdfBookmarkExtractor(1);
        //var extractionStrategy = new PdfBookmarkExtractor(2, 10);
        var extractionStrategy = new PdfFormatBasedExtractor(2);
        //var extractionStrategy = new PdfSimpleExtractor(10);
        var chunks = await _documentExtractor.ProcessDocumentAsync(pdfFilePath, extractionStrategy, cancellationToken);

        var embeddings = await _embeddingGenerator.GenerateEmbeddingsAsync(chunks, cancellationToken);

        for (int i = 0; i < chunks.Count; i++)
        {
            chunks[i].Embedding = embeddings[i];
        }

        await _embeddingStore.StoreChunksAsync(chunks, cancellationToken);

        return new IndexResult()
        {
            ChunkCount = chunks.Count,
        };
    }

    /// <summary>
    /// Index multiple PDF documents
    /// </summary>
    public async Task IndexPdfsAsync(IEnumerable<string> pdfFilePaths, CancellationToken cancellationToken = default)
    {
        foreach (var pdfPath in pdfFilePaths)
        {
            await IndexPdfAsync(pdfPath, cancellationToken);
        }
    }

    /// <summary>
    /// Search for relevant document chunks based on a query
    /// </summary>
    public async Task<List<DocumentChunk>> SearchAsync(string query, int topK = 5, CancellationToken cancellationToken = default)
    {
        // Generate embedding for the query
        var queryEmbedding = await _embeddingGenerator.GenerateEmbeddingAsync(query, cancellationToken);

        // Retrieve more results initially for reranking (2x the desired count)
        var retrievalCount = topK * 2;

        // Search for similar chunks with adjacent chunk expansion handled by the store
        var results = await _embeddingStore.SearchWithAdjacentChunksAsync(
            query, queryEmbedding, retrievalCount, _adjacentChunkCount, cancellationToken);

        //var fileResults = await _fileSystemEmbeddingStore.SearchWithAdjacentChunksAsync(
        //    query, queryEmbedding, retrievalCount, _adjacentChunkCount, cancellationToken);
        //var qResults = await _qdrantEmbeddingStore.SearchWithAdjacentChunksAsync(
        //    query, queryEmbedding, retrievalCount, _adjacentChunkCount, cancellationToken);
        //var hResults = await _hybridEmbeddingStore.SearchWithAdjacentChunksAsync(
        //    query, queryEmbedding, retrievalCount, _adjacentChunkCount, cancellationToken);
        //var hIdfResults = await _hybridIdfEmbeddingStore.SearchWithAdjacentChunksAsync(
        //    query, queryEmbedding, retrievalCount, _adjacentChunkCount, cancellationToken);

        // Rerank results and limit to topK
        var rerankedResults = await _rerankerService.RerankAsync(query, results, topK, cancellationToken);
        //var client = new HttpClient()
        //{
        //    BaseAddress = new Uri("http://localhost:8000/")
        //};
        //var crossEncoderReranker = new CrossEncoderReranker(client);
        //var rerankedResults2 = await crossEncoderReranker.RerankAsync(query, results, topK, cancellationToken);

        return rerankedResults.Select(r => r.Chunk).ToList();
    }

    /// <summary>
    /// Get context text from search results
    /// </summary>
    public string GetContextFromResults(List<DocumentChunk> searchResults)
    {
        if (searchResults.Count == 0)
        {
            return "No relevant information found.";
        }

        var context = new System.Text.StringBuilder();
        context.AppendLine("Context from documents:");
        context.AppendLine();

        for (int i = 0; i < searchResults.Count; i++)
        {
            var result = searchResults[i];
            context.Append($"[Source: {result.SourceDocument}");
            if(!string.IsNullOrEmpty(result.SectionPath) && result.SectionPath != result.SourceDocument)
            {
                context.Append($", Section: {result.SectionPath}");
            }
            else if(!string.IsNullOrEmpty(result.Section) && result.Section != result.SourceDocument)
            {
                context.Append($", Section: {result.Section}");
            }
            context.Append($", Pages: {result.StartPage}-{result.EndPage}");
            context.Append("]");
            context.AppendLine();
            context.AppendLine(result.Text);
            context.AppendLine();
        }

        return context.ToString();
    }

    /// <summary>
    /// Delete indexed document
    /// </summary>
    public async Task DeleteDocumentAsync(string documentName, CancellationToken cancellationToken = default)
    {
        await _embeddingStore.DeleteDocumentAsync(documentName, cancellationToken);
        Console.WriteLine($"Deleted document: {documentName}");
    }

    /// <summary>
    /// Clear all indexed documents
    /// </summary>
    public async Task ClearIndexAsync(CancellationToken cancellationToken = default)
    {
        await _embeddingStore.ClearAsync(cancellationToken);
        Console.WriteLine("Cleared all indexed documents");
    }
}

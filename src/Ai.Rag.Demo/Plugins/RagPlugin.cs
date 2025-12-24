using System.ComponentModel;
using Ai.Rag.Demo.Services;
using Microsoft.SemanticKernel;

namespace Ai.Rag.Demo.Plugins;

/// <summary>
/// Semantic Kernel plugin for RAG operations over PDF documents
/// </summary>
public class RagPlugin
{
    private readonly RagService _ragService;

    public RagPlugin(RagService ragService)
    {
        _ragService = ragService;
    }

    [KernelFunction("search_documents")]
    [Description("Search documents for information. Call this for any user question.")]
    public async Task<string> SearchDocumentsAsync(
        [Description("The question or search query")] string query,
        [Description("Max results to return")] int topK = 5,
        CancellationToken cancellationToken = default)
    {
        var results = await _ragService.SearchAsync(query, topK, cancellationToken);
        return _ragService.GetContextFromResults(results);
    }

    [KernelFunction("index_pdf")]
    [Description("Add a PDF file to the search index.")]
    public async Task<string> IndexPdfAsync(
        [Description("PDF filename to index")] string pdfFileName,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var result = await _ragService.IndexPdfAsync(pdfFileName, cancellationToken);
            return $"Indexed {pdfFileName}: {result.ChunkCount} chunks created";
        }
        catch (Exception ex)
        {
            return $"Failed to index PDF: {ex.Message}";
        }
    }

    [KernelFunction("delete_document")]
    [Description("Remove a document from the index.")]
    public async Task<string> DeleteDocumentAsync(
        [Description("Document name to delete")] string documentName,
        CancellationToken cancellationToken = default)
    {
        try
        {
            await _ragService.DeleteDocumentAsync(documentName, cancellationToken);
            return $"Deleted: {documentName}";
        }
        catch (Exception ex)
        {
            return $"Failed to delete: {ex.Message}";
        }
    }

    [KernelFunction("clear_index")]
    [Description("Delete all documents from the index.")]
    public async Task<string> ClearIndexAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            await _ragService.ClearIndexAsync(cancellationToken);
            return "All documents cleared";
        }
        catch (Exception ex)
        {
            return $"Failed to clear: {ex.Message}";
        }
    }
}

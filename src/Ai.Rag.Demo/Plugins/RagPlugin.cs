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
    [Description("Search through indexed PDF documents to find relevant information based on a query. Use this function by default if the user asks for any information.")]
    public async Task<string> SearchDocumentsAsync(
        [Description("The search query to find relevant information")] string query,
        [Description("Number of results to return (default: 5)")] int topK = 5,
        CancellationToken cancellationToken = default)
    {
        var results = await _ragService.SearchAsync(query, topK, cancellationToken);
        return _ragService.GetContextFromResults(results);
    }

    [KernelFunction("index_pdf")]
    [Description("Index a PDF document to make it searchable. After this function is called you should tell the user how many chunks were created from the file.")]
    public async Task<string> IndexPdfAsync(
        [Description("Name of the PDF file to index")] string pdfFileName,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var result = await _ragService.IndexPdfAsync(pdfFileName, cancellationToken);
            return $"Successfully indexed {pdfFileName} into {result.ChunkCount} chunks";
        }
        catch (Exception ex)
        {
            return $"Failed to index PDF: {ex.Message}";
        }
    }

    [KernelFunction("delete_document")]
    [Description("Delete an indexed document from the search index")]
    public async Task<string> DeleteDocumentAsync(
        [Description("Name of the document to delete")] string documentName,
        CancellationToken cancellationToken = default)
    {
        try
        {
            await _ragService.DeleteDocumentAsync(documentName, cancellationToken);
            return $"Successfully deleted: {documentName}";
        }
        catch (Exception ex)
        {
            return $"Failed to delete document: {ex.Message}";
        }
    }

    [KernelFunction("clear_index")]
    [Description("Clear all indexed documents from the search index")]
    public async Task<string> ClearIndexAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            await _ragService.ClearIndexAsync(cancellationToken);
            return "Successfully cleared all indexed documents";
        }
        catch (Exception ex)
        {
            return $"Failed to clear index: {ex.Message}";
        }
    }
}

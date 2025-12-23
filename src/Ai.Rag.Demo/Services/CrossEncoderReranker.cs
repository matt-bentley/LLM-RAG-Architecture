using System.Net.Http.Json;
using System.Text.Json;
using Ai.Rag.Demo.Models;

namespace Ai.Rag.Demo.Services;

/// <summary>
/// Re-ranks search results using an LLM to evaluate relevance to the query
/// </summary>
public class CrossEncoderReranker : IReranker
{
    private readonly HttpClient _client;

    public CrossEncoderReranker(HttpClient client)
    {
        _client = client;
    }

    /// <inheritdoc />
    public async Task<List<RerankedResult>> RerankAsync(
        string query,
        List<DocumentChunk> results,
        int topK = 5,
        CancellationToken cancellationToken = default)
    {
        if (results.Count == 0)
        {
            return [];
        }

        var serializerOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        };

        var response = await _client.PostAsync("/rerank",
            JsonContent.Create(new RerankRequest(
                query,
                results.Select(r => r.GenerateEmbeddingText())), options: serializerOptions),
            cancellationToken);
        response.EnsureSuccessStatusCode();
        var rerankResponse = await response.Content.ReadFromJsonAsync<RerankResponse>(options: serializerOptions, cancellationToken: cancellationToken);
        var rerankedResults = rerankResponse.Scores
            .Select((score, index) => new RerankedResult()
            {
                OriginalRank = index + 1,
                NewRank = 0, // to be set later
                Chunk = results[index],
                RelevanceScore = score * 10,
            })
            .OrderByDescending(r => r.RelevanceScore)
            .ToList();
        for (var i = 0; i < rerankedResults.Count; i++)
        {
            rerankedResults[i].NewRank = i + 1;
        }
        return rerankedResults.Take(topK)
                              .ToList();
    }

    private sealed record RerankRequest(string Query, IEnumerable<string> Documents);

    private sealed record RerankResponse(List<double> Scores);
}

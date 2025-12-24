using Ai.Rag.Demo.Models;
using Microsoft.Extensions.Options;
using System.Net.Http.Json;
using System.Text.Json;

namespace Ai.Rag.Demo.EmbeddingGenerators;

/// <summary>
/// Http implementation of embedding generation
/// </summary>
public class HttpEmbeddingGenerator : IEmbeddingGenerator
{
    private readonly HttpClient _client;
    private readonly JsonSerializerOptions _serializerOptions;

    public HttpEmbeddingGenerator(IOptions<EmbeddingSettings> options)
    {
        _client = new HttpClient
        {
            BaseAddress = new Uri(options.Value.Endpoint)
        };
        _serializerOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        };
    }

    public async Task<float[]> GenerateEmbeddingAsync(string text, CancellationToken cancellationToken = default)
    {
        var request = new EmbeddingRequest([text]);
        var response = await _client.PostAsync("embed",
            JsonContent.Create(request),
            cancellationToken
        );
        response.EnsureSuccessStatusCode();
        var embeddingResponse = await response.Content.ReadFromJsonAsync<EmbeddingResponse>(options: _serializerOptions, cancellationToken: cancellationToken);
        return embeddingResponse.Embeddings[0];
    }

    public async Task<IReadOnlyList<float[]>> GenerateEmbeddingsAsync(List<DocumentChunk> chunks, CancellationToken cancellationToken = default)
    {
        var allEmbeddings = new List<float[]>(chunks.Count);
        const int batchSize = 50;

        for (int i = 0; i < chunks.Count; i += batchSize)
        {
            var batch = chunks.Skip(i).Select(e => e.GenerateEmbeddingText()).Take(batchSize);
            var request = new EmbeddingRequest(batch);
            var response = await _client.PostAsync("embed",
                JsonContent.Create(request),
                cancellationToken
            );
            response.EnsureSuccessStatusCode();
            var embeddingResponse = await response.Content.ReadFromJsonAsync<EmbeddingResponse>(options: _serializerOptions, cancellationToken: cancellationToken);
            allEmbeddings.AddRange(embeddingResponse.Embeddings);

            Console.WriteLine($"Created embeddings for {allEmbeddings.Count}/{chunks.Count} chunks");
        }

        return allEmbeddings;
    }

    private sealed record EmbeddingRequest(IEnumerable<string> Texts);

    private sealed record EmbeddingResponse(IReadOnlyList<float[]> Embeddings, int Dimensions);
}

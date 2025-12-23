using Ai.Rag.Demo.Models;
using Azure;
using Azure.AI.OpenAI;
using Microsoft.Extensions.Options;

namespace Ai.Rag.Demo.EmbeddingGenerators;

/// <summary>
/// Azure OpenAI implementation of embedding generation
/// </summary>
public class AzureOpenAIEmbeddingGenerator : IEmbeddingGenerator
{
    private readonly AzureOpenAIClient _client;
    private readonly string _deploymentName;

    public AzureOpenAIEmbeddingGenerator(IOptions<EmbeddingSettings> options)
    {
        _client = new AzureOpenAIClient(new Uri(options.Value.Endpoint), new AzureKeyCredential(options.Value.ApiKey));
        _deploymentName = options.Value.DeploymentName;
    }

    public async Task<float[]> GenerateEmbeddingAsync(string text, CancellationToken cancellationToken = default)
    {
        var embeddingClient = _client.GetEmbeddingClient(_deploymentName);
        var response = await embeddingClient.GenerateEmbeddingAsync(text);
        
        return response.Value.ToFloats().ToArray();
    }

    public async Task<IReadOnlyList<float[]>> GenerateEmbeddingsAsync(List<DocumentChunk> chunks, CancellationToken cancellationToken = default)
    {
        var embeddingClient = _client.GetEmbeddingClient(_deploymentName);
        var allEmbeddings = new List<float[]>(chunks.Count);
        const int batchSize = 50;

        for (int i = 0; i < chunks.Count; i += batchSize)
        {
            var batch = chunks.Skip(i).Select(e => e.GenerateEmbeddingText()).Take(batchSize);
            var response = await embeddingClient.GenerateEmbeddingsAsync(batch, cancellationToken: cancellationToken);
            allEmbeddings.AddRange(response.Value.Select(e => e.ToFloats().ToArray()));
        }
        
        return allEmbeddings;
    }
}

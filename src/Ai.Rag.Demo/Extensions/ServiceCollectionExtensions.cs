using Ai.Rag.Demo.DocumentExtraction;
using Ai.Rag.Demo.EmbeddingGenerators;
using Ai.Rag.Demo.EmbeddingStores;
using Ai.Rag.Demo.Models;
using Ai.Rag.Demo.Plugins;
using Ai.Rag.Demo.Rerankers;
using Ai.Rag.Demo.Services;
using Ai.Rag.Demo.Settings;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using Microsoft.SemanticKernel;

namespace Ai.Rag.Demo.Extensions;

/// <summary>
/// Extension methods for registering application services
/// </summary>
public static class ServiceCollectionExtensions
{
    /// <summary>
    /// Register all RAG-related services
    /// </summary>
    public static IServiceCollection AddRagServices(this IServiceCollection services)
    {
        services.AddSingleton<IEmbeddingStore>(sp =>
        {
            var embeddingConfig = sp.GetRequiredService<IOptions<EmbeddingSettings>>();
            var qdrantConfig = sp.GetRequiredService<IOptions<QdrantSettings>>();
            return embeddingConfig.Value.StoreType switch
            {
                EmbeddingStoreType.QdrantHybrid => new QdrantHybridEmbeddingStore(embeddingConfig, qdrantConfig),
                EmbeddingStoreType.QdrantHybridIdf => new QdrantHybridIdfEmbeddingStore(embeddingConfig, qdrantConfig),
                EmbeddingStoreType.Qdrant => new QdrantEmbeddingStore(embeddingConfig, qdrantConfig),
                EmbeddingStoreType.FileSystem => new FileSystemEmbeddingStore(),
                _ => throw new NotSupportedException($"Embedding store type '{embeddingConfig.Value.StoreType}' is not supported.")
            };
        });

        // Register embedding generator
        services.AddSingleton<IEmbeddingGenerator>(sp =>
        {
            var config = sp.GetRequiredService<IOptions<EmbeddingSettings>>();
            return config.Value.Type switch
            {
                EmbeddingType.AzureOpenAi => new AzureOpenAIEmbeddingGenerator(config),
                EmbeddingType.Http => new HttpEmbeddingGenerator(config),
                _ => throw new NotSupportedException($"Embedding type '{config.Value.Type}' is not supported.")
            };
        });

        var httpHandler = new HttpClientHandler
        {
            ServerCertificateCustomValidationCallback = HttpClientHandler.DangerousAcceptAnyServerCertificateValidator
        };
        var httpClient = new HttpClient(httpHandler)
        {
            Timeout = TimeSpan.FromMinutes(5)
        };

        // Register re-ranker service
        services.AddSingleton<IReranker>(sp =>
            {
                var config = sp.GetRequiredService<IOptions<RerankerSettings>>();

                return config.Value.Type switch
                {
                    RerankerType.CrossEncoder => new CrossEncoderReranker(httpClient, config),
                    RerankerType.AzureOpenAiLlm => new LlmReranker(
                        Kernel.CreateBuilder()
                            .AddAzureOpenAIChatCompletion(
                            deploymentName: config.Value.DeploymentName,
                            endpoint: config.Value.Endpoint,
                            apiKey: config.Value.ApiKey,
                            httpClient: httpClient)
                        .Build()),
                    _ => throw new NotSupportedException($"Reranker type '{config.Value.Type}' is not supported.")
                };
            });

        services.AddSingleton<IDocumentExtractor, DocumentExtractor>();
        services.AddSingleton<RagService>();
        services.AddSingleton<RagPlugin>();

        return services;
    }

    /// <summary>
    /// Register Semantic Kernel and related services
    /// </summary>
    public static IServiceCollection AddSemanticKernel(this IServiceCollection services)
    {
        services.AddSingleton(sp =>
        {
            var builder = Kernel.CreateBuilder();
            var config = sp.GetRequiredService<IOptions<LlmSettings>>();

            var httpHandler = new HttpClientHandler
            {
                ServerCertificateCustomValidationCallback = HttpClientHandler.DangerousAcceptAnyServerCertificateValidator
            };
            var httpClient = new HttpClient(httpHandler)
            {
                Timeout = TimeSpan.FromMinutes(5)
            };

            switch (config.Value.Type)
            {
                case LlmType.OpenAi:
                    builder.AddOpenAIChatCompletion(
                        modelId: config.Value.ChatDeploymentName,
                        endpoint: new Uri(config.Value.Endpoint),
                        apiKey: config.Value.ApiKey);
                    break;
                case LlmType.AzureOpenAi:
                    builder.AddAzureOpenAIChatCompletion(
                        deploymentName: config.Value.ChatDeploymentName,
                        endpoint: config.Value.Endpoint,
                        apiKey: config.Value.ApiKey,
                        httpClient: httpClient);
                    break;
            }


            var kernel = builder.Build();

            // Add RAG plugin to kernel
            var ragPlugin = sp.GetRequiredService<RagPlugin>();
            kernel.Plugins.AddFromObject(ragPlugin, "RagPlugin");

            return kernel;
        });

        return services;
    }
}

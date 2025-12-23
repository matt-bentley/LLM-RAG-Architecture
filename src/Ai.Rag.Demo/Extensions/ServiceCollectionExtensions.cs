using Ai.Rag.Demo.EmbeddingStores;
using Ai.Rag.Demo.Models;
using Ai.Rag.Demo.Plugins;
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
        // Register embedding store

        services.AddSingleton<IEmbeddingStore>(sp =>
            new QdrantHybridEmbeddingStoreWithQdrantIdf(vectorSize: 3072));

        //services.AddSingleton<IEmbeddingStore>(sp =>
        //    new QdrantEmbeddingStore("localhost", 6334, "documents", 3072));

        services.AddSingleton(sp =>
            new QdrantEmbeddingStore("localhost", 6334, "documents", 3072));

        services.AddSingleton(sp =>
            new FileSystemEmbeddingStore("./embeddings"));

        services.AddSingleton(sp =>
            new QdrantHybridEmbeddingStore(vectorSize: 3072));

        services.AddSingleton(sp =>
            new QdrantHybridEmbeddingStoreWithQdrantIdf(vectorSize: 3072));

        //services.AddSingleton<IEmbeddingStore>(sp => 
        //    new FileSystemEmbeddingStore("./embeddings"));

        // Register embedding generator
        services.AddSingleton<IEmbeddingGenerator>(sp =>
        {
            var config = sp.GetRequiredService<IOptions<EmbeddingSettings>>();
            return new AzureOpenAIEmbeddingGenerator(
                config.Value.Endpoint,
                config.Value.ApiKey,
                config.Value.DeploymentName);
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
                    RerankerType.CrossEncoder => new CrossEncoderReranker(httpClient),
                    RerankerType.AzureOpenAiLlm => new LlmReranker(
                        Kernel.CreateBuilder()
                            .AddAzureOpenAIChatCompletion(
                            deploymentName: config.Value.DeploymentName,
                            endpoint: config.Value.Endpoint,
                            apiKey: config.Value.ApiKey,
                            httpClient: httpClient)
                        .Build())
                };
            });

        // Register document processor (combines PDF extraction and chunking)
        //services.AddSingleton<IDocumentExtractor>(sp =>
        //    new SimpleDocumentExtractor(chunkSize: 3000, chunkOverlap: 450));
        //services.AddSingleton<IDocumentExtractor>(sp =>
        //    new SectionBasedDocumentExtractor(maxChunkSize: 3000, chunkOverlap: 450, maxHeadingFontSize: 30));
        services.AddSingleton<IDocumentExtractor, DocumentExtractor>();

        // Register RAG service
        services.AddSingleton<RagService>();

        // Register RAG plugin
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

using Ai.Rag.Demo.Extensions;
using Ai.Rag.Demo.HostedServices;
using Ai.Rag.Demo.Models;
using Ai.Rag.Demo.Settings;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

var builder = Host.CreateApplicationBuilder(args);

builder.Services.Configure<LlmSettings>(builder.Configuration.GetSection("Llm"));
builder.Services.Configure<EmbeddingSettings>(builder.Configuration.GetSection("Embedding"));
builder.Services.Configure<RerankerSettings>(builder.Configuration.GetSection("Reranker"));
builder.Services.Configure<QdrantSettings>(builder.Configuration.GetSection("Qdrant"));

builder.Services.AddRagServices();
builder.Services.AddSemanticKernel();

// Register hosted service
builder.Services.AddHostedService<AiAssistantHostedService>();

// Build and run
var host = builder.Build();
await host.RunAsync();

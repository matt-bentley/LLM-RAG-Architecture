using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.AzureOpenAI;

namespace Ai.Rag.Demo.HostedServices;

/// <summary>
/// Hosted service that runs the interactive AI assistant
/// </summary>
public class AiAssistantHostedService : IHostedService
{
    private readonly ILogger<AiAssistantHostedService> _logger;
    private readonly IHostApplicationLifetime _lifetime;
    private readonly Kernel _kernel;

    public AiAssistantHostedService(
        ILogger<AiAssistantHostedService> logger,
        IHostApplicationLifetime lifetime,
        Kernel kernel)
    {
        _logger = logger;
        _lifetime = lifetime;
        _kernel = kernel;
    }

    public Task StartAsync(CancellationToken cancellationToken)
    {
        _lifetime.ApplicationStarted.Register(() =>
        {
            Task.Run(async () => await RunAssistantAsync(cancellationToken), cancellationToken);
        });

        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("Stopping AI Assistant...");
        return Task.CompletedTask;
    }

    private async Task RunAssistantAsync(CancellationToken cancellationToken)
    {
        try
        {
            Console.WriteLine("=== AI RAG Assistant with Semantic Kernel ===");
            Console.WriteLine();

            var chatHistory = new ChatHistory();

            chatHistory.AddSystemMessage("""
You are an AI assistant that answers questions using indexed documents.
    
IMPORTANT: You MUST use the available tools to answer questions. Do not make up information.
    
Available tools:
- search_documents: ALWAYS call this first when the user asks a question. Pass the user's question as the query.
- index_pdf: Call when user wants to index/add a PDF file.
- delete_document: Call when user wants to remove a document.
- clear_index: Call when user wants to clear all documents.
    
After receiving search results, summarize the relevant information and cite sources.
If no results are found, tell the user no relevant documents were found.
""");

            Console.WriteLine("Assistant initialized! You can:");
            Console.WriteLine("1. Index PDFs in the Data directory: Type 'index <pdf-name>' to add documents");
            Console.WriteLine("2. Ask questions: The assistant will search indexed documents");
            Console.WriteLine("3. Type 'exit' to quit");
            Console.WriteLine();

            // Interactive loop
            while (!cancellationToken.IsCancellationRequested)
            {
                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.Write("You: ");
                Console.ResetColor();
                var userInput = Console.ReadLine();

                if (string.IsNullOrWhiteSpace(userInput))
                {
                    continue;
                }

                if (userInput.Trim().Equals("exit", StringComparison.OrdinalIgnoreCase))
                {
                    Console.WriteLine("Goodbye!");
                    _lifetime.StopApplication();
                    break;
                }

                chatHistory.AddUserMessage(userInput);

                try
                {
                    // Get AI response with automatic function calling
                    var executionSettings = new AzureOpenAIPromptExecutionSettings
                    {
                        FunctionChoiceBehavior = FunctionChoiceBehavior.Auto()
                    };

                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.Write("Assistant: ");
                    Console.ResetColor();

                    var chatService = _kernel.GetRequiredService<IChatCompletionService>();

                    // Stream the response back to the console
                    var fullResponse = new System.Text.StringBuilder();
                    await foreach (var chunk in chatService.GetStreamingChatMessageContentsAsync(
                        chatHistory,
                        executionSettings,
                        _kernel,
                        cancellationToken))
                    {
                        if (!string.IsNullOrEmpty(chunk.Content))
                        {
                            Console.Write(chunk.Content);
                            fullResponse.Append(chunk.Content);
                        }
                    }

                    Console.WriteLine();
                    Console.WriteLine();

                    // Add assistant response to chat history
                    chatHistory.AddAssistantMessage(fullResponse.ToString());
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error processing chat message");
                    Console.WriteLine($"Error: {ex.Message}");
                    Console.WriteLine();
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Fatal error in AI Assistant");
            _lifetime.StopApplication();
        }
    }
}

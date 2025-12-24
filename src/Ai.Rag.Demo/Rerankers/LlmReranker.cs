using System.Text.Json;
using Ai.Rag.Demo.Models;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

namespace Ai.Rag.Demo.Rerankers;

/// <summary>
/// Re-ranks search results using an LLM to evaluate relevance to the query
/// </summary>
public class LlmReranker : IReranker
{
    private readonly Kernel _kernel;
    private readonly double _minimumRelevanceScore;

    /// <summary>
    /// Creates a new LLM reranker service
    /// </summary>
    /// <param name="kernel">The Semantic Kernel instance</param>
    /// <param name="minimumRelevanceScore">Minimum score (0-10) for a result to be included. Default is 5.0</param>
    public LlmReranker(Kernel kernel, double minimumRelevanceScore = 5.0)
    {
        _kernel = kernel;
        _minimumRelevanceScore = minimumRelevanceScore;
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

        var chatCompletionService = _kernel.GetRequiredService<IChatCompletionService>();

        var systemPrompt = """
            You are a relevance scoring assistant. Your task is to evaluate how relevant each document chunk is to a given query.
            
            You must respond with ONLY a JSON array in the following format:
            [
                { "id": 0, "score": <number between 0 and 10> },
                { "id": 1, "score": <number between 0 and 10> },
                ...
            ]
            
            Scoring guidelines:
            - 0-2: Not relevant at all, the content does not address the query
            - 3-4: Slightly relevant, tangentially related to the query
            - 5-6: Moderately relevant, partially addresses the query
            - 7-8: Highly relevant, directly addresses the query
            - 9-10: Perfectly relevant, comprehensive answer to the query
            
            Be strict and objective in your scoring. Focus on semantic relevance, not just keyword matches.
            Return scores for ALL chunks in the same order they were provided.
            """;

        var chunksText = string.Join("\n\n", results.Select((chunk, index) => 
            $"Chunk: {index}\n{chunk.GenerateEmbeddingText()}"));

        var userPrompt = $"""
            Query: {query}
            
            Document chunks to evaluate:
            
            {chunksText}
            
            Evaluate the relevance of each document chunk to the query. Return a JSON array with scores for all {results.Count} chunks.
            """;

        var chatHistory = new ChatHistory();
        chatHistory.AddSystemMessage(systemPrompt);
        chatHistory.AddUserMessage(userPrompt);

        var executionSettings = new PromptExecutionSettings
        {
            ExtensionData = new Dictionary<string, object>
            {
                ["temperature"] = 0.0,
                ["max_tokens"] = 20_000
            }
        };

        try
        {
            var response = await chatCompletionService.GetChatMessageContentAsync(
                chatHistory,
                executionSettings,
                _kernel,
                cancellationToken);

            var content = response.Content ?? string.Empty;
            var scores = ParseBatchScoreResponse(content, results.Count);

            var rerankedResults = results.Select((chunk, index) => new RerankedResult
            {
                Chunk = chunk,
                RelevanceScore = scores[index].Score,
                OriginalRank = index + 1
            }).ToList();

            // Filter out non-relevant results, sort by score, and limit to topK
            var sortedResults = rerankedResults
                .Where(r => r.RelevanceScore >= _minimumRelevanceScore)
                .OrderByDescending(r => r.RelevanceScore)
                .Take(topK)
                .ToList();

            for (int i = 0; i < sortedResults.Count; i++)
            {
                sortedResults[i].NewRank = i + 1;
            }

            return sortedResults;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: Failed to rerank chunks: {ex.Message}");
            // Return original order with default scores
            return results
                .Take(topK)
                .Select((chunk, index) => new RerankedResult
                {
                    Chunk = chunk,
                    RelevanceScore = 5.0 - (index * 0.1),
                    OriginalRank = index + 1,
                    NewRank = index + 1
                })
                .ToList();
        }
    }

    private static List<ScoreResponse> ParseBatchScoreResponse(string content, int expectedCount)
    {
        try
        {
            var cleanedContent = content.Trim();

            // Handle markdown code blocks if present
            if (cleanedContent.StartsWith("```"))
            {
                var startIndex = cleanedContent.IndexOf('[');
                var endIndex = cleanedContent.LastIndexOf(']');
                if (startIndex >= 0 && endIndex > startIndex)
                {
                    cleanedContent = cleanedContent[startIndex..(endIndex + 1)];
                }
            }

            var responses = JsonSerializer.Deserialize<List<ScoreResponse>>(cleanedContent, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });

            if (responses != null && responses.Count == expectedCount)
            {
                return responses;
            }

            // If count doesn't match, return defaults
            return CreateDefaultScores(expectedCount);
        }
        catch
        {
            return CreateDefaultScores(expectedCount);
        }
    }

    private static List<ScoreResponse> CreateDefaultScores(int count)
    {
        return Enumerable.Range(0, count)
            .Select(i => new ScoreResponse { Id = i, Score = 5.0, Explanation = "Could not parse response" })
            .ToList();
    }

    private class ScoreResponse
    {
        public int Id { get; set; }
        public double Score { get; set; }
        public string? Explanation { get; set; }
    }
}

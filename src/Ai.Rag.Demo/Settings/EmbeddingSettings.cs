namespace Ai.Rag.Demo.Models;

public class EmbeddingSettings
{
    public string Endpoint { get; set; }
    public string ApiKey { get; set; }
    public string DeploymentName { get; set; }
    public int MaxChunkTokens { get; set; }
    public int ChunkOverlapTokens { get; set; }
}
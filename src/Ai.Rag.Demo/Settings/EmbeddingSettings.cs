namespace Ai.Rag.Demo.Models;

public class EmbeddingSettings
{
    public string Endpoint { get; set; }
    public string ApiKey { get; set; }
    public string DeploymentName { get; set; }
    public EmbeddingType Type { get; set; }
    public int MaxChunkTokens { get; set; }
    public int ChunkOverlapTokens { get; set; }
    public int VectorSize { get; set; }
    public float DenseVectorWeight { get; set; } = 0.7f;
    public EmbeddingStoreType StoreType { get; set; }
}

public enum EmbeddingType
{
    Http,
    AzureOpenAi
}

public enum EmbeddingStoreType
{
    FileSystem,
    Qdrant,
    QdrantHybrid,
    QdrantHybridIdf
}
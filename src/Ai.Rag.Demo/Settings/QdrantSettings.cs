namespace Ai.Rag.Demo.Settings;

public class QdrantSettings
{
    public string Host { get; set; } = "localhost";
    public int Port { get; set; } = 6334;
    public string ApiKey { get; set; }
    public bool UseTls { get; set; }
    public string CollectionName { get; set; } = "documents";
}

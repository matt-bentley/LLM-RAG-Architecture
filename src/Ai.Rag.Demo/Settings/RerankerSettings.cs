namespace Ai.Rag.Demo.Settings;

public class RerankerSettings
{
    public string Endpoint { get; set; }
    public string ApiKey { get; set; }
    public string DeploymentName { get; set; }
    public RerankerType Type { get; set; }
}

public enum RerankerType
{
    AzureOpenAiLlm,
    CrossEncoder
}

namespace Ai.Rag.Demo.Settings;

public class LlmSettings
{
    public string Endpoint { get; set; }
    public string ApiKey { get; set; }
    public string ChatDeploymentName { get; set; }
    public LlmType Type { get; set; }
}

public enum LlmType
{
    OpenAi,
    AzureOpenAi
}

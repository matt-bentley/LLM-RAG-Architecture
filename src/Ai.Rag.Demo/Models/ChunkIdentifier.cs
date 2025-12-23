
namespace Ai.Rag.Demo.Models
{
    public sealed record ChunkIdentifier(string SourceDocument, string SectionPath, int ChunkIndex);
}

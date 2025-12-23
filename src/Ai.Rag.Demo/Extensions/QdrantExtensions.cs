using Ai.Rag.Demo.Models;
using Google.Protobuf.Collections;
using Qdrant.Client.Grpc;
using System.Text.Json;

namespace Qdrant.Client
{
    public static class QdrantExtensions
    {
        public static PointStruct ToPointStruct(this DocumentChunk chunk)
        {
            return new PointStruct
            {
                Id = new PointId { Uuid = Guid.NewGuid().ToString() },
                Vectors = chunk.Embedding,
                Payload =
                {
                    ["chunkId"] = chunk.Id,
                    ["text"] = chunk.Text,
                    ["sourceDocument"] = chunk.SourceDocument,
                    ["chunkIndex"] = chunk.ChunkIndex,
                    ["chunkTotal"] = chunk.ChunkTotal,
                    ["startPage"] = chunk.StartPage,
                    ["endPage"] = chunk.EndPage,
                    ["section"] = chunk.Section,
                    ["sectionPath"] = chunk.SectionPath,
                    ["metadata"] = JsonSerializer.Serialize(chunk.Metadata)
                }
            };
        }

        public static DocumentChunk ToDocumentChunk(this MapField<string, Value> payload)
        {
            return new DocumentChunk
            {
                Id = payload["chunkId"].StringValue,
                Text = payload["text"].StringValue,
                SourceDocument = payload["sourceDocument"].StringValue,
                ChunkIndex = (int)payload["chunkIndex"].IntegerValue,
                ChunkTotal = (int)payload["chunkTotal"].IntegerValue,
                StartPage = (int)payload["startPage"].IntegerValue,
                EndPage = (int)payload["endPage"].IntegerValue,
                Section = payload["section"].StringValue,
                SectionPath = payload["sectionPath"].StringValue,
                Metadata = JsonSerializer.Deserialize<Dictionary<string, string>>(payload["metadata"].StringValue) ?? new Dictionary<string, string>()
            };
        }
    }
}

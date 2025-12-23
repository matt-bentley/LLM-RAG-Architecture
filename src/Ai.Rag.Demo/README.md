# AI RAG Assistant with Semantic Kernel

A production-ready AI assistant using Semantic Kernel with RAG (Retrieval Augmented Generation) capabilities for PDF documents, built with Azure OpenAI and proper dependency injection.

## Architecture

### Dependency Injection & Hosting
The application uses the **Generic Host** pattern with proper dependency injection:

- **Program.cs**: Host builder configuration and startup
- **AiAssistantHostedService**: IHostedService implementation for the interactive chat
- **ServiceCollectionExtensions**: Clean service registration
- **appsettings.json**: Configuration file for Azure OpenAI settings

### Core Services

#### Embedding Storage
- **IEmbeddingStore**: Interface for vector storage
- **FileSystemEmbeddingStore**: JSON-based file system implementation with cosine similarity

#### Document Processing
- **PdfDocumentProcessor**: Extract text from PDFs using PdfPig
- **ChunkingService**: Split text into overlapping chunks with sentence boundary detection

#### Embedding Generation
- **IEmbeddingGenerator**: Interface for embedding generation
- **AzureOpenAIEmbeddingGenerator**: Direct Azure OpenAI client implementation

#### RAG Pipeline
- **RagService**: Orchestrates indexing, embedding, and retrieval
- **RagPlugin**: Semantic Kernel plugin with searchable functions

### Models
- **ChunkMetadata**: Represents text chunks with embeddings and metadata
- **AzureOpenAIConfig**: Configuration for Azure OpenAI services

## Configuration

### Option 1: appsettings.json
```json
{
  "AzureOpenAI": {
    "Endpoint": "https://your-resource.openai.azure.com",
    "ApiKey": "your-api-key",
    "ChatDeploymentName": "gpt-4",
    "EmbeddingDeploymentName": "text-embedding-ada-002"
  }
}
```

### Option 2: Environment Variables
```bash
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
```

Environment variables override appsettings.json values.

## Usage

### Run the Application
```bash
dotnet run
```

### Index PDFs
```
You: index path/to/document.pdf
```

### Ask Questions
```
You: What is the main topic of the document?
Assistant: [AI will search indexed documents and provide answer with citations]
```

### Exit
```
You: exit
```

## Dependency Injection Benefits

? **Testability**: All services use interfaces, making unit testing easy  
? **Lifetime Management**: Proper service lifetimes (Singleton, Scoped, Transient)  
? **Configuration**: Centralized configuration management  
? **Logging**: Built-in ILogger support throughout  
? **Extensibility**: Easy to swap implementations (e.g., use a vector database instead of file system)  
? **Separation of Concerns**: Clean architecture with single responsibility

## Service Lifetimes

| Service | Lifetime | Reason |
|---------|----------|--------|
| IEmbeddingStore | Singleton | Shared state across application |
| IEmbeddingGenerator | Singleton | Stateless, can be reused |
| PdfDocumentProcessor | Singleton | Stateless utility |
| ChunkingService | Singleton | Configured once, immutable |
| RagService | Singleton | Orchestrates other singletons |
| Kernel | Singleton | Configured once with plugins |
| IChatCompletionService | Singleton | Resolved from Kernel |
| AiAssistantHostedService | Singleton | Host lifetime service |

## Extending the Application

### Add a Vector Database
Implement `IEmbeddingStore` for your vector DB (e.g., Azure AI Search, Pinecone, Qdrant):

```csharp
public class VectorDbEmbeddingStore : IEmbeddingStore
{
    // Your implementation
}

// Register in ServiceCollectionExtensions
services.AddSingleton<IEmbeddingStore, VectorDbEmbeddingStore>();
```

### Add Custom Plugins
Register additional Semantic Kernel plugins:

```csharp
kernel.Plugins.AddFromObject(new MyCustomPlugin(), "MyPlugin");
```

### Add More Document Types
Extend the processor pattern:

```csharp
services.AddSingleton<IDocumentProcessor, WordDocumentProcessor>();
services.AddSingleton<IDocumentProcessor, MarkdownProcessor>();
```

## Project Structure

```
Ai.Rag.Demo/
??? Extensions/
?   ??? ServiceCollectionExtensions.cs    # DI registration
??? HostedServices/
?   ??? AiAssistantHostedService.cs       # Interactive chat host
??? Models/
?   ??? AzureOpenAIConfig.cs              # Configuration model
?   ??? ChunkMetadata.cs                  # Data model
??? Plugins/
?   ??? RagPlugin.cs                      # SK plugin
??? Services/
?   ??? IEmbeddingStore.cs                # Storage interface
?   ??? FileSystemEmbeddingStore.cs       # File implementation
?   ??? IEmbeddingGenerator.cs            # Embedding interface
?   ??? AzureOpenAIEmbeddingGenerator.cs  # Azure OpenAI impl
?   ??? PdfDocumentProcessor.cs           # PDF extraction
?   ??? ChunkingService.cs                # Text chunking
?   ??? RagService.cs                     # RAG orchestration
??? appsettings.json                      # Configuration
??? Program.cs                            # Host builder
```

## Technologies

- **.NET 10**: Latest .NET framework
- **Microsoft.Extensions.Hosting**: Generic host for DI and lifecycle
- **Azure.AI.OpenAI**: Official Azure OpenAI SDK
- **Microsoft.SemanticKernel**: AI orchestration framework
- **PdfPig**: PDF text extraction
- **System.Numerics.Tensors**: Vector operations (cosine similarity)

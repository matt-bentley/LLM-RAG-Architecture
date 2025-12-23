from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from contextlib import asynccontextmanager

# Global model and tokenizer
model = None
tokenizer = None
model_id = "BAAI/bge-small-en-v1.5"

def mean_pooling(model_output, attention_mask):
    """Mean pooling - take attention mask into account for correct averaging."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model, tokenizer
    
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    model.eval()
    print("Model loaded successfully")
    
    yield
    
    # Cleanup (if needed)
    print("Shutting down...")

app = FastAPI(
    title="Embedding API",
    description="Embedding service for RAG applications",
    version="1.0.0",
    lifespan=lifespan
)

class EmbedRequest(BaseModel):
    texts: list[str]

class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dimensions: int

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": model_id}

@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """
    Generate embeddings for the provided texts.
    """
    # Tokenize
    encoded_input = tokenizer(
        request.texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )
    
    # Generate embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
    
    embeddings_list = embeddings.tolist()
    dimensions = len(embeddings_list[0]) if embeddings_list else 0
    
    return EmbedResponse(embeddings=embeddings_list, dimensions=dimensions)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

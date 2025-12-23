from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from contextlib import asynccontextmanager

# Global model and tokenizer
model = None
tokenizer = None
model_id = "cross-encoder/ms-marco-MiniLM-L6-v2"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model, tokenizer
    
    print(f"Loading model: {model_id}")
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Model loaded successfully")
    
    yield
    
    # Cleanup (if needed)
    print("Shutting down...")


app = FastAPI(
    title="Reranker API",
    description="Cross-encoder reranking service for RAG applications",
    version="1.0.0",
    lifespan=lifespan
)


class RerankRequest(BaseModel):
    query: str
    documents: list[str]


class RerankResponse(BaseModel):
    scores: list[float]


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": model_id}


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """
    Rerank documents based on relevance to the query.
    
    Returns scores for each document in the same order as provided.
    """
    # Create query-document pairs (same query paired with each document)
    queries = [request.query] * len(request.documents)
    
    # Tokenize
    features = tokenizer(
        queries,
        request.documents,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )
    
    # Get scores
    with torch.no_grad():
        logits = model(**features).logits
        scores = torch.sigmoid(logits).squeeze(-1).tolist()
    
    # Handle single document case
    if isinstance(scores, float):
        scores = [scores]
    
    return RerankResponse(scores=scores)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

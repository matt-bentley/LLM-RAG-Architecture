# Use an official Python base image  
FROM python:3.10-slim  
  
# Avoid interactive prompts & enable unbuffered logs  
ENV DEBIAN_FRONTEND=noninteractive \  
    PYTHONUNBUFFERED=1  
  
# Install system dependencies (curl for debugging, git if needed)  
RUN apt-get update && apt-get install -y --no-install-recommends \  
    build-essential \  
    git \  
    && rm -rf /var/lib/apt/lists/*  
  
# Create app directory  
WORKDIR /app  
  
RUN pip install --upgrade pip

# Install Python dependencies  
# torch CPU build + transformers + FastAPI  
RUN pip install --no-cache-dir \  
    "torch" \  
    "transformers" \
    "fastapi" \
    "uvicorn[standard]"

# Download the model once at build time (optional but speeds up first run)  
# Comment this out if you prefer download at container runtime  
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \  
    model_id = 'BAAI/bge-reranker-base'; \
    AutoModelForSequenceClassification.from_pretrained(model_id); \  
    AutoTokenizer.from_pretrained(model_id)"  

# Copy your Python script into the container  
COPY main.py /app/main.py  

# Expose the API port
EXPOSE 8000
  
# Run the FastAPI app with uvicorn  
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]  
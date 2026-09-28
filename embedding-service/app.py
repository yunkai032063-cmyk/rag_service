import os
from typing import Literal
import torch
from fastapi import FastAPI, HTTPException, Header, Depends, status
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(
    MODEL_NAME,
    device=DEVICE,
)

EMBEDDING_DIM = model.get_embedding_dimension()
API_KEY = os.environ.get("EMBEDDING_API_KEY")

def verify_api_key(x_api_key: str = Header(default=None)):
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server API key not configured",
        )
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return True


app = FastAPI(
    title="Local Embedding Service",
    version="1.0.0",
)

class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1)
    input_type: Literal["query", "document"] = "document"

class EmbedResponse(BaseModel):
    model: str
    device: str
    dimension: int
    embeddings: list[list[float]]

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
        "dimension": EMBEDDING_DIM
    }

@app.post("/embed", response_model=EmbedResponse, dependencies=[Depends(verify_api_key)])
def embed(request: EmbedRequest):
    if not request.texts:
        raise HTTPException(
            status_code=400,
            detail="texts cannot be empty",
        )
    try:
        
        embeddings = model.encode(
            request.texts,
            prompt_name="query" if request.input_type == "query" else None,
            batch_size=8,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return {
            "model": MODEL_NAME,
            "device": DEVICE,
            "dimension": EMBEDDING_DIM,
            "embeddings": embeddings.tolist(),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )
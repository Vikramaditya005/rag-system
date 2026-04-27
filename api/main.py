"""
api/main.py
-----------
FastAPI backend exposing RAG pipeline endpoints.

Endpoints:
  POST /query         - RAG or baseline query
  POST /compare       - Side-by-side RAG vs baseline
  GET  /health        - Health check
  GET  /index-stats   - FAISS index info
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger
from dotenv import load_dotenv

from generation.pipeline import RAGPipeline

load_dotenv()

# ─────────────────────────────────────────
# Startup / Shutdown
# ─────────────────────────────────────────

pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Loading RAG pipeline...")
    pipeline = RAGPipeline(
        use_reranker=True,
        top_k_retrieval=10,
        top_n_rerank=5,
        rewrite_query=True,
    )
    pipeline.load_index()
    logger.success("Pipeline ready.")
    yield
    logger.info("Shutting down...")


# ─────────────────────────────────────────
# App
# ─────────────────────────────────────────

app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation over arXiv papers",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000)
    mode: Literal["rag", "baseline"] = "rag"
    top_k: Optional[int] = Field(5, ge=1, le=20)


class SourceChunk(BaseModel):
    paper_title: str
    paper_id: str
    score: float
    rank: int
    snippet: str  # First 200 chars


class QueryResponse(BaseModel):
    question: str
    answer: str
    mode: str
    sources: list[SourceChunk]
    latency: dict
    token_usage: dict


class CompareResponse(BaseModel):
    question: str
    rag_answer: str
    baseline_answer: str
    sources: list[SourceChunk]
    rag_latency: dict
    baseline_latency: dict


# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "index_loaded": pipeline is not None}


@app.get("/index-stats")
def index_stats():
    if pipeline is None:
        raise HTTPException(503, "Pipeline not loaded")
    vs = pipeline.vector_store
    return {
        "total_vectors": vs.index.ntotal if vs.index else 0,
        "total_chunks": len(vs.chunks),
        "embedding_model": vs.model.get_sentence_embedding_dimension(),
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if pipeline is None:
        raise HTTPException(503, "Pipeline not loaded")

    try:
        response = pipeline.query(req.question, mode=req.mode)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(500, str(e))

    sources = [
        SourceChunk(
            paper_title=c.paper_title,
            paper_id=c.paper_id,
            score=round(c.score, 4),
            rank=c.rank,
            snippet=c.text[:200] + "...",
        )
        for c in response.retrieved_chunks
    ]

    return QueryResponse(
        question=response.query,
        answer=response.answer,
        mode=response.mode,
        sources=sources,
        latency=response.latency,
        token_usage=response.token_usage,
    )


@app.post("/compare", response_model=CompareResponse)
def compare(req: QueryRequest):
    if pipeline is None:
        raise HTTPException(503, "Pipeline not loaded")

    try:
        result = pipeline.compare(req.question)
    except Exception as e:
        logger.error(f"Compare failed: {e}")
        raise HTTPException(500, str(e))

    sources = [
        SourceChunk(
            paper_title=s["title"],
            paper_id="",
            score=round(s["score"], 4),
            rank=s["rank"],
            snippet="",
        )
        for s in result["rag"]["sources"]
    ]

    return CompareResponse(
        question=result["question"],
        rag_answer=result["rag"]["answer"],
        baseline_answer=result["baseline"]["answer"],
        sources=sources,
        rag_latency=result["rag"]["latency"],
        baseline_latency=result["baseline"]["latency"],
    )
"""
generation/pipeline.py
-----------------------
Main RAG pipeline. Orchestrates:
  retrieval → (optional rerank) → generation → latency tracking

This is the main entry point for running queries.
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

from retrieval.vector_store import VectorStore, RetrievedChunk
from retrieval.reranker import Reranker
from generation.llm import LLMGenerator


@dataclass
class RAGResponse:
    query: str
    answer: str
    mode: str  # "rag" | "baseline"
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    latency: dict = field(default_factory=dict)  # detailed timing
    token_usage: dict = field(default_factory=dict)


class RAGPipeline:
    def __init__(
        self,
        use_reranker: bool = True,
        top_k_retrieval: int = 10,
        top_n_rerank: int = 5,
        rewrite_query: bool = True,
    ):
        self.use_reranker = use_reranker
        self.top_k_retrieval = top_k_retrieval
        self.top_n_rerank = top_n_rerank
        self.rewrite_query = rewrite_query

        logger.info("Initializing RAG Pipeline...")
        self.vector_store = VectorStore()
        self.llm = LLMGenerator()
        self.reranker = Reranker() if use_reranker else None
        logger.success("RAG Pipeline ready.")

    def load_index(self, index_path=None):
        """Load pre-built FAISS index."""
        if index_path:
            self.vector_store.load(index_path)
        else:
            self.vector_store.load()

    # ─────────────────────────────────────────
    # RAG Query
    # ─────────────────────────────────────────

    def query(self, question: str, mode: str = "rag") -> RAGResponse:
        """
        Run a full RAG query.

        Args:
            question: User's question
            mode: "rag" (with retrieval) or "baseline" (LLM only)

        Returns:
            RAGResponse with answer + full latency breakdown
        """
        total_start = time.perf_counter()
        latency = {}

        if mode == "rag":
            # Step 1: Retrieve
            retrieved, retrieval_latency = self.vector_store.retrieve(
                query=question,
                top_k=self.top_k_retrieval,
                rewrite_query=self.rewrite_query,
            )
            latency.update(retrieval_latency)

            # Step 2: Re-rank (optional)
            if self.use_reranker and self.reranker and retrieved:
                retrieved, rerank_ms = self.reranker.rerank(
                    query=question,
                    chunks=retrieved,
                    top_n=self.top_n_rerank,
                )
                latency["rerank_ms"] = round(rerank_ms, 2)

            context_chunks = retrieved
        else:
            context_chunks = None
            retrieved = []

        # Step 3: Generate
        gen_result = self.llm.generate(
            query=question,
            context_chunks=context_chunks,
        )
        latency["generation_ms"] = gen_result["generation_ms"]
        latency["total_ms"] = round((time.perf_counter() - total_start) * 1000, 2)

        return RAGResponse(
            query=question,
            answer=gen_result["answer"],
            mode=mode,
            retrieved_chunks=retrieved,
            latency=latency,
            token_usage={
                "prompt_tokens": gen_result["prompt_tokens"],
                "completion_tokens": gen_result["completion_tokens"],
            },
        )

    # ─────────────────────────────────────────
    # Both modes for comparison
    # ─────────────────────────────────────────

    def compare(self, question: str) -> dict:
        """
        Run the same question in both RAG and baseline mode.
        Returns dict with both responses for side-by-side comparison.
        """
        logger.info(f"Running comparison for: '{question}'")
        rag_response = self.query(question, mode="rag")
        baseline_response = self.query(question, mode="baseline")

        return {
            "question": question,
            "rag": {
                "answer": rag_response.answer,
                "latency": rag_response.latency,
                "sources": [
                    {"title": c.paper_title, "score": c.score, "rank": c.rank}
                    for c in rag_response.retrieved_chunks
                ],
            },
            "baseline": {
                "answer": baseline_response.answer,
                "latency": baseline_response.latency,
            },
        }
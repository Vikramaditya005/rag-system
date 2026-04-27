import os
import time
import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from loguru import logger
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from retrieval.chunker import Chunk

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", "embeddings/faiss_index"))
TOP_K = int(os.getenv("TOP_K", 5))


@dataclass
class RetrievedChunk:
    text: str
    paper_id: str
    paper_title: str
    chunk_index: int
    score: float  # L2 distance (lower = more similar)
    rank: int


class VectorStore:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index: Optional[faiss.Index] = None
        self.chunks: list[Chunk] = []

    # ─────────────────────────────────────────
    # Build index
    # ─────────────────────────────────────────

    def build(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        """Embed all chunks and build FAISS index."""
        logger.info(f"Embedding {len(chunks)} chunks (batch_size={batch_size})...")
        texts = [c.text for c in chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # For cosine similarity via inner product
        )
        embeddings = embeddings.astype(np.float32)

        # Use IndexFlatIP (inner product = cosine similarity when normalized)
        self.index = faiss.IndexFlatIP(self.dimension)
        # Wrap with IDMap so we can track chunk ids
        self.index = faiss.IndexIDMap(self.index)
        ids = np.arange(len(chunks)).astype(np.int64)
        self.index.add_with_ids(embeddings, ids)

        self.chunks = chunks
        logger.success(f"FAISS index built with {self.index.ntotal} vectors.")

    # ─────────────────────────────────────────
    # Save / Load
    # ─────────────────────────────────────────

    def save(self, path: Path = FAISS_INDEX_PATH) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        logger.success(f"Index saved to {path}")

    def load(self, path: Path = FAISS_INDEX_PATH) -> None:
        path = Path(path)
        if not (path / "index.faiss").exists():
            raise FileNotFoundError(f"No FAISS index at {path}. Run build() first.")
        self.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        logger.success(f"Index loaded: {self.index.ntotal} vectors, {len(self.chunks)} chunks.")

    # ─────────────────────────────────────────
    # Retrieval
    # ─────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        rewrite_query: bool = False,
    ) -> tuple[list[RetrievedChunk], dict]:
        """
        Retrieve top-K chunks for a query.
        Returns (chunks, latency_info)
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() or load() first.")

        if rewrite_query:
            query = self._rewrite_query(query)

        t0 = time.perf_counter()
        query_emb = self.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)
        embed_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        scores, ids = self.index.search(query_emb, top_k)
        search_time = time.perf_counter() - t1

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], ids[0])):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append(RetrievedChunk(
                text=chunk.text,
                paper_id=chunk.paper_id,
                paper_title=chunk.paper_title,
                chunk_index=chunk.chunk_index,
                score=float(score),
                rank=rank + 1,
            ))

        latency = {
            "embed_ms": round(embed_time * 1000, 2),
            "search_ms": round(search_time * 1000, 2),
            "total_retrieval_ms": round((embed_time + search_time) * 1000, 2),
        }

        return results, latency

    # ─────────────────────────────────────────
    # Query Rewriting (Upgrade Feature)
    # ─────────────────────────────────────────

    def _rewrite_query(self, query: str) -> str:
        """
        Expand query with synonyms/keywords for better recall.
        Simple rule-based expansion (no LLM needed here).
        Can be upgraded to use LLM for HyDE (Hypothetical Document Embeddings).
        """
        expansions = {
            "rag": "retrieval augmented generation",
            "llm": "large language model",
            "nlp": "natural language processing",
            "qa": "question answering",
            "bert": "bidirectional encoder representations transformers",
        }
        query_lower = query.lower()
        for abbr, full in expansions.items():
            if abbr in query_lower and full not in query_lower:
                query = query + f" {full}"

        logger.debug(f"Rewritten query: {query}")
        return query
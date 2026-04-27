import time
from loguru import logger
from sentence_transformers import CrossEncoder
from retrieval.vector_store import RetrievedChunk


RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        logger.info(f"Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int = 3,
    ) -> tuple[list[RetrievedChunk], float]:
        """
        Re-rank retrieved chunks using cross-encoder.
        Returns (reranked_chunks, latency_ms)
        """
        if not chunks:
            return [], 0.0

        pairs = [(query, chunk.text) for chunk in chunks]

        t0 = time.perf_counter()
        scores = self.model.predict(pairs)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Attach cross-encoder scores and sort
        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)

        reranked = sorted(chunks, key=lambda x: x.score, reverse=True)

        # Update ranks
        for i, chunk in enumerate(reranked):
            chunk.rank = i + 1

        logger.debug(f"Reranked {len(chunks)} → top {top_n} | latency={latency_ms:.1f}ms")
        return reranked[:top_n], latency_ms
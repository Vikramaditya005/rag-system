"""
retrieval/chunker.py
--------------------
Implements and compares two chunking strategies:
  1. Fixed-size chunking (by token count)
  2. Semantic chunking (by sentence similarity boundaries)

This comparison is a key evaluation metric in the project.
"""

import re
from dataclasses import dataclass
from typing import Literal
from loguru import logger


@dataclass
class Chunk:
    text: str
    paper_id: str
    paper_title: str
    chunk_index: int
    strategy: str
    start_char: int = 0


ChunkStrategy = Literal["fixed", "semantic"]


# ─────────────────────────────────────────────
# Fixed-size chunking
# ─────────────────────────────────────────────

def fixed_chunk(
    text: str,
    paper_id: str,
    paper_title: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    """
    Split text into fixed-size word chunks with overlap.
    Simple, fast, but can break mid-sentence.
    """
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk_words = words[i: i + chunk_size]
        if len(chunk_words) < 20:  # Skip tiny trailing chunks
            continue
        chunks.append(Chunk(
            text=" ".join(chunk_words),
            paper_id=paper_id,
            paper_title=paper_title,
            chunk_index=len(chunks),
            strategy="fixed",
            start_char=len(" ".join(words[:i])),
        ))

    return chunks


# ─────────────────────────────────────────────
# Semantic chunking
# ─────────────────────────────────────────────

def semantic_chunk(
    text: str,
    paper_id: str,
    paper_title: str,
    max_chunk_size: int = 512,
    similarity_threshold: float = 0.75,
) -> list[Chunk]:
    """
    Split text at semantic boundaries using sentence embeddings.
    Groups sentences until similarity drops below threshold.
    More coherent chunks, but slower.
    """
    # Lazy import to avoid loading model when not needed
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    embeddings = model.encode(sentences, show_progress_bar=False)

    chunks = []
    current_sentences = [sentences[0]]
    current_words = len(sentences[0].split())

    for i in range(1, len(sentences)):
        # Cosine similarity between current group centroid and next sentence
        current_emb = np.mean(
            embeddings[i - len(current_sentences): i], axis=0
        )
        next_emb = embeddings[i]
        sim = _cosine_sim(current_emb, next_emb)

        next_words = len(sentences[i].split())

        # Start new chunk if similarity drops OR chunk too big
        if sim < similarity_threshold or (current_words + next_words) > max_chunk_size:
            chunk_text = " ".join(current_sentences)
            if len(chunk_text.split()) >= 20:
                chunks.append(Chunk(
                    text=chunk_text,
                    paper_id=paper_id,
                    paper_title=paper_title,
                    chunk_index=len(chunks),
                    strategy="semantic",
                ))
            current_sentences = [sentences[i]]
            current_words = next_words
        else:
            current_sentences.append(sentences[i])
            current_words += next_words

    # Add last chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        if len(chunk_text.split()) >= 20:
            chunks.append(Chunk(
                text=chunk_text,
                paper_id=paper_id,
                paper_title=paper_title,
                chunk_index=len(chunks),
                strategy="semantic",
            ))

    return chunks


def _split_into_sentences(text: str) -> list[str]:
    """Naive sentence splitter using regex."""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _cosine_sim(a, b) -> float:
    import numpy as np
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ─────────────────────────────────────────────
# Unified interface
# ─────────────────────────────────────────────

def chunk_papers(
    papers: list[dict],
    strategy: ChunkStrategy = "fixed",
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    """
    Chunk a list of paper dicts using the specified strategy.
    Each paper dict must have: id, title, text
    """
    all_chunks = []
    for paper in papers:
        if strategy == "fixed":
            chunks = fixed_chunk(
                text=paper["text"],
                paper_id=paper["id"],
                paper_title=paper["title"],
                chunk_size=chunk_size,
                overlap=overlap,
            )
        elif strategy == "semantic":
            chunks = semantic_chunk(
                text=paper["text"],
                paper_id=paper["id"],
                paper_title=paper["title"],
                max_chunk_size=chunk_size,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        all_chunks.extend(chunks)

    logger.info(f"[{strategy}] Created {len(all_chunks)} chunks from {len(papers)} papers")
    return all_chunks
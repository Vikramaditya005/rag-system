"""
scripts/build_index.py
-----------------------
Standalone script to ingest papers and build FAISS index.
Run this ONCE before starting the API.

Usage:
  python scripts/build_index.py
  python scripts/build_index.py --query "attention mechanism transformers" --max 30
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from arXiv papers")
    parser.add_argument("--query", type=str, default=None, help="arXiv search query")
    parser.add_argument("--max", type=int, default=None, help="Max papers to download")
    parser.add_argument("--strategy", choices=["fixed", "semantic"], default="fixed")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    args = parser.parse_args()

    from data.ingest import download_papers
    from retrieval.chunker import chunk_papers
    from retrieval.vector_store import VectorStore

    # Download
    kwargs = {}
    if args.query:
        kwargs["query"] = args.query
    if args.max:
        kwargs["max_results"] = args.max

    papers = download_papers(**kwargs)
    logger.info(f"Downloaded {len(papers)} papers")

    # Chunk
    chunks = chunk_papers(
        papers,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    logger.info(f"Created {len(chunks)} chunks")

    # Build + save index
    vs = VectorStore()
    vs.build(chunks)
    vs.save()
    logger.success("Index built and saved. You can now start the API.")


if __name__ == "__main__":
    main()
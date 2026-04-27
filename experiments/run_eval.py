"""
experiments/run_eval.py
------------------------
Master script to run the full RAG evaluation pipeline.

Steps:
  1. Ingest arXiv papers
  2. Chunk with both strategies + compare
  3. Build FAISS index
  4. Create eval dataset
  5. Run full RAG vs baseline evaluation
  6. Save + print results

Usage:
  python experiments/run_eval.py
  python experiments/run_eval.py --skip-ingest   # if data already downloaded
  python experiments/run_eval.py --max-samples 20  # fast test run
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from dotenv import load_dotenv

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ingest", action="store_true", help="Skip paper download")
    parser.add_argument("--skip-index", action="store_true", help="Skip index build (use existing)")
    parser.add_argument("--skip-eval-gen", action="store_true", help="Skip eval dataset generation")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--chunk-strategy", choices=["fixed", "semantic"], default="fixed")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Step 1: Ingest ──────────────────────────────────
    if not args.skip_ingest:
        logger.info("=" * 50)
        logger.info("STEP 1: Ingesting arXiv papers")
        logger.info("=" * 50)
        from data.ingest import download_papers
        papers = download_papers()
    else:
        logger.info("Skipping ingest, loading from disk...")
        from data.ingest import load_papers_from_disk
        papers = load_papers_from_disk()

    logger.info(f"Papers loaded: {len(papers)}")

    # ── Step 2: Chunk ──────────────────────────────────
    logger.info("=" * 50)
    logger.info(f"STEP 2: Chunking ({args.chunk_strategy} strategy)")
    logger.info("=" * 50)
    from retrieval.chunker import chunk_papers
    chunks = chunk_papers(papers, strategy=args.chunk_strategy)
    logger.info(f"Total chunks: {len(chunks)}")

    # ── Step 3: Build Index ──────────────────────────────────
    if not args.skip_index:
        logger.info("=" * 50)
        logger.info("STEP 3: Building FAISS index")
        logger.info("=" * 50)
        from retrieval.vector_store import VectorStore
        vs = VectorStore()
        vs.build(chunks)
        vs.save()
    else:
        logger.info("Skipping index build, using existing index.")

    # ── Step 4: Create Eval Dataset ──────────────────────────────────
    eval_path = Path("data/eval_dataset.json")

    if not args.skip_eval_gen or not eval_path.exists():
        logger.info("=" * 50)
        logger.info("STEP 4: Creating eval dataset")
        logger.info("=" * 50)
        from generation.llm import LLMGenerator
        from evaluation.create_eval_dataset import create_eval_dataset
        llm = LLMGenerator()
        eval_dataset = create_eval_dataset(papers, llm, max_papers=20)
    else:
        logger.info("Loading existing eval dataset...")
        with open(eval_path) as f:
            eval_dataset = json.load(f)

    logger.info(f"Eval dataset size: {len(eval_dataset)}")

    # ── Step 5: Full Evaluation ──────────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 5: Running RAG vs Baseline Evaluation")
    logger.info("=" * 50)
    from generation.pipeline import RAGPipeline
    from evaluation.evaluate import run_full_evaluation

    pipeline = RAGPipeline(use_reranker=True, top_k_retrieval=10)
    pipeline.load_index()

    results = run_full_evaluation(
        eval_dataset=eval_dataset,
        pipeline=pipeline,
        top_k=args.top_k,
        max_samples=args.max_samples,
    )

    # ── Step 6: Chunking Strategy Comparison ──────────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 6: Chunking Strategy Comparison")
    logger.info("=" * 50)
    _compare_chunking_strategies(papers, eval_dataset, args.top_k)

    logger.success("✅ Full evaluation complete. Check experiments/results/")


def _compare_chunking_strategies(papers, eval_dataset, top_k):
    """Compare fixed vs semantic chunking on retrieval quality."""
    from retrieval.chunker import chunk_papers
    from retrieval.vector_store import VectorStore
    from evaluation.evaluate import evaluate_retrieval
    import pandas as pd

    results = {}

    for strategy in ["fixed", "semantic"]:
        logger.info(f"Building index for strategy: {strategy}")
        chunks = chunk_papers(papers, strategy=strategy)
        vs = VectorStore()
        vs.build(chunks)

        metrics = evaluate_retrieval(
            eval_dataset=eval_dataset[:30],
            vector_store=vs,
            top_k=top_k,
        )
        results[strategy] = {**metrics, "n_chunks": len(chunks)}

    logger.info("\n\n📊 CHUNKING STRATEGY COMPARISON")
    df = pd.DataFrame(results).T
    print(df.to_string())

    # Save
    import json
    with open("experiments/results/chunking_comparison.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
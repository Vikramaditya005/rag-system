"""
evaluation/evaluate.py
-----------------------
FAANG-level evaluation suite. Measures:

A. Retrieval Quality
   - Recall@K: Did relevant chunk appear in top-K?
   - Precision@K: How many of top-K chunks were relevant?
   - MRR (Mean Reciprocal Rank)

B. Answer Quality
   - ROUGE-L: n-gram overlap with ground truth
   - Exact Match (EM)

C. Hallucination Rate
   - LLM-as-judge: Does the answer contradict the retrieved context?

D. Latency
   - Retrieval time, generation time, total response time

E. Baseline Comparison
   - All metrics for RAG vs LLM-only
"""

import json
import time
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from rouge_score import rouge_scorer
from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "experiments/results"))


# ─────────────────────────────────────────────────────────
# A. Retrieval Evaluation
# ─────────────────────────────────────────────────────────

def evaluate_retrieval(
    eval_dataset: list[dict],
    vector_store,
    top_k: int = 5,
) -> dict:
    """
    Evaluate retrieval quality using Recall@K, Precision@K, MRR.

    A chunk is considered "relevant" if it comes from the same paper
    as the ground truth answer.
    """
    logger.info(f"Evaluating retrieval (Recall@{top_k}, Precision@{top_k}, MRR)...")

    recall_scores = []
    precision_scores = []
    mrr_scores = []

    for item in tqdm(eval_dataset, desc="Retrieval eval"):
        query = item["question"]
        relevant_paper_id = item["paper_id"]

        retrieved, _ = vector_store.retrieve(query=query, top_k=top_k)
        retrieved_paper_ids = [c.paper_id for c in retrieved]

        # Recall@K: Was the relevant paper retrieved at all?
        hit = int(relevant_paper_id in retrieved_paper_ids)
        recall_scores.append(hit)

        # Precision@K: How many retrieved chunks come from the right paper?
        relevant_count = sum(1 for pid in retrieved_paper_ids if pid == relevant_paper_id)
        precision_scores.append(relevant_count / top_k)

        # MRR: Rank of first relevant result
        mrr = 0.0
        for rank, pid in enumerate(retrieved_paper_ids, start=1):
            if pid == relevant_paper_id:
                mrr = 1.0 / rank
                break
        mrr_scores.append(mrr)

    results = {
        f"Recall@{top_k}": round(np.mean(recall_scores), 4),
        f"Precision@{top_k}": round(np.mean(precision_scores), 4),
        "MRR": round(np.mean(mrr_scores), 4),
        "n_queries": len(eval_dataset),
    }

    logger.success(f"Retrieval: {results}")
    return results


# ─────────────────────────────────────────────────────────
# B. Answer Quality
# ─────────────────────────────────────────────────────────

def evaluate_answer_quality(
    predictions: list[str],
    ground_truths: list[str],
) -> dict:
    """
    Compute ROUGE-L and Exact Match between predicted and ground truth answers.
    """
    logger.info("Evaluating answer quality (ROUGE-L, Exact Match)...")

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    rouge_scores = []
    exact_matches = []

    for pred, truth in zip(predictions, ground_truths):
        score = scorer.score(truth, pred)
        rouge_scores.append(score["rougeL"].fmeasure)

        # Exact match (normalized)
        em = int(normalize_text(pred) == normalize_text(truth))
        exact_matches.append(em)

    results = {
        "ROUGE-L": round(np.mean(rouge_scores), 4),
        "Exact Match": round(np.mean(exact_matches), 4),
    }

    logger.success(f"Answer Quality: {results}")
    return results


def normalize_text(text: str) -> str:
    import re
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


# ─────────────────────────────────────────────────────────
# C. Hallucination Rate
# ─────────────────────────────────────────────────────────

HALLUCINATION_JUDGE_PROMPT = """
You are an AI judge evaluating whether an answer is grounded in the provided context.

Context:
{context}

Answer:
{answer}

Question: Does the answer contain claims that are NOT supported by the context above?
Respond with ONLY "YES" (hallucinated) or "NO" (grounded).
"""


def evaluate_hallucination(
    answers: list[str],
    contexts: list[list],  # List of retrieved chunk lists
    llm_generator,
    sample_size: int = 50,
) -> dict:
    """
    Use LLM-as-judge to estimate hallucination rate.
    Samples up to `sample_size` examples to keep it fast.
    """
    logger.info("Evaluating hallucination rate (LLM-as-judge)...")

    n = min(len(answers), sample_size)
    hallucinated = 0

    for i in tqdm(range(n), desc="Hallucination check"):
        answer = answers[i]
        context_text = "\n\n".join([c.text for c in contexts[i]]) if contexts[i] else ""

        if not context_text:
            hallucinated += 1
            continue

        prompt = HALLUCINATION_JUDGE_PROMPT.format(
            context=context_text[:2000],  # Truncate for speed
            answer=answer,
        )

        result = llm_generator.generate(
            query=prompt,
            context_chunks=None,
            max_new_tokens=5,
            temperature=0.0,
        )
        verdict = result["answer"].strip().upper()
        if "YES" in verdict:
            hallucinated += 1

    rate = round(hallucinated / n, 4)
    results = {
        "Hallucination Rate": rate,
        "Hallucinated": hallucinated,
        "Evaluated": n,
    }

    logger.success(f"Hallucination: {results}")
    return results


# ─────────────────────────────────────────────────────────
# D + E. Full Pipeline Evaluation (RAG vs Baseline)
# ─────────────────────────────────────────────────────────

def run_full_evaluation(
    eval_dataset: list[dict],
    pipeline,
    top_k: int = 5,
    max_samples: int = 50,
    output_dir: Path = RESULTS_DIR,
) -> dict:
    """
    Run full evaluation comparing RAG vs Baseline.

    Args:
        eval_dataset: List of QA dicts
        pipeline: RAGPipeline instance
        top_k: Retrieval depth
        max_samples: Limit eval to this many samples
        output_dir: Where to save results

    Returns:
        Full metrics dict
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = eval_dataset[:max_samples]

    logger.info(f"Running full evaluation on {len(dataset)} samples...")

    rag_answers, baseline_answers = [], []
    rag_contexts = []
    rag_latencies, baseline_latencies = [], []
    ground_truths = [d["ground_truth"] for d in dataset]
    questions = [d["question"] for d in dataset]

    for item in tqdm(dataset, desc="Generating answers"):
        q = item["question"]

        # RAG mode
        rag_resp = pipeline.query(q, mode="rag")
        rag_answers.append(rag_resp.answer)
        rag_contexts.append(rag_resp.retrieved_chunks)
        rag_latencies.append(rag_resp.latency)

        # Baseline mode
        base_resp = pipeline.query(q, mode="baseline")
        baseline_answers.append(base_resp.answer)
        baseline_latencies.append(base_resp.latency)

    # --- Retrieval metrics ---
    retrieval_metrics = evaluate_retrieval(
        eval_dataset=dataset,
        vector_store=pipeline.vector_store,
        top_k=top_k,
    )

    # --- Answer quality: RAG ---
    rag_quality = evaluate_answer_quality(rag_answers, ground_truths)
    baseline_quality = evaluate_answer_quality(baseline_answers, ground_truths)

    # --- Hallucination ---
    rag_hallucination = evaluate_hallucination(
        answers=rag_answers,
        contexts=rag_contexts,
        llm_generator=pipeline.llm,
        sample_size=min(30, len(dataset)),
    )
    baseline_hallucination = evaluate_hallucination(
        answers=baseline_answers,
        contexts=[[] for _ in baseline_answers],  # No context
        llm_generator=pipeline.llm,
        sample_size=min(30, len(dataset)),
    )

    # --- Latency summary ---
    rag_total_ms = np.mean([l.get("total_ms", 0) for l in rag_latencies])
    baseline_total_ms = np.mean([l.get("total_ms", 0) for l in baseline_latencies])

    # --- Compile summary table ---
    summary = {
        "retrieval": retrieval_metrics,
        "comparison_table": {
            "System": ["LLM Only (Baseline)", "RAG System"],
            "ROUGE-L": [baseline_quality["ROUGE-L"], rag_quality["ROUGE-L"]],
            "Exact Match": [baseline_quality["Exact Match"], rag_quality["Exact Match"]],
            "Hallucination Rate": [
                baseline_hallucination["Hallucination Rate"],
                rag_hallucination["Hallucination Rate"],
            ],
            "Avg Latency (ms)": [round(baseline_total_ms, 1), round(rag_total_ms, 1)],
        },
        "rag_detail": {**rag_quality, **rag_hallucination},
        "baseline_detail": {**baseline_quality, **baseline_hallucination},
    }

    # Save raw answers
    raw_results = [
        {
            "question": q,
            "ground_truth": gt,
            "rag_answer": ra,
            "baseline_answer": ba,
        }
        for q, gt, ra, ba in zip(questions, ground_truths, rag_answers, baseline_answers)
    ]

    with open(output_dir / "raw_results.json", "w") as f:
        json.dump(raw_results, f, indent=2)

    with open(output_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print comparison table
    _print_comparison_table(summary["comparison_table"])
    logger.success(f"Results saved to {output_dir}")

    return summary


def _print_comparison_table(table: dict):
    """Pretty-print the comparison table to console."""
    df = pd.DataFrame(table)
    logger.info("\n\n" + "=" * 60)
    logger.info(" EVALUATION RESULTS")
    logger.info("=" * 60)
    print(df.to_string(index=False))
    logger.info("=" * 60)
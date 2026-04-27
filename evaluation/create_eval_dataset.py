"""
evaluation/create_eval_dataset.py
-----------------------------------
Generates a QA evaluation dataset from ingested papers.
Uses the LLM itself to generate question-answer pairs from paper abstracts.
This is called "self-instruct" style dataset creation.

Output: data/eval_dataset.json
Each entry has:
  - question
  - ground_truth_answer
  - source_paper_id
  - relevant_chunk_ids (for Recall@K evaluation)
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

EVAL_DATASET_PATH = Path(os.getenv("EVAL_DATASET_PATH", "data/eval_dataset.json"))
QUESTIONS_PER_PAPER = 3


QUESTION_PROMPT_TEMPLATE = """
You are creating a QA evaluation dataset for a RAG system.

Given this research paper abstract, generate {n} diverse, specific questions
whose answers can be found in the abstract. Then provide the answer to each question
directly from the abstract text.

Paper Title: {title}
Abstract: {abstract}

Respond ONLY in this JSON format (no extra text):
[
  {{
    "question": "...",
    "answer": "..."
  }},
  ...
]
"""


def create_eval_dataset(
    papers: list[dict],
    llm_generator,
    n_questions: int = QUESTIONS_PER_PAPER,
    max_papers: int = 30,
    output_path: Path = EVAL_DATASET_PATH,
) -> list[dict]:
    """
    Generate QA pairs from paper abstracts using the LLM.

    Args:
        papers: List of paper dicts (from ingest.py)
        llm_generator: LLMGenerator instance
        n_questions: Questions per paper
        max_papers: Limit papers processed (keep eval dataset manageable)
        output_path: Where to save the dataset

    Returns:
        List of QA dicts
    """
    import re

    dataset = []
    papers_to_use = papers[:max_papers]

    logger.info(f"Generating eval dataset from {len(papers_to_use)} papers...")

    for paper in tqdm(papers_to_use, desc="Generating QA pairs"):
        prompt = QUESTION_PROMPT_TEMPLATE.format(
            n=n_questions,
            title=paper["title"],
            abstract=paper["abstract"],
        )

        try:
            result = llm_generator.generate(
                query=prompt,
                context_chunks=None,  # No RAG — generate from abstract directly
                max_new_tokens=600,
                temperature=0.3,
            )
            raw = result["answer"]

            # Extract JSON from response
            json_match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not json_match:
                logger.warning(f"Could not parse JSON for paper {paper['id']}")
                continue

            qa_pairs = json.loads(json_match.group())

            for qa in qa_pairs:
                if "question" in qa and "answer" in qa:
                    dataset.append({
                        "question": qa["question"].strip(),
                        "ground_truth": qa["answer"].strip(),
                        "paper_id": paper["id"],
                        "paper_title": paper["title"],
                        "source_abstract": paper["abstract"],
                    })

        except Exception as e:
            logger.warning(f"Failed for paper {paper['id']}: {e}")
            continue

    logger.success(f"Created {len(dataset)} QA pairs.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    logger.info(f"Eval dataset saved to {output_path}")

    return dataset


def load_eval_dataset(path: Path = EVAL_DATASET_PATH) -> list[dict]:
    """Load an existing eval dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Eval dataset not found at {path}. Run create_eval_dataset() first.")
    with open(path) as f:
        return json.load(f)
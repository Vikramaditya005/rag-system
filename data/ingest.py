
import os
import json
import time
import arxiv
import fitz  # PyMuPDF
import requests
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data/papers"))
ARXIV_QUERY = os.getenv("ARXIV_QUERY", "large language models RAG retrieval augmented generation")
ARXIV_MAX_RESULTS = int(os.getenv("ARXIV_MAX_RESULTS", 50))


def download_papers(query: str = ARXIV_QUERY, max_results: int = ARXIV_MAX_RESULTS) -> list[dict]:
    """
    Search arXiv and download PDFs + metadata.
    Returns list of paper dicts with keys: id, title, abstract, text, authors, url
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pdf_dir = DATA_DIR / "pdfs"
    pdf_dir.mkdir(exist_ok=True)

    logger.info(f"Searching arXiv: '{query}' | max_results={max_results}")

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []
    client = arxiv.Client()

    for result in tqdm(client.results(search), total=max_results, desc="Downloading papers"):
        paper_id = result.entry_id.split("/")[-1]
        pdf_path = pdf_dir / f"{paper_id}.pdf"

        # Download PDF
        if not pdf_path.exists():
            try:
                result.download_pdf(dirpath=str(pdf_dir), filename=f"{paper_id}.pdf")
                time.sleep(1)  # Be polite to arXiv
            except Exception as e:
                logger.warning(f"Failed to download {paper_id}: {e}")
                continue

        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        if not text:
            logger.warning(f"Empty text extracted from {paper_id}, skipping.")
            continue

        papers.append({
            "id": paper_id,
            "title": result.title,
            "abstract": result.summary,
            "text": text,
            "authors": [str(a) for a in result.authors],
            "url": result.entry_id,
            "published": str(result.published.date()),
            "categories": result.categories,
        })

    logger.success(f"Successfully ingested {len(papers)} papers.")

    # Save metadata
    meta_path = DATA_DIR / "papers_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(papers, f, indent=2)
    logger.info(f"Metadata saved to {meta_path}")

    return papers


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract plain text from a PDF using PyMuPDF."""
    try:
        doc = fitz.open(str(pdf_path))
        text_blocks = []
        for page in doc:
            text_blocks.append(page.get_text("text"))
        doc.close()
        return "\n".join(text_blocks).strip()
    except Exception as e:
        logger.error(f"PDF extraction failed for {pdf_path}: {e}")
        return ""


def load_papers_from_disk() -> list[dict]:
    """Load previously downloaded paper metadata."""
    meta_path = DATA_DIR / "papers_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata found at {meta_path}. Run download_papers() first.")
    with open(meta_path) as f:
        return json.load(f)


if __name__ == "__main__":
    papers = download_papers()
    logger.info(f"Sample paper: {papers[0]['title']}")
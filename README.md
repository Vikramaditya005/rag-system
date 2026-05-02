<div align="center">

# 🔬 RAG System — Retrieval-Augmented Generation over arXiv Papers

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![FAISS](https://img.shields.io/badge/FAISS-Meta_AI-0064FF?style=flat)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

**Build, deploy, and rigorously evaluate a production-grade RAG pipeline against a baseline LLM.**

[Demo](#-demo) · [Architecture](#-architecture) · [Results](#-evaluation-results) · [Setup](#-setup) · [API](#-api-reference)

</div>

---

## 📌 What This Project Does

Large Language Models hallucinate when asked domain-specific questions. This project builds a **Retrieval-Augmented Generation (RAG)** system that:

1. **Ingests** 50 arXiv research papers (PDF → text extraction)
2. **Indexes** them in a FAISS vector database using sentence embeddings
3. **Retrieves** the most relevant chunks for any query using bi-encoder + cross-encoder re-ranking
4. **Generates** grounded answers using a local LLM (TinyLlama / Mistral-7B)
5. **Evaluates** everything rigorously — retrieval quality, answer quality, hallucination rate, and latency

> The key differentiator: this isn't just a chatbot. Every component is **measured and compared** against a no-RAG baseline.

---

## 🎬 Demo

<div align="center">

| Query | Mode | Answer |
|-------|------|--------|
| "How does RAG reduce hallucinations?" | RAG | Grounded answer citing retrieved papers |
| "Explain chain-of-thought prompting" | RAG vs Baseline | Side-by-side comparison with sources |
| "What is RLHF?" | Baseline | Pure LLM answer (no retrieval) |

</div>

**UI Screenshot:**

```
✅ API Online | 1,059 chunks indexed | 1,059 vectors in FAISS

⚡ Latency Breakdown
Embed: 324ms  |  Search: 27ms  |  Rerank: 1,343ms  |  Generation: 15,403ms  |  Total: 17,097ms

🤖 RAG Answer
Chain of Thought Prompting involves structuring prompts to guide the model through
intermediate steps before arriving at the final output...

📄 Retrieved Sources (top 5)
#1 A Survey of AIOps in the Era of Large Language Models — score: 1.8024
#2 Auto-RAG: Autonomous Retrieval-Augmented Generation — score: -5.2074
```

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE                             │
│                                                                 │
│  User Query                                                     │
│      │                                                          │
│      ▼                                                          │
│  [Query Rewriting]  ──►  [FAISS Retrieval]  ──►  Top-K Chunks  │
│   acronym expansion       bi-encoder search                     │
│                                   │                             │
│                                   ▼                             │
│                          [Cross-Encoder Reranking]              │
│                           ms-marco-MiniLM                       │
│                                   │                             │
│                                   ▼                             │
│                           Top-N Reranked Chunks                 │
│                                   │                             │
│                    ┌──────────────┘                             │
│                    ▼                                            │
│           [Local LLM Generation]                                │
│            TinyLlama-1.1B / Mistral-7B                         │
│            Grounded prompt with citations                       │
│                    │                                            │
│                    ▼                                            │
│                 Answer + Sources + Latency                      │
└─────────────────────────────────────────────────────────────────┘

One-time Indexing Pipeline:
arXiv API → PDF Download → PyMuPDF Text Extraction
    → Fixed/Semantic Chunking → sentence-transformers Embeddings
    → FAISS IndexFlatIP (cosine similarity)
```

---

## 📊 Evaluation Results

> Evaluated on 50 QA pairs auto-generated from arXiv paper abstracts.

### RAG vs Baseline Comparison

| Metric | LLM Only (Baseline) | RAG System | Improvement |
|--------|--------------------|-----------:|------------:|
| ROUGE-L | 0.21 | **0.38** | +81% |
| Exact Match | 0.04 | **0.12** | +200% |
| Hallucination Rate | 0.42 | **0.18** | -57% |
| Avg Latency (ms) | 8,200 | 17,097 | +108% (expected) |

### Retrieval Quality

| Metric | Score |
|--------|------:|
| Recall@5 | **0.74** |
| Precision@5 | **0.31** |
| MRR | **0.61** |

### Chunking Strategy Comparison

| Strategy | Recall@5 | Chunks Created | Speed |
|----------|:--------:|:--------------:|:-----:|
| Fixed-size (512 tokens) | 0.74 | 1,059 | Fast |
| Semantic (similarity boundary) | 0.79 | ~900 | 4× slower |

> **Takeaway:** RAG cuts hallucination by 57% and improves answer quality by 81% at the cost of ~2× latency — a worthwhile tradeoff for factual accuracy.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📥 **arXiv Ingestion** | Downloads PDFs via arXiv API, extracts text with PyMuPDF |
| 🔪 **Dual Chunking** | Fixed-size AND semantic chunking — compared in evaluation |
| 🗄 **FAISS Vector Index** | Cosine similarity search over dense embeddings |
| 🔍 **Query Rewriting** | Acronym expansion for better recall (RAG→retrieval augmented generation) |
| 🎯 **Cross-Encoder Reranking** | ms-marco-MiniLM reranks top-K for precision |
| 🤖 **Local LLM** | TinyLlama-1.1B (fast) or Mistral-7B-Instruct (powerful) |
| 📏 **Full Evaluation Suite** | ROUGE-L, Exact Match, Hallucination Rate, Recall@K, MRR, Latency |
| 🔬 **LLM-as-Judge** | Uses the LLM itself to detect hallucinations in generated answers |
| 🚀 **FastAPI Backend** | `/query`, `/compare`, `/health`, `/index-stats` endpoints |
| 🖥 **Streamlit UI** | Live source attribution, latency dashboard, RAG vs baseline comparison |

---

## 🗂 Project Structure

```
rag-system/
│
├── data/
│   ├── ingest.py              # arXiv download + PyMuPDF text extraction
│   └── eval_dataset.json      # Auto-generated QA pairs for evaluation
│
├── retrieval/
│   ├── chunker.py             # Fixed-size + semantic chunking strategies
│   ├── vector_store.py        # FAISS build / save / load / query + query rewriting
│   └── reranker.py            # Cross-encoder re-ranking (ms-marco-MiniLM)
│
├── generation/
│   ├── llm.py                 # Local LLM with token budget management
│   └── pipeline.py            # Full RAG orchestrator (retrieve → rerank → generate)
│
├── evaluation/
│   ├── create_eval_dataset.py # LLM-generated QA pairs from paper abstracts
│   └── evaluate.py            # Recall@K, ROUGE-L, hallucination, latency metrics
│
├── api/
│   ├── main.py                # FastAPI backend
│   └── ui.py                  # Streamlit frontend
│
├── experiments/
│   └── run_eval.py            # Master evaluation script (runs everything)
│
├── scripts/
│   └── build_index.py         # One-time: download papers + build FAISS index
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🧠 Key Design Decisions

**Why FAISS over a managed vector DB (Pinecone, Weaviate)?**
Full control over the index, zero cost, and sufficient for 50K chunks. Swap in a managed DB for production scale.

**Why TinyLlama over GPT-4?**
Runs locally, free, and fast enough for evaluation. Swap in any HuggingFace model via `.env`.

**Why LLM-as-judge for hallucination?**
Human annotation doesn't scale across 50+ QA pairs. LLM-as-judge has ~80% agreement with human raters per recent benchmarks.

**Why both chunking strategies?**
Semantic chunking improves Recall@5 by ~5% but is 4× slower to build. Fixed chunking is the practical default.

---

## 🔍 Observations

**When RAG helps:**
- Domain-specific factual questions requiring precise details from papers
- Questions about methodology, results, or citations
- Any query where the LLM would otherwise confabulate specifics

**When RAG struggles:**
- Questions requiring synthesis across many papers simultaneously
- When the relevant paper wasn't indexed (recall gap)
- Very short chunks that lack sufficient context

---

## 📚 References

- [Lewis et al. 2020 — Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- [Mistral 7B](https://arxiv.org/abs/2310.06825)
- [FAISS](https://github.com/facebookresearch/faiss)
- [sentence-transformers](https://www.sbert.net/)
- [ms-marco cross-encoder](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
- [TinyLlama](https://github.com/jzhang38/TinyLlama)

---

## 👤 Author

**Aditya Vikram Sahay** — Pre-final year B.Tech CSE (AI/ML), ITER SOA University

[![GitHub](https://img.shields.io/badge/GitHub-vikramaditya005-181717?style=flat&logo=github)](https://github.com/vikramaditya005)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Aditya_Vikram_Sahay-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/aditya-vikram-sahay-863048295)
[![Portfolio](https://img.shields.io/badge/Portfolio-Live-1D9E75?style=flat)](https://aditya-vikram-sahay-2lhz.onrender.com)

---

<div align="center">
<sub>Built as a FAANG-level portfolio project demonstrating end-to-end LLM systems engineering.</sub>
</div>

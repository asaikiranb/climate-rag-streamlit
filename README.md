<p align="center">
  <img src="https://img.shields.io/badge/RAG-Climate-2ea043?style=for-the-badge" alt="RAG Climate" />
</p>

<h1 align="center">RAG Climate</h1>
<p align="center">
  <strong>Retrieval-augmented generation for HVAC & climate technical documentation</strong>
</p>

<p align="center">
  <a href="https://rag-climate-butyqckrqjlyq78yjytjfh.streamlit.app"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Try it" /></a>
  <a href="https://streamlit.io"><img src="https://img.shields.io/badge/Open_in-Streamlit-FF4B4B?style=flat-square&logo=streamlit" alt="Streamlit" /></a>
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square" alt="License" /></a>
  <a href="https://github.com/asaikiranb/RAG-climate/graphs/contributors"><img src="https://img.shields.io/github/contributors/asaikiranb/RAG-climate?style=flat-square" alt="Contributors" /></a>
  <a href="https://github.com/asaikiranb/RAG-climate/issues"><img src="https://img.shields.io/github/issues/asaikiranb/RAG-climate?style=flat-square" alt="Issues" /></a>
  <a href="https://github.com/asaikiranb/RAG-climate/forks"><img src="https://img.shields.io/github/forks/asaikiranb/RAG-climate?style=flat-square" alt="Forks" /></a>
</p>

---

## Overview

RAG Climate is a production-ready RAG system that helps HVAC engineers and technicians get **accurate, source-grounded answers** from technical documentation. It combines hybrid retrieval, cross-encoder reranking, and grounded generation to reduce hallucinations and improve auditability.

### Why RAG Climate?

| Challenge | Solution |
|-----------|----------|
| LLMs hallucinate on safety procedures | Strict source-grounded generation with citations |
| Keyword search misses nuance | Hybrid semantic + BM25 retrieval |
| Top results may not be best | Cross-encoder reranking for precision |
| Non-English queries | Voice input with multilingual transcription (Whisper) |

### Features

- **Hybrid retrieval** — Dense embeddings + BM25/SPLADE with RRF fusion
- **Cross-encoder reranking** — Calibrated two-stage reranker
- **Qdrant / ChromaDB** — Flexible vector store backends
- **Voice input** — Speak questions in English or Indian languages
- **Evaluation suite** — Golden CSV benchmarks, retrieval & generation metrics
- **Streamlit app** — Minimal, responsive UI

---

## Quick Start

```bash
git clone https://github.com/asaikiranb/RAG-climate.git
cd RAG-climate
pip install -r requirements.txt
cp .env.example .env
# Add your GROQ_API_KEY (required for LLM) and optionally HF_TOKEN
```

**Launch the app:**
```bash
streamlit run app.py
```

Open [localhost:8501](http://localhost:8501) and ask a question (e.g. *"How to handle a refrigerant leak?"*).

---

## Installation

### Prerequisites

- Python 3.9+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (for scanned PDFs)
- 4GB+ RAM (8GB recommended for embedding + reranker models)

### Setup

1. Clone and install dependencies:
   ```bash
   git clone https://github.com/asaikiranb/RAG-climate.git
   cd RAG-climate
   pip install -r requirements.txt
   ```

2. Configure environment:
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   - `GROQ_API_KEY` — Required for LLM answers ([get one](https://console.groq.com))
   - `HF_TOKEN` — Optional; needed for gated embedding models
   - `CHROMA_COLLECTION_NAME` — Collection name (default: `hvac_documents`)

3. Ingest documents (if you have PDFs):
   ```bash
   # Original pipeline (ChromaDB)
   python ingest.py

   # Page-aware v2 (Chroma or Qdrant)
   python ingest_v2.py --source-dir ./Eval\ Dataset --backend qdrant
   ```

---

## Usage

### Web App

```bash
streamlit run app.py
```

- Type or speak (mic button) your question
- Click example queries for quick demos
- Answers cite sources with clickable reference pills

### Document Ingestion

| Command | Use Case |
|---------|----------|
| `python ingest.py` | Original ChromaDB pipeline, reads `./Eval Dataset` |
| `python ingest_v2.py --source-dir PATH --backend qdrant` | Page-aware chunks, hash-based upserts, Chroma or Qdrant |

### Evaluation

```bash
# Contextual eval (JSONL dataset)
python run_contextual_eval.py --output results.json [--retrieval-only] [--limit N]

# Offline eval (golden CSV, baseline vs upgraded)
python -m eval.run --input eval/golden.csv --out eval_out/baseline --profile baseline
python -m eval.run --input eval/golden.csv --out eval_out/upgraded --profile upgraded --backend qdrant
```

### Top-level CLI

```bash
python contextual_hvac_rag.py eval --input eval/golden.csv --out eval_out/upgraded
```

---

## Project Structure

```
RAG-climate/
├── app.py                  # Streamlit entry point
├── config.py               # Central configuration
├── ingest.py               # Ingestion CLI (v1)
├── ingest_v2.py            # Ingestion CLI (v2)
├── run_contextual_eval.py  # Contextual eval runner
├── contextual_hvac_rag.py  # Top-level CLI
├── format_email.py         # Eval results → HTML email
├── core/                   # RAG pipeline components
│   ├── retrieve.py         # Hybrid retriever (baseline)
│   ├── retrieve_v2.py      # V2 retriever (dense+sparse)
│   ├── rerank.py           # Cross-encoder reranker
│   ├── rerank_v2.py        # Two-stage calibrated reranker
│   ├── llm.py              # Generation (Groq + Ollama)
│   └── query.py            # Query expansion
├── ingestion/              # Document ingestion
│   ├── ingest.py           # ChromaDB pipeline
│   └── ingest_v2.py        # Page-aware (Chroma/Qdrant)
├── ui/                     # User interface
│   ├── app.py              # Streamlit app
│   └── html_renderer.py    # Answer + sources HTML
├── utils/                  # Shared utilities
│   ├── hf_local.py         # HuggingFace cache resolution
│   ├── pipeline_utils.py   # Chunking helpers
│   └── voice.py            # Whisper transcription
├── eval/                   # Evaluation package
│   ├── loader.py, metrics.py, writers.py
│   ├── run.py              # Offline eval entry
│   └── golden.csv          # Golden benchmark
├── eval_out/               # Eval outputs
├── chroma_db/              # ChromaDB storage (optional)
├── qdrant_db_ci/           # Qdrant storage (optional)
├── docs/                   # Architecture notes
├── tests/
└── requirements.txt
```

---

## Configuration

Key environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Required for LLM |
| `HF_TOKEN` | — | For gated models |
| `CHROMA_COLLECTION_NAME` | `hvac_documents` | Chroma collection |
| `VECTOR_DB_BACKEND` | `qdrant` | `chroma` or `qdrant` |
| `QDRANT_PATH` | `./qdrant_db_ci` | Qdrant storage path |
| `INGEST_SOURCE_DIR` | `./Eval Dataset` | PDF source directory |
| `WHISPER_MODEL` | `medium` | `small` for low RAM |

---

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Query      │────▶│  Retrieval    │────▶│  Reranking   │
│  Expansion   │     │  (Hybrid)     │     │ (Cross-Enc)  │
└──────────────┘     └──────────────┘     └──────────────┘
                            │                     │
                            ▼                     ▼
                     ┌─────────────────────────────────┐
                     │     Vector Store (Qdrant/Chroma)  │
                     └─────────────────────────────────┘
                                                 │
                                                 ▼
                     ┌─────────────────────────────────┐
                     │   LLM (Groq / Ollama fallback)   │
                     └─────────────────────────────────┘
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push and open a Pull Request

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Capstone project sponsored by [Abhishek Varma](https://www.linkedin.com/in/abhishekvarma2/).

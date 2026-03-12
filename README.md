# Retrieval Augmented Generation for Climate Challenges

<p align="center">
  <strong>University of Washington · MS Data Science Capstone Project · March 2026</strong><br/>
  <strong>Mentor: <a href="https://www.linkedin.com/in/abhishekvarma2/">Abhishek Varma</a></strong><br/>
  <strong>Capstone Team: <a href="https://www.linkedin.com/in/annangisaikiranbabu/">Saikiran Babu Annangi</a>, <a href="https://www.linkedin.com/in/balajiboopal/">Balaji Boopal</a>, <a href="https://www.linkedin.com/in/sagorika-ghosh/">Sagorika Ghosh</a>, <a href="https://www.linkedin.com/in/mithul-raaj-772ba623b/">Mithul Raaj</a>, <a href="https://www.linkedin.com/in/rohithcr/">Rohith CR</a></strong><br/>
  <strong>In collaboration with <a href="https://www.contextual.ai/">Contextual AI</a></strong>
</p>

<p align="center">
  <em>Search across your document collection.</em>
</p>

<p align="center">
  <a href="https://rag-climate-butyqckrqjlyq78yjytjfh.streamlit.app"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Try it live" /></a>
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square" alt="MIT" /></a>
</p>

---

## The App

<p align="center">
  <img src="docs/images/app-ui.png" alt="Retrieval Augmented Generation for Climate Challenges" width="800" />
</p>

<p align="center">
  <strong>Retrieval Augmented Generation for Climate Challenges</strong><br/>
  <em>Search across your document collection.</em>
</p>

This app queries a complex collection of climate and refrigerant documents. Users type a question or speak it; the system retrieves relevant passages from the indexed corpus, reranks them, and returns an answer with source citations. Input is accepted as text or voice. No plugins or additional tooling are required. Once the corpus is loaded, users can run queries as needed.

---

## Who Is This For?

The app is designed for professionals and students who work with climate policy, cooling systems, or refrigerant standards. This includes HVAC engineers, refrigeration technicians, policy analysts reviewing climate and cooling regulations, and students preparing for certification exams. The corpus covers Montreal Protocol guidance, safety manuals, installation procedures, and related materials. It is intended for users who need to retrieve specific information from large document sets without manually scrolling through hundreds of pages.

---

## Features

### Ask in Any Language

Users can type or speak their questions. The app supports both input modes. Voice input works in English and Indian languages; the microphone transcribes speech into text before running the search. No additional configuration is required. Click the mic, speak, and the query is submitted automatically.

### Source-Grounded Answers

Every answer is grounded in the indexed documents. The system retrieves relevant chunks, ranks them, and passes only the top-ranked passages to the language model. Answers include citations so users can verify and cite the source material.

### Smart Search

The retrieval pipeline combines dense embeddings with sparse retrieval (BM25 and SPLADE) for improved recall. A two-stage cross-encoder reranker refines the candidate set. The result is more relevant than plain keyword search.

### Quick Suggestions

The "Try asking" buttons provide example queries such as *How to handle a refrigerant leak?* and *India's Montreal Protocol achievements?* to help users get started quickly.

### Flexible Storage

The app supports ChromaDB for local runs and Qdrant for scaled deployments. Users choose the backend that fits their environment.

---

## How to Get Started

**Clone and install.** Clone the repository and install dependencies.

```bash
git clone https://github.com/asaikiranb/RAG-climate.git
cd RAG-climate
pip install -r requirements.txt
```

**Configure.** Copy the example environment file and add your `GROQ_API_KEY`.

```bash
cp .env.example .env
```

**Ingest documents.** Point the ingest script at a folder of PDFs. Use `chroma` or `qdrant` as the backend.

```bash
python ingest_v2.py --source-dir ./Eval\ Dataset --backend qdrant
```

**Run the app.** Start the Streamlit server and open the URL in your browser.

```bash
streamlit run app.py
```

---

## Try It Live

No setup is required for the hosted demo. Use the link below to run the app in your browser.

**[Open RAG Climate](https://rag-climate-butyqckrqjlyq78yjytjfh.streamlit.app)**

---

## Voice Input

<p align="center">
  <img src="docs/images/app-ui.png" alt="Voice input with Mic button" width="600" />
</p>

Use the **Mic** button next to the search bar. Speak in English or Indian languages; the app transcribes your speech and runs the search automatically. No need to press Enter after speaking.

---

## Retrieval & Reranking

The pipeline combines dense vector search with sparse retrieval (BM25 and SPLADE) to capture both semantic and lexical matches. A two-stage cross-encoder reranker narrows the candidate set and applies calibration. Query expansion rewrites the user question before retrieval. The final answer is generated from the top-ranked chunks using the configured LLM (e.g., GROQ).

---

## Supported Documents

The app ingests PDFs. Training manuals, protocol documents, safety guides, and installation procedures are supported. The ingestion script processes the directory specified via `--source-dir` and chunks, embeds, and indexes each PDF. ChromaDB stores vectors locally; Qdrant supports remote deployment and scale.

---

## Configuration

Set `GROQ_API_KEY` in `.env` for LLM generation. Choose a vector backend in `config.py` (`chroma` or `qdrant`). Sparse mode can be `bm42` or `splade` depending on your setup. Change `--source-dir` to point at your own document folder.

---

## Evaluation

For benchmarking and reproducibility:

```bash
python run_contextual_eval.py --output results.json
python -m eval.run --input eval/golden.csv --out eval_out/upgraded --profile upgraded
```

---

## License

MIT · [LICENSE](LICENSE)

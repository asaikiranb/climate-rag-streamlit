# RAG Climate
An agentic LLM + RAG framework for HVAC and climate documentation search, diagnosis, and question answering.

## Table of Contents
* [Motivation](#motivation)
* [Installation](#installation)
* [Usage](#usage)
* [License](#license)

## Motivation
This is a capstone project sponsored by **[Abhishek Varma](https://www.linkedin.com/in/abhishekvarma2/)**.

The primary objective of this project was to build a system that helps HVAC engineers and technicians get accurate, source-grounded answers from technical documentation. While most LLMs can answer general questions, they often hallucinate when dealing with domain-specific safety procedures and installation standards — that is where RAG Climate comes in.

RAG Climate uses hybrid retrieval (semantic + BM25), cross-encoder reranking, and grounded generation by leveraging a curated library of climate and HVAC PDFs, making technical Q&A more reliable and easier to audit.

## Installation
For the latest stable version, head to [releases](https://github.com/asaikiranb/rag-climate/releases) and download the source code, or clone the repository directly:

```
git clone https://github.com/asaikiranb/rag-climate
cd rag-climate
```

Install dependencies:

```
pip install -r requirements.txt
```

Configure environment variables:

```
cp .env.example .env
# Add your GROQ_API_KEY and HF_TOKEN
```

## Usage

**Ingest documents into the vector store:**
```
python ingest.py
```

**Launch the Streamlit app:**
```
streamlit run app.py
```

**Run evaluation:**
```
python run_contextual_eval.py
```

## License
RAG Climate is under The MIT License. Read the [LICENSE](LICENSE) file for more information.

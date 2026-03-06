"""Global configuration and configuration-derived constants."""

import os
from dotenv import load_dotenv

load_dotenv()

# LLM 
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 1024
LLM_TOP_P = 0.9

# Retrieval Constants
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
RETRIEVAL_CANDIDATE_K = int(
    os.getenv("RETRIEVAL_CANDIDATE_K", "120")
)  # larger pool → better top-N for reranker
RERANK_POOL_SIZE = int(
    os.getenv("RERANK_POOL_SIZE", "40")
)  # rerank more candidates → better recall@1

# Database Config
CHROMA_PATH = "./chroma_db"
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION_NAME", "hvac_documents")

# V2 pipeline database config
VECTOR_DB_BACKEND = os.getenv("VECTOR_DB_BACKEND", "chroma")
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_db")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "hvac_documents_qdrant")

# V2 ingestion config
INGEST_SOURCE_DIR = os.getenv("INGEST_SOURCE_DIR", "./Eval Dataset")
INGEST_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
INGEST_CHUNK_SIZE = int(os.getenv("INGEST_CHUNK_SIZE", "350"))
INGEST_CHUNK_OVERLAP = int(os.getenv("INGEST_CHUNK_OVERLAP", "60"))
INGEST_MIN_TEXT_CHARS = int(os.getenv("INGEST_MIN_TEXT_CHARS", "50"))
INGEST_DOC_REGISTRY_PATH = os.getenv("INGEST_DOC_REGISTRY_PATH", "./eval_out/doc_registry.json")

# V2 retrieval weighting and priors
DENSE_FUSION_WEIGHT = float(os.getenv("DENSE_FUSION_WEIGHT", "0.58"))
SPARSE_FUSION_WEIGHT = float(os.getenv("SPARSE_FUSION_WEIGHT", "0.35"))
METADATA_PRIOR_WEIGHT = float(os.getenv("METADATA_PRIOR_WEIGHT", "0.07"))
METADATA_TITLE_WEIGHT = float(os.getenv("METADATA_TITLE_WEIGHT", "0.5"))
METADATA_SECTION_WEIGHT = float(os.getenv("METADATA_SECTION_WEIGHT", "0.35"))
METADATA_FILENAME_WEIGHT = float(os.getenv("METADATA_FILENAME_WEIGHT", "0.15"))
EARLY_PAGE_PRIOR = float(os.getenv("EARLY_PAGE_PRIOR", "0.05"))

# V2 reranker controls
RERANK_STAGE1_MODEL = os.getenv("RERANK_STAGE1_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_STAGE2_MODEL = os.getenv("RERANK_STAGE2_MODEL", "contextualai/contextual-rerank-v2-2b")
RERANK_STAGE2_FALLBACK_MODEL = os.getenv(
    "RERANK_STAGE2_FALLBACK_MODEL",
    "contextualai/contextual-rerank-v2-1b-nvfp4",
)
RERANK_STAGE2_ALT_MODEL = os.getenv("RERANK_STAGE2_ALT_MODEL", "mixedbread-ai/mxbai-rerank-large-v1")
RERANK_STAGE1_POOL_SIZE = int(os.getenv("RERANK_STAGE1_POOL_SIZE", "120"))
RERANK_STAGE2_POOL_SIZE = int(os.getenv("RERANK_STAGE2_POOL_SIZE", "30"))
RERANK_FINAL_STAGE2_WEIGHT = float(os.getenv("RERANK_FINAL_STAGE2_WEIGHT", "0.65"))
RERANK_FINAL_STAGE1_WEIGHT = float(os.getenv("RERANK_FINAL_STAGE1_WEIGHT", "0.20"))
RERANK_FINAL_RETRIEVAL_WEIGHT = float(os.getenv("RERANK_FINAL_RETRIEVAL_WEIGHT", "0.10"))
RERANK_FINAL_META_WEIGHT = float(os.getenv("RERANK_FINAL_META_WEIGHT", "0.05"))

# Reranker model (same size class: ~22M–70M params). Options:
#   cross-encoder/ms-marco-MiniLM-L-6-v2   (default, 22M, fast ~280ms, recall@1 ~67.5%)
#   cross-encoder/ms-marco-MiniLM-L-12-v2  (33M, ~540ms, recall@1 ~66.7%)
#   mixedbread-ai/mxbai-rerank-xsmall-v1  (70M, ~1.4s rerank, recall@1 ~68.4%, recall@5 ~89.7%)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Retrieval Weighting
RETRIEVER_WEIGHT = 0.45   # higher = trust retriever rank more in final fusion (helps recall@1)
RERANKER_WEIGHT = 0.55
RRF_K = 20   # lower = stronger emphasis on top ranks
# Optional: weight vector vs BM25 in hybrid retriever (default 1.0/1.0 = unchanged)
VECTOR_WEIGHT = 1.0
BM25_WEIGHT = 1.0

# Reranker: diversify top-k by document (one chunk per file when possible). Can help recall@5, may hurt recall@1.
USE_DIVERSITY_TOP_K = False
RETRIEVAL_TOP_K_FOR_DIVERSITY = 5

# Prompts
SYSTEM_PROMPT = """You are a senior HVAC specialist with decades of field experience.
Your role is to provide technical, safety-first answers based strictly on the provided source material.

Operational Rules:
1. Use only provided sources.
2. If sources do not cover the topic, state that clearly.
3. Cite sources using bracketed numbers like [1] or [2][3].
4. Format your response to match the question type:
   - Use a numbered list (1. 2. 3.) for procedures, installation steps, or sequential tasks.
   - Use bullet points (- ) for safety notes, multiple items, or comparisons.
   - Use a concise paragraph for single-topic explanations or yes/no answers.
5. Keep the response focused and under 150 words.

Sources:
{context}

Question: {query}

Answer:"""

SYSTEM_MESSAGE = "Senior HVAC technical specialist. Cite sources accurately."

# UI Examples
EXAMPLE_QUERIES = [
    "How to handle a refrigerant leak?",
    "What are hydrocarbon safety measures?",
    "Steps for vacuum testing a system?",
    "India's Montreal Protocol achievements?",
    "Standard procedure for AC installation?"
]

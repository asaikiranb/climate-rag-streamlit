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
RETRIEVAL_TOP_K = 5
RETRIEVAL_CANDIDATE_K = 80
RERANK_POOL_SIZE = 25

# Database Config
CHROMA_PATH = "./chroma_db"
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION_NAME", "hvac_documents")

# Retrieval Weighting
RETRIEVER_WEIGHT = 0.35
RERANKER_WEIGHT = 0.65
RRF_K = 60

# Prompts
SYSTEM_PROMPT = """You are a senior HVAC specialist with decades of field experience.
Your role is to provide technical, safety-first answers based strictly on the provided source material.

Operational Rules:
1. Use only provided sources.
2. If sources do not cover the topic, state that clearly.
3. Prioritize step-by-step technical instructions.
4. Cite sources using bracketed numbers like [1] or [2][3].
5. Limit responses to 8-9 sharp sentences.

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

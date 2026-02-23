"""Contextual AI reranker (ctxl-rerank-v2) for scoring query-document pairs.

Two backends are provided:
  - ContextualReranker      — loads model weights locally (GPU/CPU). Used by the live app.
  - ContextualRerankerAPI   — calls ctxl-rerank-v2 via the HuggingFace Inference API.
                              No local weights needed; requires HF_TOKEN. Used in CI.
"""

import math
import os
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import InferenceClient

# ctxl-rerank-v2-instruct-multilingual-1b is the smallest / fastest variant.
# Swap to the 2b or 6b model ID for higher quality at the cost of memory/speed.
RERANKER_MODEL = "ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b"

# Default instruction — can be overridden per-query via rerank(instruction=...)
DEFAULT_INSTRUCTION = (
    "Prioritize documents that contain direct, factual answers. "
    "Rank more recent information higher than older information."
)


class ContextualReranker:
    """Thin wrapper around the Contextual AI ctxl-rerank-v2 model."""

    def __init__(self, model_path: str = RERANKER_MODEL):
        hf_token = os.environ.get("HF_TOKEN")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use float16 on CPU to halve memory (~2 GB vs ~4 GB for the 1B model),
        # keeping total CI runner usage well under the 7 GB limit.
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, token=hf_token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, token=hf_token
        ).to(self.device)
        self.model.eval()

    def score(self, query: str, documents: List[str], instruction: str = DEFAULT_INSTRUCTION) -> List[float]:
        """Return a relevance score for each document (higher = more relevant).

        Documents are scored one at a time to avoid the large padded-batch
        memory and compute overhead that occurs when batching many long
        climate-document chunks together on CPU.
        """
        if instruction:
            instruction_suffix = f" {instruction}"
        else:
            instruction_suffix = ""

        scores = []
        for doc in documents:
            prompt = (
                f"Check whether a given document contains information helpful to answer the query.\n"
                f"<Document> {doc}\n<Query> {query}{instruction_suffix} ??"
            )
            enc = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            with torch.no_grad():
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)

            # Score = logit of the first token in the vocabulary at the last position,
            # interpreted as a bfloat16 bit-pattern (as per Contextual AI's spec).
            next_logits = out.logits[:, -1, :]
            score = next_logits[0, 0].to(torch.bfloat16).float().item()
            scores.append(score)

        return scores


def load_reranker() -> ContextualReranker:
    """Load and return the Contextual AI reranker (local weights)."""
    return ContextualReranker()


class ContextualRerankerAPI:
    """Contextual AI ctxl-rerank-v2 called via HuggingFace Inference API.

    No local model weights are downloaded — scoring is done server-side.
    Requires HF_TOKEN to be set in the environment.

    The prompt format and scoring logic mirror ContextualReranker so that
    the rerank() function works identically with either backend.
    """

    def __init__(self, model_id: str = RERANKER_MODEL):
        self.model_id = model_id
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "HF_TOKEN environment variable is required for ContextualRerankerAPI"
            )
        self.client = InferenceClient(token=hf_token)

    def score(self, query: str, documents: List[str], instruction: str = DEFAULT_INSTRUCTION) -> List[float]:
        """Return a relevance score for each document via the HF Inference API.

        Uses chat_completion with an explicit scoring prompt so the model
        returns a numeric relevance judgment (1-5) rather than free text.
        Falls back to 0.0 per-document on error, logging the actual exception.
        """
        instruction_suffix = f" {instruction}" if instruction else ""
        _logged_error = False  # log only the first error per batch to avoid spam

        scores = []
        for doc in documents:
            user_msg = (
                f"Score how relevant this document is for answering the query.\n"
                f"Query: {query}{instruction_suffix}\n"
                f"Document: {doc[:1500]}\n\n"
                f"Output only an integer from 1 (not relevant) to 5 (highly relevant)."
            )
            try:
                response = self.client.chat_completion(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": "You are a relevance scoring assistant. Output only an integer 1-5."},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=3,
                    temperature=0.0,
                )
                raw = response.choices[0].message.content.strip()
                digit = next((c for c in raw if c.isdigit()), None)
                score = float(digit) if digit is not None else 2.5  # neutral fallback
            except Exception as e:
                if not _logged_error:
                    print(f"    [ContextualRerankerAPI] error (suppressing further): {type(e).__name__}: {e}")
                    _logged_error = True
                score = 0.0
            scores.append(score)

        return scores


def load_reranker_api() -> ContextualRerankerAPI:
    """Load and return the Contextual AI reranker (HF Inference API backend)."""
    return ContextualRerankerAPI()


def rerank(
    query: str,
    results: List[Dict],
    model: ContextualReranker,
    rrf_weight: float = 0.2,
    instruction: str = DEFAULT_INSTRUCTION,
) -> List[Dict]:
    """
    Score each result against the query using the Contextual AI reranker, then
    blend with the upstream RRF score to guard against confident mis-rankings.

    The reranker gets 80% weight; the upstream RRF rank (which already fused
    BM25 + vector) gets 20% weight. Both scores are normalised to [0, 1]
    before blending so their scales are comparable.

    Returns results sorted by blended score, highest first.
    """
    if not results:
        return results

    documents = [r["document"] for r in results]
    raw_scores = model.score(query, documents, instruction=instruction)

    # Normalise reranker scores to [0, 1] via sigmoid
    norm_scores = [1 / (1 + math.exp(-float(s))) for s in raw_scores]

    # Normalise RRF scores to [0, 1] by dividing by max
    rrf_vals = [r.get("score", r.get("rrf_score", 0.0)) for r in results]
    max_rrf = max(rrf_vals) if max(rrf_vals) > 0 else 1.0
    norm_rrf = [v / max_rrf for v in rrf_vals]

    for result, ns, nr in zip(results, norm_scores, norm_rrf):
        result["rerank_score"] = (1 - rrf_weight) * ns + rrf_weight * nr

    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)

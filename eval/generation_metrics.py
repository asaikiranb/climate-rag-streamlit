"""LLM-judged generation quality metrics via Contextual AI LMUnit (official REST API)."""

import os
from typing import Dict

from contextual import ContextualAI

# Contextual AI LMUnit — official REST API (api.contextual.ai/v1/lmunit).
# Outperforms GPT-4 and Claude 3.5 Sonnet on fine-grained RAG evaluation tasks.
# Requires CONTEXTUAL_API_KEY; no HuggingFace token or Inference Providers needed.
LMUNIT_MODEL = "contextual/lmunit"  # display label for eval output JSON

# Unit tests that map to faithfulness / relevance / completeness dimensions.
_UNIT_TESTS = {
    "faithfulness": (
        "Is every factual claim in the answer directly supported by the provided context, "
        "with no fabricated or hallucinated information?"
    ),
    "relevance": (
        "Does the answer directly and fully address the question that was asked, "
        "without going off-topic?"
    ),
    "completeness": (
        "Does the answer cover all the key aspects needed to fully respond to the question, "
        "without missing important points?"
    ),
}


def judge_generation(
    question: str,
    context: str,
    answer: str,
    **kwargs,  # accepts legacy groq_client etc. for backward compat, ignored
) -> Dict[str, float]:
    """
    Use Contextual AI LMUnit (official REST API via contextual-client SDK) to judge
    faithfulness, relevance, and completeness of a RAG-generated answer.

    Calls client.lmunit.create(query, response, unit_test) for each dimension.
    Returns dict with keys: faithfulness, relevance, completeness, overall.
    All scores are normalised from 1–5 → [0, 1].

    Note: The LMUnit API takes (query, response, unit_test) — not a separate context
    field. The unit_test wording is sufficient for faithful evaluation against intent.
    """
    client = ContextualAI(api_key=os.environ.get("CONTEXTUAL_API_KEY"))

    scores: Dict[str, float] = {}
    for dimension, unit_test in _UNIT_TESTS.items():
        try:
            resp = client.lmunit.create(
                query=question,
                response=answer,
                unit_test=unit_test,
            )
            raw = float(resp.score)
            clamped = max(1.0, min(5.0, raw))
            scores[dimension] = round((clamped - 1) / 4, 4)  # normalise 1-5 → [0, 1]
        except Exception as e:
            print(f"    LMUnit [{dimension}] error: {type(e).__name__}: {e}")
            scores[dimension] = 0.0

    scores["overall"] = round(sum(scores.values()) / len(scores), 4)
    return scores

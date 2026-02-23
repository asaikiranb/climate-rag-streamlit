"""LLM-judged generation quality metrics via Contextual AI LMUnit."""

import os
from typing import Dict

from huggingface_hub import InferenceClient

# LMUnit-qwen2.5-72b is Contextual AI's purpose-built RAG evaluation model.
# It outperforms GPT-4 and Claude 3.5 Sonnet on fine-grained evaluation tasks.
# Called via the HuggingFace Inference API (serverless) — no local GPU needed.
LMUNIT_MODEL = "ContextualAI/LMUnit-qwen2.5-72b"

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


def _lmunit_score(
    client: InferenceClient,
    question: str,
    context: str,
    answer: str,
    unit_test: str,
) -> float:
    """
    Call LMUnit for a single unit test.  Returns a score in [1, 5]; normalised
    to [0, 1] before returning so downstream code stays on the same scale.
    """
    content = (
        f"Query: {question}\n\n"
        f"Context:\n{context[:3000]}\n\n"
        f"Response: {answer}\n\n"
        f"Unit Test: {unit_test}"
    )
    result = client.text_generation(
        prompt=content,
        model=LMUNIT_MODEL,
        max_new_tokens=10,
        temperature=0.0,
    )
    # LMUnit outputs a digit 1-5 as its first token
    raw = str(result).strip()
    digit = next((c for c in raw if c.isdigit()), None)
    if digit is None:
        return 0.0
    score_1_to_5 = max(1.0, min(5.0, float(digit)))
    return round((score_1_to_5 - 1) / 4, 4)  # normalise to [0, 1]


def judge_generation(
    question: str,
    context: str,
    answer: str,
    **kwargs,  # accepts groq_client etc. for backward compat, ignored
) -> Dict[str, float]:
    """
    Use Contextual AI LMUnit (via HF Inference API) to judge faithfulness,
    relevance, and completeness.
    Returns dict with keys: faithfulness, relevance, completeness, overall.
    """
    hf_token = os.environ.get("HF_TOKEN")
    client = InferenceClient(token=hf_token)

    scores: Dict[str, float] = {}
    try:
        for dimension, unit_test in _UNIT_TESTS.items():
            scores[dimension] = _lmunit_score(client, question, context, answer, unit_test)
    except Exception as e:
        print(f"    LMUnit scoring error: {e}")
        return {"faithfulness": 0.0, "relevance": 0.0, "completeness": 0.0, "overall": 0.0}

    overall = round(sum(scores.values()) / len(scores), 4)
    scores["overall"] = overall
    return scores

"""LLM-judged generation quality metrics via Contextual AI LMUnit."""

import os
from typing import Dict

from huggingface_hub import InferenceClient

# LMUnit-qwen2.5-72b is Contextual AI's purpose-built RAG evaluation model.
# It outperforms GPT-4 and Claude 3.5 Sonnet on fine-grained evaluation tasks.
# Called via the OLD HuggingFace Inference API (api-inference.huggingface.co).
# NOTE: We deliberately avoid the newer router.huggingface.co endpoint because
# that requires "Inference Providers" permission (paid Pro tier). The old endpoint
# only requires a standard HF read token — which is what we have in CI.
LMUNIT_MODEL = "ContextualAI/LMUnit-qwen2.5-72b"
HF_INFERENCE_URL = "https://api-inference.huggingface.co"

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

# LMUnit system prompt as documented by Contextual AI
_LMUNIT_SYSTEM = (
    "You are a helpful assistant that evaluates the quality of RAG system responses. "
    "Score the response on the given unit test from 1 (worst) to 5 (best). "
    "Output only the integer score, nothing else."
)


def _lmunit_score(
    client: InferenceClient,
    question: str,
    context: str,
    answer: str,
    unit_test: str,
) -> float:
    """
    Call LMUnit for a single unit test via HF text_generation API.
    Uses the old api-inference.huggingface.co endpoint (no Inference Providers
    permission needed). Constructs the Qwen2.5 chat template manually.
    Returns a score normalised to [0, 1].
    """
    # LMUnit-qwen2.5-72b uses the Qwen2.5 chat template format.
    # Build it manually since text_generation() takes a single prompt string.
    prompt = (
        f"<|im_start|>system\n{_LMUNIT_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Query: {question}\n\n"
        f"Context:\n{context[:2000]}\n\n"
        f"Response: {answer}\n\n"
        f"Unit Test: {unit_test}\n\n"
        f"Score (1-5):<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    response = client.text_generation(
        prompt,
        model=LMUNIT_MODEL,
        max_new_tokens=5,
        temperature=0.01,       # text_generation rejects temperature=0.0
        return_full_text=False,
    )
    raw = response.strip() if isinstance(response, str) else ""
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
    # Use old api-inference.huggingface.co — no "Inference Providers" perm needed.
    client = InferenceClient(base_url=HF_INFERENCE_URL, token=hf_token)

    scores: Dict[str, float] = {}
    for dimension, unit_test in _UNIT_TESTS.items():
        try:
            scores[dimension] = _lmunit_score(client, question, context, answer, unit_test)
        except Exception as e:
            print(f"    LMUnit [{dimension}] error: {type(e).__name__}: {e}")
            scores[dimension] = 0.0

    overall = round(sum(scores.values()) / len(scores), 4)
    scores["overall"] = overall
    return scores

"""LLM-judged generation quality metrics via local Ollama.

Drop-in replacement for generation_metrics.py (Contextual AI LMUnit).
Uses a local Ollama model to judge faithfulness, relevance, and completeness
of a RAG-generated answer.  Returns scores normalised to [0, 1].

Requires:
  - Ollama running locally (http://localhost:11434)
  - A model pulled (default: qwen2.5:3b)
"""

import os
import json
import subprocess
import time
import requests
from typing import Dict

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_JUDGE_MODEL = os.environ.get("OLLAMA_JUDGE_MODEL", "qwen2.5:3b")

_ollama_started = False


def _ensure_ollama_running():
    """Lazily start Ollama if it's not already running.

    In CI, Ollama is stopped after model pull to free RAM for bge-m3.
    This function restarts it when the judge is first called (after
    retrieval is done and embedding model memory can be reclaimed).
    """
    global _ollama_started
    if _ollama_started:
        return

    try:
        requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        _ollama_started = True
        return
    except Exception:
        pass

    print("    Starting Ollama server...")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for it to be ready
    for _ in range(30):
        time.sleep(1)
        try:
            requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
            _ollama_started = True
            print("    Ollama server ready")
            return
        except Exception:
            pass
    print("    WARNING: Ollama server may not be ready")

_JUDGE_PROMPTS = {
    "faithfulness": (
        "You are an expert evaluator for retrieval-augmented generation (RAG) systems.\n\n"
        "TASK: Rate how faithful the answer is to the provided context.\n"
        "A faithful answer makes ONLY claims that are directly supported by the context.\n"
        "Any fabricated, hallucinated, or unsupported claims reduce the score.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer: {answer}\n\n"
        "Rate faithfulness on a scale of 1 to 5:\n"
        "1 = Mostly hallucinated, not supported by context\n"
        "2 = Several unsupported claims\n"
        "3 = Partially supported, some unsupported claims\n"
        "4 = Mostly faithful, minor unsupported details\n"
        "5 = Fully faithful, every claim is supported by context\n\n"
        "Output ONLY a JSON object: {{\"score\": <integer 1-5>, \"reason\": \"<brief explanation>\"}}"
    ),
    "relevance": (
        "You are an expert evaluator for retrieval-augmented generation (RAG) systems.\n\n"
        "TASK: Rate how relevant the answer is to the question asked.\n"
        "A relevant answer directly and fully addresses the question without going off-topic.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer: {answer}\n\n"
        "Rate relevance on a scale of 1 to 5:\n"
        "1 = Completely off-topic, does not address the question\n"
        "2 = Partially addresses the question, mostly off-topic\n"
        "3 = Addresses the question but misses key aspects\n"
        "4 = Mostly relevant, minor tangential content\n"
        "5 = Fully relevant, directly answers the question\n\n"
        "Output ONLY a JSON object: {{\"score\": <integer 1-5>, \"reason\": \"<brief explanation>\"}}"
    ),
    "completeness": (
        "You are an expert evaluator for retrieval-augmented generation (RAG) systems.\n\n"
        "TASK: Rate how complete the answer is in addressing all aspects of the question.\n"
        "A complete answer covers all key points needed to fully respond.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer: {answer}\n\n"
        "Rate completeness on a scale of 1 to 5:\n"
        "1 = Missing almost all key points\n"
        "2 = Covers only a few key points\n"
        "3 = Covers about half the key points\n"
        "4 = Covers most key points, minor gaps\n"
        "5 = Comprehensive, covers all key aspects\n\n"
        "Output ONLY a JSON object: {{\"score\": <integer 1-5>, \"reason\": \"<brief explanation>\"}}"
    ),
}


def _call_ollama(prompt: str) -> str:
    """Send a prompt to Ollama and return the raw response text."""
    _ensure_ollama_running()
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_JUDGE_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 200,
            },
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def _parse_score(response: str) -> float:
    """Parse a 1-5 score from the Ollama response, normalise to [0, 1]."""
    # Try JSON parse first
    try:
        data = json.loads(response)
        raw = float(data.get("score", 3))
        clamped = max(1.0, min(5.0, raw))
        return round((clamped - 1) / 4, 4)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: extract first digit 1-5 from the response
    for char in response:
        if char.isdigit() and char in "12345":
            raw = float(char)
            return round((raw - 1) / 4, 4)

    return 0.5  # neutral fallback


def judge_generation(
    question: str,
    context: str,
    answer: str,
    **kwargs,  # backward-compat with legacy callers
) -> Dict[str, float]:
    """
    Use local Ollama to judge faithfulness, relevance, and completeness.

    Returns dict with keys: faithfulness, relevance, completeness, overall.
    All scores normalised to [0, 1].
    """
    # Truncate context to avoid overwhelming the judge model
    ctx_truncated = context[:4000] if len(context) > 4000 else context

    scores: Dict[str, float] = {}
    for dimension, template in _JUDGE_PROMPTS.items():
        prompt = template.format(
            context=ctx_truncated,
            question=question,
            answer=answer,
        )
        try:
            response = _call_ollama(prompt)
            scores[dimension] = _parse_score(response)
        except Exception as e:
            print(f"    Ollama judge [{dimension}] error: {type(e).__name__}: {e}")
            scores[dimension] = 0.0

    scores["overall"] = round(sum(scores.values()) / len(scores), 4)
    return scores

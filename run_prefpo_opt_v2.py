#!/opt/homebrew/bin/python3.11
"""
PrefPO-style optimization using Ollama for ALL roles (task, discriminator, optimizer).

Phase 1 (--prefetch): Retrieve contexts for 20 training questions.
               Saves prefpo_contexts.json. Requires Qdrant (stop Streamlit first).

Phase 2 (default): Run full optimization loop using cached contexts + Ollama only.
               Saves optimized_prompt.txt.

Rubric from run_prefpo_optimization.py:
  1. Must be highly factual, answering the specific question.
  2. Should securely cite sources in an array like [1].
  3. Must be concise, under 150 words.
  4. Failure to follow format or inserting hallucinations is severely penalized.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Load env vars
with open(ROOT / ".env") as _f:
    for _line in _f:
        _line = _line.strip()
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ[_k.strip()] = _v.strip()

import aiohttp

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:latest"

CONTEXTS_FILE = ROOT / "prefpo_contexts.json"
OUTPUT_FILE = ROOT / "optimized_prompt.txt"
TRAIN_SIZE = 20
ITERATIONS = 5
CONCURRENCY = 2          # keep Ollama from being overloaded
MAX_PROMPT_CHARS = 900   # hard cap — stop the optimizer from bloating

RUBRIC = [
    "Must be highly factual, answering the specific question.",
    "Should securely cite sources in an array like [1].",
    "Must be concise, under 150 words.",
    "Failure to follow format or inserting hallucinations is severely penalized.",
]

# ---------------------------------------------------------------------------
# Load RAG-climate config
# ---------------------------------------------------------------------------

import importlib.util as _ilu

def _load_rag_config():
    spec = _ilu.spec_from_file_location("rag_config", ROOT / "config.py")
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_rag_cfg = _load_rag_config()
SYSTEM_PROMPT: str = _rag_cfg.SYSTEM_PROMPT
QDRANT_PATH: str = _rag_cfg.QDRANT_PATH
QDRANT_COLLECTION: str = _rag_cfg.QDRANT_COLLECTION


# ---------------------------------------------------------------------------
# Phase 1: Pre-fetch contexts
# ---------------------------------------------------------------------------

def prefetch_contexts() -> None:
    import pandas as pd
    from retrieve_v2 import HybridRetrieverV2
    from rerank_v2 import TwoStageCalibratedReranker

    print(f"Loading retriever from {QDRANT_PATH} …")
    retriever = HybridRetrieverV2(
        qdrant_path=QDRANT_PATH,
        qdrant_collection=QDRANT_COLLECTION,
    )
    reranker = TwoStageCalibratedReranker()

    import pandas as pd
    df = pd.read_csv(ROOT / "eval/golden.csv")
    questions = df["Question"].tolist()[:TRAIN_SIZE]
    anchors = df["anchor_text"].tolist()[:TRAIN_SIZE]

    contexts: list[dict] = []
    for i, (q, anchor) in enumerate(zip(questions, anchors), 1):
        print(f"  [{i}/{TRAIN_SIZE}] {q[:70]}")
        candidates = retriever.search(q, top_k=120)
        reranked = reranker.rerank(q, candidates)
        chunks = [r.get("document", "") for r in reranked[:5]]
        ctx_text = "\n\n".join(
            f"[{j+1}] {chunk[:600]}" for j, chunk in enumerate(chunks) if chunk
        )
        contexts.append({
            "index": i - 1,
            "question": q,
            "anchor_text": anchor or "",
            "context": ctx_text,
        })

    with open(CONTEXTS_FILE, "w") as f:
        json.dump(contexts, f, indent=2)
    print(f"\nSaved {len(contexts)} contexts → {CONTEXTS_FILE}")


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

async def _ollama(messages: list[dict], semaphore: asyncio.Semaphore,
                  max_tokens: int = 512) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": max_tokens},
    }
    async with semaphore:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                OLLAMA_URL, json=payload,
                timeout=aiohttp.ClientTimeout(total=200)
            ) as resp:
                data = await resp.json()
                return data.get("message", {}).get("content", "").strip()


# ---------------------------------------------------------------------------
# Core optimization functions
# ---------------------------------------------------------------------------

def _fill_prompt(prompt_text: str, context: str, question: str) -> str:
    try:
        return prompt_text.format(context=context, query=question)
    except KeyError:
        return f"{prompt_text}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def _score_response(response: str, anchor: str) -> float:
    """Score one response against the rubric."""
    score = 0.0
    words = response.split()

    if re.search(r'\[\d+\]', response):
        score += 0.35   # cites sources
    if len(words) <= 150:
        score += 0.25   # concise
    if len(words) > 5:
        score += 0.10   # non-trivial

    if anchor.strip():
        anchor_words = set(anchor.lower().split())
        resp_words = set(response.lower().split())
        overlap = len(anchor_words & resp_words) / max(len(anchor_words), 1)
        score += 0.30 * min(overlap * 3, 1.0)

    return min(score, 1.0)


async def _grade_prompt(
    prompt_text: str,
    contexts: list[dict],
    semaphore: asyncio.Semaphore,
) -> tuple[float, list[str]]:
    """Run all training samples through Ollama and return avg score + responses."""
    tasks = [
        _ollama(
            [{"role": "user", "content": _fill_prompt(prompt_text, c["context"], c["question"])}],
            semaphore,
        )
        for c in contexts
    ]
    responses = await asyncio.gather(*tasks)
    scores = [
        _score_response(r, c["anchor_text"])
        for r, c in zip(responses, contexts)
    ]
    return round(sum(scores) / len(scores), 4), list(responses)


async def _generate_variant(prompt_text: str, semaphore: asyncio.Semaphore) -> str:
    """Ask Ollama to generate an alternative version of the prompt."""
    rubric_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(RUBRIC))
    user_msg = f"""You are an expert prompt engineer. Given a prompt and its requirements, write an alternative version.

Original prompt:
{prompt_text}

Requirements the prompt must satisfy:
{rubric_text}

Write an alternative version that describes the same task but uses different wording. Keep all placeholders like {{context}} and {{query}} exactly as-is. Return ONLY the alternative prompt text, nothing else."""

    return await _ollama([{"role": "user", "content": user_msg}], semaphore, max_tokens=1024)


async def _discriminate(
    prompt_a: str, responses_a: list[str],
    prompt_b: str, responses_b: list[str],
    contexts: list[dict],
    semaphore: asyncio.Semaphore,
) -> tuple[int, str]:
    """
    Use Ollama to compare two sets of responses and pick a winner.
    Returns (winner=1 or 2, feedback).
    """
    rubric_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(RUBRIC))

    # Show a sample of 3 comparisons to keep context short
    sample_indices = list(range(min(3, len(contexts))))
    comparisons = ""
    for i in sample_indices:
        q = contexts[i]["question"]
        ra = responses_a[i][:200]
        rb = responses_b[i][:200]
        comparisons += f"\nQuestion: {q}\nResponse A: {ra}\nResponse B: {rb}\n---"

    user_msg = f"""You are evaluating two AI prompts based on the quality of their outputs.

Evaluation criteria:
{rubric_text}

Sample outputs (first 200 chars each):
{comparisons}

Based on these sample outputs, which prompt (A or B) produces better answers?
Reply with JSON: {{"winner": 1, "feedback": "reason"}} for prompt A, or {{"winner": 2, "feedback": "reason"}} for prompt B."""

    resp = await _ollama([{"role": "user", "content": user_msg}], semaphore, max_tokens=256)

    # Parse JSON
    m = re.search(r'\{"winner":\s*([12])[^}]*"feedback":\s*"([^"]*)"', resp)
    if m:
        return int(m.group(1)), m.group(2)
    # Fallback: look for winner number
    m2 = re.search(r'\b([12])\b', resp)
    winner = int(m2.group(1)) if m2 else 1
    return winner, resp[:100]


async def _optimize_prompt(
    losing_prompt: str,
    winner_responses: list[str],
    loser_responses: list[str],
    contexts: list[dict],
    feedback: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """Ask Ollama to rewrite the losing prompt to be more like the winning one."""
    rubric_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(RUBRIC))

    # Show one contrast example
    q = contexts[0]["question"]
    good = winner_responses[0][:250]
    bad = loser_responses[0][:250]

    user_msg = f"""You are an expert prompt engineer. Improve a prompt based on feedback.

Current (underperforming) prompt:
{losing_prompt[:600]}

Requirements:
{rubric_text}

Discriminator feedback: {feedback}

Example of GOOD output (from better prompt):
{good}

Example of WEAKER output (from current prompt):
{bad}

Rewrite the prompt to produce better outputs. IMPORTANT: Keep it under 850 characters total. Keep {{context}} and {{query}} placeholders exactly as-is.
Return ONLY the improved prompt text, nothing else."""

    result = await _ollama([{"role": "user", "content": user_msg}], semaphore, max_tokens=512)
    # Hard cap — if optimizer ignores the instruction, trim to last complete sentence
    if len(result) > MAX_PROMPT_CHARS:
        trimmed = result[:MAX_PROMPT_CHARS]
        last_period = max(trimmed.rfind("."), trimmed.rfind("\n"))
        result = trimmed[:last_period + 1] if last_period > 400 else trimmed
    return result


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------

async def run_optimization(contexts: list[dict]) -> None:
    semaphore = asyncio.Semaphore(CONCURRENCY)

    print("\n=== Starting PrefPO-style optimization (all-Ollama) ===")
    print(f"Model      : {OLLAMA_MODEL}")
    print(f"Iterations : {ITERATIONS}")
    print(f"Samples    : {len(contexts)}")
    print(f"Concurrency: {CONCURRENCY}")
    print()

    history: list[dict] = []

    # Seed prompts
    prompt_a = SYSTEM_PROMPT
    print("Generating initial variant (prompt B)…")
    prompt_b = await _generate_variant(SYSTEM_PROMPT, semaphore)
    print(f"  Prompt B (initial variant, {len(prompt_b)} chars):")
    print(f"  {prompt_b[:120]}…\n")

    best_prompt = prompt_a
    best_score = 0.0
    score_a: float | None = None
    responses_a: list[str] = []

    for iteration in range(1, ITERATIONS + 1):
        print(f"\n{'='*60}")
        print(f"  Iteration {iteration}/{ITERATIONS}")
        print(f"{'='*60}")

        # Score A
        if score_a is None:
            print(f"  Grading prompt A ({len(contexts)} samples)…")
            score_a, responses_a = await _grade_prompt(prompt_a, contexts, semaphore)
        print(f"\n  PROMPT A (score={score_a:.4f}) [{len(prompt_a)} chars]:")
        print("  " + "-"*56)
        print(prompt_a)
        print("  " + "-"*56)

        # Score B
        print(f"\n  Grading prompt B ({len(contexts)} samples)…")
        score_b, responses_b = await _grade_prompt(prompt_b, contexts, semaphore)
        print(f"\n  PROMPT B (score={score_b:.4f}) [{len(prompt_b)} chars]:")
        print("  " + "-"*56)
        print(prompt_b)
        print("  " + "-"*56)

        # Discriminate
        print("\n  Discriminating…")
        winner, feedback = await _discriminate(
            prompt_a, responses_a,
            prompt_b, responses_b,
            contexts, semaphore,
        )
        print(f"  → Winner = Prompt {winner}")
        print(f"  → Feedback: {feedback}")

        if winner == 1:
            winning_prompt, winning_score, winning_responses = prompt_a, score_a, responses_a
            losing_prompt, losing_responses = prompt_b, responses_b
            losing_score = score_b
        else:
            winning_prompt, winning_score, winning_responses = prompt_b, score_b, responses_b
            losing_prompt, losing_responses = prompt_a, responses_a
            losing_score = score_a

        if winning_score > best_score:
            best_score = winning_score
            best_prompt = winning_prompt

        # Optimize the loser
        print("\n  Optimizing losing prompt…")
        new_prompt = await _optimize_prompt(
            losing_prompt, winning_responses, losing_responses,
            contexts, feedback, semaphore,
        )
        print(f"\n  New optimized prompt ({len(new_prompt)} chars):")
        print("  " + "-"*56)
        print(new_prompt)
        print("  " + "-"*56)

        history.append({
            "iteration": iteration,
            "prompt_a": prompt_a,
            "score_a": score_a,
            "prompt_b": prompt_b,
            "score_b": score_b,
            "winner": winner,
            "feedback": feedback,
            "winning_score": winning_score,
            "losing_score": losing_score,
            "new_prompt": new_prompt,
            "best_score_so_far": max(winning_score, best_score),
        })

        # Save history after every iteration
        history_path = ROOT / "prefpo_optimization_history.json"
        with open(history_path, "w") as f:
            json.dump({
                "seed_prompt": SYSTEM_PROMPT,
                "initial_variant": prompt_b if iteration == 1 else history[0]["prompt_b"],
                "rubric": RUBRIC,
                "model": OLLAMA_MODEL,
                "iterations": history,
                "best_score": best_score,
                "best_prompt": best_prompt,
            }, f, indent=2)

        # Next iteration: winner stays, new_prompt replaces loser
        prompt_a = winning_prompt
        score_a = winning_score
        responses_a = winning_responses
        prompt_b = new_prompt

        print(f"\n  Best so far: {best_score:.4f}")

    # Final score of prompt_b
    print(f"\nFinal grading of last optimized prompt…")
    final_score_b, _ = await _grade_prompt(prompt_b, contexts, semaphore)
    print(f"  Final prompt B score: {final_score_b:.4f}")
    if final_score_b > best_score:
        best_score = final_score_b
        best_prompt = prompt_b

    print(f"\n{'='*60}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Best score : {best_score:.4f}")
    print(f"  Best prompt:\n{best_prompt}")

    with open(OUTPUT_FILE, "w") as f:
        f.write(best_prompt)
    print(f"\nSaved best prompt → {OUTPUT_FILE}")
    print(f"Full history     → {ROOT / 'prefpo_optimization_history.json'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if "--prefetch" in sys.argv:
        prefetch_contexts()
        return

    if not CONTEXTS_FILE.exists():
        print(f"ERROR: {CONTEXTS_FILE} not found.")
        print("Run: /opt/homebrew/bin/python3.11 run_prefpo_opt_v2.py --prefetch")
        sys.exit(1)

    with open(CONTEXTS_FILE) as f:
        contexts = json.load(f)
    print(f"Loaded {len(contexts)} cached contexts.")

    asyncio.run(run_optimization(contexts))


if __name__ == "__main__":
    main()

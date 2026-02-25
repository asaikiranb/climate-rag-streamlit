"""
Comprehensive RAG evaluation with Contextual AI reranking.

Metrics computed:
  - Retrieval: recall@k, MRR@k, NDCG@k for k=1,3,5
  - Generation: faithfulness, relevance, completeness (LMUnit-judged via HF Inference API)
  - Citation: validity, coverage, source grounding
  - Latency: embed_ms, search_ms, rerank_ms, generate_ms (mean/p50/p95/min/max)
  - Breakdown by difficulty (Easy/Medium/Hard)

Usage:
    python run_contextual_eval.py                        # full run with reranker
    python run_contextual_eval.py --no-reranker          # skip reranker
    python run_contextual_eval.py --retrieval-only       # skip LLM, just test retrieval
    python run_contextual_eval.py --output results.json  # custom output path
"""

import os
import sys

# When HF_TOKEN is not set, force HuggingFace libs to use the local cache only.
# Without this, AutoTokenizer / AutoConfig calls to HF Hub can hang indefinitely
# on unauthenticated rate limits (silent infinite stall, not a fast failure).
# CI always sets HF_TOKEN, so it stays in online mode and can download models fresh.
if not os.environ.get("HF_TOKEN"):
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import argparse
import time
import requests
import numpy as np
from datetime import datetime
from typing import List, Dict

from retrieve import HybridRetriever
from rerank import load_reranker, load_reranker_api
from llm import build_context
from config import SYSTEM_PROMPT, SYSTEM_MESSAGE, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TOP_P
from eval.metrics import citation_validity, citation_coverage, source_grounding
from eval.retrieval_metrics import compute_retrieval_metrics_at_k

# Judge routing: LMUnit (paid, premium) vs Ollama (free, local)
_USE_OLLAMA_JUDGE = not os.environ.get("CONTEXTUAL_API_KEY")
if _USE_OLLAMA_JUDGE:
    from eval.generation_metrics_ollama import judge_generation, OLLAMA_JUDGE_MODEL
    JUDGE_MODEL_LABEL = f"ollama/{OLLAMA_JUDGE_MODEL}"
else:
    from eval.generation_metrics import judge_generation, LMUNIT_MODEL
    JUDGE_MODEL_LABEL = LMUNIT_MODEL


K_VALUES = [1, 3, 5]

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:latest"
GROQ_MODEL = "llama-3.3-70b-versatile"      # fallback when CONTEXTUAL_API_KEY not set
CONTEXTUAL_GENERATE_MODEL = "v1"             # Contextual AI GLM model version

# Max context chars passed to Groq/Ollama fallback generators
CONTEXT_CHAR_LIMIT = 6000


def _truncate_context(context: str) -> str:
    """Truncate context to keep Groq token usage manageable."""
    return context[:CONTEXT_CHAR_LIMIT]


def generate_answer_contextual(query: str, search_results: list) -> str:
    """Generate answer using Contextual AI GLM (grounded, no TPD limits).

    Passes retrieved chunks as `knowledge` — the GLM is engineered to stay faithful
    to the provided documents and minimise hallucinations.
    """
    from contextual import ContextualAI
    client = ContextualAI(api_key=os.environ["CONTEXTUAL_API_KEY"])
    knowledge = [r["document"] for r in search_results]
    response = client.generate.create(
        model=CONTEXTUAL_GENERATE_MODEL,
        messages=[{"role": "user", "content": query}],
        knowledge=knowledge,
        system_prompt=SYSTEM_MESSAGE,
        avoid_commentary=True,
        max_new_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
        top_p=LLM_TOP_P,
    )
    return response.response.strip()


def generate_answer_ollama(query: str, context: str) -> str:
    """Generate answer using local Ollama (used when no GROQ_API_KEY is set)."""
    prompt = SYSTEM_PROMPT.format(context=_truncate_context(context), query=query)
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "system": SYSTEM_MESSAGE,
            "stream": False,
            "options": {
                "temperature": LLM_TEMPERATURE,
                "num_predict": LLM_MAX_TOKENS,
                "top_p": LLM_TOP_P,
            },
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def generate_answer_groq(query: str, context: str) -> str:
    """Generate answer via Groq API (used in CI where Ollama on CPU is too slow)."""
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    prompt = SYSTEM_PROMPT.format(context=_truncate_context(context), query=query)
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        top_p=LLM_TOP_P,
    )
    return completion.choices[0].message.content.strip()


def generate_answer(query: str, context: str, search_results: list = None) -> str:
    """Route: Contextual GLM (premium) → Ollama (CI / local dev).
    
    GROQ_API_KEY is reserved for query expansion only.
    """
    if os.environ.get("CONTEXTUAL_API_KEY"):
        return generate_answer_contextual(query, search_results or [])
    return generate_answer_ollama(query, context)


def load_eval_dataset(path: str) -> List[Dict]:
    """Load the JSONL eval dataset."""
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def percentile_stats(values: List[float]) -> Dict:
    """Compute mean, p50, p95, min, max for a list of values."""
    if not values:
        return {"mean": 0, "p50": 0, "p95": 0, "min": 0, "max": 0}
    arr = np.array(values)
    return {
        "mean": round(float(np.mean(arr)), 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2),
    }


def _write_results(
    output_path, results_log, all_retrieval_metrics, all_generation_metrics,
    citation_scores, latency_embed, latency_search, latency_rerank,
    latency_generate, difficulty_buckets, top_k, retrieval_only,
    use_reranker, total, partial=False,
):
    """Write current results to disk. Called after every question (partial=True) and at the end."""
    if not all_retrieval_metrics:
        return

    agg_retrieval = {}
    for k in K_VALUES:
        for metric in ["recall", "mrr", "ndcg"]:
            key = f"{metric}@{k}"
            vals = [m[key] for m in all_retrieval_metrics]
            agg_retrieval[key] = round(float(np.mean(vals)), 4)

    agg_generation = {}
    if all_generation_metrics:
        for key in ["faithfulness", "relevance", "completeness", "overall"]:
            vals = [m[key] for m in all_generation_metrics]
            agg_generation[key] = round(float(np.mean(vals)), 4)

    agg_citation = {}
    if citation_scores["validity"]:
        for key in ["validity", "coverage", "grounding"]:
            agg_citation[f"citation_{key}"] = round(float(np.mean(citation_scores[key])), 4)

    latency_summary = {}
    for name, values in [
        ("embed_ms", latency_embed), ("search_ms", latency_search),
        ("rerank_ms", latency_rerank), ("generate_ms", latency_generate),
    ]:
        latency_summary[name] = percentile_stats(values)

    difficulty_summary = {}
    for diff, data in sorted(difficulty_buckets.items()):
        n = data["count"]
        r1 = round(float(np.mean([m["recall@1"] for m in data["retrieval"]])), 4)
        r5 = round(float(np.mean([m["recall@5"] for m in data["retrieval"]])), 4)
        mrr5 = round(float(np.mean([m["mrr@5"] for m in data["retrieval"]])), 4)
        ndcg5 = round(float(np.mean([m["ndcg@5"] for m in data["retrieval"]])), 4)
        faith = round(float(np.mean([m["faithfulness"] for m in data["generation"]])), 4) if data["generation"] else 0.0
        difficulty_summary[diff] = {"count": n, "recall@1": r1, "recall@5": r5, "mrr@5": mrr5, "ndcg@5": ndcg5, "faithfulness": faith}

    output = {
        "timestamp": datetime.now().isoformat(),
        "partial": partial,
        "questions_completed": len(results_log),
        "config": {
            "top_k": top_k,
            "retrieval_only": retrieval_only,
            "reranker": "ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b" if use_reranker else "none",
            "answer_model": (
                f"contextual/glm-{CONTEXTUAL_GENERATE_MODEL} (grounded)"
                if os.environ.get("CONTEXTUAL_API_KEY")
                else f"groq/{GROQ_MODEL} (ctx:{CONTEXT_CHAR_LIMIT}chars)"
                if os.environ.get("GROQ_API_KEY")
                else f"ollama/{OLLAMA_MODEL}"
            ),
            "judge_model": JUDGE_MODEL_LABEL,
            "total_questions": total,
        },
        "retrieval_metrics": agg_retrieval,
        "generation_metrics": agg_generation,
        "citation_metrics": agg_citation,
        "latency_summary": latency_summary,
        "difficulty_breakdown": difficulty_summary,
        "per_question": results_log,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def run_eval(
    dataset_path: str,
    output_path: str,
    top_k: int = 5,
    retrieval_only: bool = False,
    use_reranker: bool = True,
):
    dataset = load_eval_dataset(dataset_path)
    total = len(dataset)
    print(f"\nLoaded {total} questions from {os.path.basename(dataset_path)}")

    print("Loading retriever...")
    retriever = HybridRetriever()

    # Load Contextual AI reranker.
    # When HF_TOKEN is available (e.g. in CI), use the HF Inference API backend so
    # no model weights are downloaded locally — avoids OOM and timeout on CPU runners.
    # Locally (no HF_TOKEN), load the 1B weights directly.
    reranker = None
    if use_reranker:
        if os.environ.get("HF_TOKEN"):
            print("Loading Contextual AI reranker via HF Inference API...")
            reranker = load_reranker_api()
            print("Reranker ready (ContextualAI/ctxl-rerank-v2 — HF Inference API)")
        else:
            print("Loading Contextual AI reranker (local weights)...")
            reranker = load_reranker()
            print("Reranker ready (ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b — local)")
    else:
        print("Reranker disabled")

    # LLM + judge
    if not retrieval_only:
        if os.environ.get("CONTEXTUAL_API_KEY"):
            print(f"Using Contextual AI GLM (model={CONTEXTUAL_GENERATE_MODEL}) for answer generation")
        elif os.environ.get("GROQ_API_KEY"):
            print(f"Using Groq API ({GROQ_MODEL}) for answer generation")
        else:
            print(f"Using local Ollama ({OLLAMA_MODEL}) for answer generation")
        print(f"Using {JUDGE_MODEL_LABEL} for generation scoring")

    # Accumulators
    results_log = []
    all_retrieval_metrics = []
    all_generation_metrics = []
    citation_scores = {"validity": [], "coverage": [], "grounding": []}
    latency_embed, latency_search, latency_rerank, latency_generate = [], [], [], []
    difficulty_buckets = {}

    for idx, entry in enumerate(dataset):
        question = entry["question"]
        gold_sources = entry.get("gold_sources", [])
        difficulty = entry.get("metadata", {}).get("difficulty", "unknown")
        q_short = question[:80] + "..." if len(question) > 80 else question
        print(f"\n[{idx + 1}/{total}] ({difficulty}) {q_short}")

        # ---------- Retrieval with timing ----------
        retrieval_data = retriever.hybrid_search_timed(
            query=question, top_k=top_k, reranker=reranker
        )
        search_results = retrieval_data["results"]
        timings = retrieval_data["timings"]

        latency_embed.append(timings["embed_ms"])
        latency_search.append(timings["search_ms"])
        latency_rerank.append(timings["rerank_ms"])

        retrieved_filenames = [r["metadata"]["filename"] for r in search_results]

        ret_metrics = compute_retrieval_metrics_at_k(
            retrieved_filenames, gold_sources, k_values=K_VALUES
        )
        all_retrieval_metrics.append(ret_metrics)

        hit = any(fn in gold_sources for fn in retrieved_filenames)
        hit_marker = "✓" if hit else "✗"
        print(f"  Retrieval: {hit_marker} | recall@1={ret_metrics['recall@1']:.2f} recall@5={ret_metrics['recall@5']:.2f} | mrr@5={ret_metrics['mrr@5']:.2f} | ndcg@5={ret_metrics['ndcg@5']:.2f}")

        result_entry = {
            "question": question,
            "gold_sources": gold_sources,
            "difficulty": difficulty,
            "retrieval_metrics": ret_metrics,
            "retrieved_sources": retrieved_filenames,
            "timings": timings,
        }

        # ---------- Generate answer + metrics ----------
        gen_metrics = None
        if not retrieval_only and search_results:
            try:
                context = build_context(search_results)

                # Generate answer (Contextual GLM in CI, Groq fallback, Ollama local)
                gen_start = time.perf_counter()
                answer = generate_answer(question, context, search_results=search_results)
                gen_ms = (time.perf_counter() - gen_start) * 1000
                latency_generate.append(round(gen_ms, 2))

                # Citation metrics
                cv = citation_validity(answer, len(search_results))
                cc = citation_coverage(answer)
                sg = source_grounding(answer, search_results)

                citation_scores["validity"].append(cv["score"])
                citation_scores["coverage"].append(cc["score"])
                citation_scores["grounding"].append(sg["score"])

                # LLM-judged generation quality (via LMUnit / HF Inference API)
                gen_metrics = judge_generation(
                    question=question, context=context, answer=answer,
                )
                all_generation_metrics.append(gen_metrics)

                print(f"  Generation: faith={gen_metrics['faithfulness']:.2f} | relev={gen_metrics['relevance']:.2f} | compl={gen_metrics['completeness']:.2f}")
                print(f"  Citations:  valid={cv['score']:.2f} | cover={cc['score']:.2f} | ground={sg['score']:.2f}")

                result_entry["answer"] = answer
                result_entry["citation_validity"] = cv["score"]
                result_entry["citation_coverage"] = cc["score"]
                result_entry["source_grounding"] = sg["score"]
                result_entry["generation_metrics"] = gen_metrics
                result_entry["generate_ms"] = round(gen_ms, 2)

            except Exception as e:
                print(f"  LLM error: {e}")

        # Track by difficulty
        if difficulty not in difficulty_buckets:
            difficulty_buckets[difficulty] = {
                "retrieval": [], "generation": [], "count": 0,
            }
        difficulty_buckets[difficulty]["retrieval"].append(ret_metrics)
        if gen_metrics:
            difficulty_buckets[difficulty]["generation"].append(gen_metrics)
        difficulty_buckets[difficulty]["count"] += 1

        results_log.append(result_entry)

        # Checkpoint: flush partial results after every question so a timeout
        # doesn't lose everything. The final save at the end overwrites this.
        _write_results(
            output_path, results_log, all_retrieval_metrics, all_generation_metrics,
            citation_scores, latency_embed, latency_search, latency_rerank,
            latency_generate, difficulty_buckets, top_k, retrieval_only,
            use_reranker, total, partial=True,
        )

    # ==================== Final save + print ====================
    _write_results(
        output_path, results_log, all_retrieval_metrics, all_generation_metrics,
        citation_scores, latency_embed, latency_search, latency_rerank,
        latency_generate, difficulty_buckets, top_k, retrieval_only,
        use_reranker, total, partial=False,
    )

    # Read back the saved file for console summary
    with open(output_path) as f:
        saved = json.load(f)

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"\n  Questions evaluated: {saved['questions_completed']}/{total}")
    print(f"  Top-K: {top_k}")
    print(f"  Reranker: {'ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b' if use_reranker else 'disabled'}")

    print("\n  RETRIEVAL METRICS:")
    for k in K_VALUES:
        r = saved["retrieval_metrics"]
        print(f"    k={k}:  recall={r[f'recall@{k}']:.4f}  mrr={r[f'mrr@{k}']:.4f}  ndcg={r[f'ndcg@{k}']:.4f}")

    if saved["generation_metrics"]:
        print("\n  GENERATION METRICS:")
        for key, val in saved["generation_metrics"].items():
            print(f"    {key:20s}{val:.4f}")
    else:
        print("\n  GENERATION METRICS: (skipped)")

    if saved["citation_metrics"]:
        print("\n  CITATION METRICS:")
        for key, val in saved["citation_metrics"].items():
            print(f"    {key}: {val:.4f}")
    else:
        print("\n  CITATION METRICS: (skipped)")

    print("\n  LATENCY SUMMARY:")
    for name, stats in saved["latency_summary"].items():
        print(f"    {name:20s}mean={stats['mean']:8.1f}ms  p50={stats['p50']:8.1f}ms  p95={stats['p95']:8.1f}ms  min={stats['min']:8.1f}ms  max={stats['max']:8.1f}ms")

    print("\n  BY DIFFICULTY:")
    for diff, data in sorted(saved["difficulty_breakdown"].items()):
        print(f"    {diff:10s} n={data['count']:3d}  recall@5={data['recall@5']:.4f}  mrr@5={data['mrr@5']:.4f}  ndcg@5={data['ndcg@5']:.4f}  faith={data['faithfulness']:.4f}")

    print("\n" + "=" * 70)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comprehensive RAG evaluation")
    parser.add_argument("--dataset", default="contextual_eval_dataset.jsonl")
    parser.add_argument("--output", default="eval_results_comprehensive.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--no-reranker", action="store_true", help="Disable Contextual AI reranking")
    args = parser.parse_args()

    print("=" * 70)
    print("  RAG Contextual Evaluation — Comprehensive Metrics")
    if not args.no_reranker:
        print("  (with Contextual AI reranking)")
    print("=" * 70)

    run_eval(
        dataset_path=args.dataset,
        output_path=args.output,
        top_k=args.top_k,
        retrieval_only=args.retrieval_only,
        use_reranker=not args.no_reranker,
    )

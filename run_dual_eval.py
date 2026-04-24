#!/opt/homebrew/bin/python3.11
"""
Dual evaluation: Standard RAG vs PrefPO RAG on all 126 golden questions.
Uses local Ollama (llama3.2:latest) for generation.
Computes retrieval metrics (Recall@k, MRR) and citation quality metrics.
Outputs evaluation_results.json consumed by the Streamlit dashboard.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

from config import (
    INGEST_EMBEDDING_MODEL,
    QDRANT_COLLECTION,
    QDRANT_PATH,
    SPARSE_MODE,
    SYSTEM_PROMPT,
)
from eval.loader import load_golden_csv
from eval.generation_metrics_ollama import judge_generation
from eval.metrics import (
    citation_coverage,
    citation_validity,
    compute_doc_retrieval_scores,
    compute_page_retrieval_scores,
    source_grounding,
)
from eval.normalize import normalize_retrievals
from retrieve_v2 import HybridRetrieverV2
from rerank_v2 import TwoStageCalibratedReranker

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:latest"
OLLAMA_TIMEOUT = 180

_opt_file = Path("optimized_prompt.txt")
PREFPO_OPTIMIZED_PROMPT = _opt_file.read_text().strip() if _opt_file.exists() else SYSTEM_PROMPT

OUT_PATH = Path("evaluation_results.json")
PARTIAL_PATH = Path("evaluation_results.partial.json")
# Write checkpoint every N questions (1 = maximum crash safety).
CHECKPOINT_EVERY = int(os.environ.get("EVAL_CHECKPOINT_EVERY", "1"))


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _load_checkpoint(ollama_model: str) -> list:
    if not PARTIAL_PATH.exists():
        return []
    try:
        data = json.loads(PARTIAL_PATH.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"  Warning: could not read {PARTIAL_PATH}: {e}", file=sys.stderr)
        return []
    if data.get("ollama_model") != ollama_model:
        print("  Partial checkpoint: Ollama model changed — starting from scratch")
        return []
    samples = data.get("samples")
    if not isinstance(samples, list):
        return []
    print(f"  Resuming: {len(samples)} questions already in {PARTIAL_PATH.name}")
    return samples


def _save_checkpoint(ollama_model: str, samples: list) -> None:
    payload = {"ollama_model": ollama_model, "samples": samples}
    _atomic_write(PARTIAL_PATH, json.dumps(payload, indent=2))


def _find_qdrant() -> tuple[str, str]:
    def exists(path: str, col: str) -> bool:
        return (Path(path) / "collection" / col / "storage.sqlite").exists()

    for path, col in [
        (QDRANT_PATH, QDRANT_COLLECTION),
        ("./qdrant_db_ci", "hvac_documents_qdrant_ci"),
        ("./qdrant_db", "hvac_documents_qdrant"),
    ]:
        if exists(path, col):
            return path, col
    return QDRANT_PATH, QDRANT_COLLECTION


def _build_context(hits: list) -> str:
    parts = []
    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        parts.append(
            f"[Source {i}] (Document: {meta['filename']}, Page: {meta['page_number']})\n"
            f"{hit['document']}\n"
        )
    return "\n---\n".join(parts)


def _ollama_generate(prompt: str) -> str:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 384},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        return resp.json().get("response", "").strip()
    except Exception as exc:
        print(f"  Ollama error: {exc}", file=sys.stderr)
        return ""


def _generate(query: str, hits: list, system_prompt: str) -> tuple[str, float]:
    context = _build_context(hits[:5])
    prompt = system_prompt.format(context=context, query=query)
    t0 = time.perf_counter()
    answer = _ollama_generate(prompt)
    ms = (time.perf_counter() - t0) * 1000.0
    return answer, ms


def _gen_metrics(question: str, answer: str, hits: list) -> dict:
    top5 = hits[:5]
    cv = citation_validity(answer, len(top5))
    cc = citation_coverage(answer)
    sg = source_grounding(answer, top5)
    context = _build_context(top5)
    gen = judge_generation(question=question, context=context, answer=answer)
    return {
        "citation_validity": round(cv["score"], 3),
        "citation_coverage": round(cc["score"], 3),
        "source_grounding": round(sg["score"], 3),
        "faithfulness": round(gen.get("faithfulness", 0.0), 3),
        "relevance": round(gen.get("relevance", 0.0), 3),
        "completeness": round(gen.get("completeness", 0.0), 3),
        "overall": round(gen.get("overall", 0.0), 3),
    }


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def main() -> None:
    print("Loading retriever and reranker…")
    qdrant_path, qdrant_col = _find_qdrant()
    sparse = SPARSE_MODE if SPARSE_MODE in {"none", "bm42", "splade"} else "none"
    retriever = HybridRetrieverV2(
        backend="qdrant",
        embedding_model=INGEST_EMBEDDING_MODEL,
        sparse_mode=sparse,
        qdrant_path=qdrant_path,
        qdrant_collection=qdrant_col,
    )
    reranker = TwoStageCalibratedReranker()

    print("Loading 126 golden questions…")
    rows = load_golden_csv("eval/golden.csv")
    print(f"Loaded {len(rows)} rows.\n")

    print(f"Using Ollama model: {OLLAMA_MODEL}")
    print("Each question runs retrieval once, then generation twice (Standard + PrefPO).\n")

    results = _load_checkpoint(OLLAMA_MODEL)
    start = len(results)
    if start and start < len(rows):
        print(f"  Continuing at question index {start + 1} / {len(rows)}")

    tail = list(enumerate(rows[start:], start))
    for _idx, row in tqdm(
        tail,
        desc="Evaluating",
        unit="q",
        initial=start,
        total=len(rows),
    ):
        # ── Retrieval (shared) ─────────────────────────────────────────────
        t_ret = time.perf_counter()
        candidates = retriever.search(row.question, top_k=120)
        reranked = reranker.rerank(row.question, candidates)
        retrieval_ms = (time.perf_counter() - t_ret) * 1000.0
        top5 = reranked[:5]

        # Normalize for metric functions
        normalized = normalize_retrievals(reranked, 10)

        doc_scores = compute_doc_retrieval_scores(normalized, row.gold_sources)
        page_scores = compute_page_retrieval_scores(
            retrievals=normalized,
            gold_source=row.gold_sources,
            gold_pages=row.gold_pages,
            anchor_text=row.anchor_text,
            anchor_threshold=80,
        )

        # ── Standard generation ────────────────────────────────────────────
        std_answer, std_gen_ms = _generate(row.question, top5, SYSTEM_PROMPT)
        std_metrics = _gen_metrics(row.question, std_answer, top5)

        # ── PrefPO generation ──────────────────────────────────────────────
        po_answer, po_gen_ms = _generate(row.question, top5, PREFPO_OPTIMIZED_PROMPT)
        po_metrics = _gen_metrics(row.question, po_answer, top5)

        results.append({
            "query": row.question,
            "difficulty": row.difficulty,
            "gold_source": row.gold_sources,
            "doc_hit@1": doc_scores.hits.get(1) or 0,
            "doc_hit@3": doc_scores.hits.get(3) or 0,
            "doc_hit@5": doc_scores.hits.get(5) or 0,
            "doc_rr": doc_scores.rr or 0.0,
            "page_hit@5": page_scores.hits.get(5) or 0,
            "retrieval_ms": round(retrieval_ms, 1),
            "standard": {
                "answer": std_answer,
                "latency_ms": round(retrieval_ms + std_gen_ms, 1),
                "generate_ms": round(std_gen_ms, 1),
                **std_metrics,
            },
            "prefpo": {
                "answer": po_answer,
                "latency_ms": round(retrieval_ms + po_gen_ms, 1),
                "generate_ms": round(po_gen_ms, 1),
                **po_metrics,
            },
        })
        if CHECKPOINT_EVERY > 0 and len(results) % CHECKPOINT_EVERY == 0:
            _save_checkpoint(OLLAMA_MODEL, results)
        gc.collect()

    # ── Aggregate ──────────────────────────────────────────────────────────
    scored = [r for r in results if r["doc_rr"] is not None]
    n = len(scored) or 1

    retrieval_metrics = {
        "recall@1": _avg([r["doc_hit@1"] for r in scored]),
        "recall@3": _avg([r["doc_hit@3"] for r in scored]),
        "recall@5": _avg([r["doc_hit@5"] for r in scored]),
        "mrr@10":   _avg([r["doc_rr"]    for r in scored]),
        "page_recall@5": _avg([r["page_hit@5"] for r in scored]),
    }

    def gen_agg(key: str, prompt_key: str) -> float:
        return _avg([r[prompt_key][key] for r in results if r[prompt_key].get(key) is not None])

    output = {
        "total_questions": len(results),
        "model": OLLAMA_MODEL,
        "retrieval_metrics": retrieval_metrics,
        "standard_avg_latency_ms": _avg([r["standard"]["latency_ms"] for r in results]),
        "prefpo_avg_latency_ms":   _avg([r["prefpo"]["latency_ms"]   for r in results]),
        "standard": {
            "avg_generate_ms":    _avg([r["standard"]["generate_ms"]    for r in results]),
            "citation_validity":  gen_agg("citation_validity",  "standard"),
            "citation_coverage":  gen_agg("citation_coverage",  "standard"),
            "source_grounding":   gen_agg("source_grounding",   "standard"),
            "faithfulness":       gen_agg("faithfulness",       "standard"),
            "relevance":          gen_agg("relevance",          "standard"),
            "completeness":       gen_agg("completeness",       "standard"),
            "overall":            gen_agg("overall",            "standard"),
        },
        "prefpo": {
            "avg_generate_ms":   _avg([r["prefpo"]["generate_ms"]    for r in results]),
            "citation_validity": gen_agg("citation_validity", "prefpo"),
            "citation_coverage": gen_agg("citation_coverage", "prefpo"),
            "source_grounding":  gen_agg("source_grounding",  "prefpo"),
            "faithfulness":      gen_agg("faithfulness",      "prefpo"),
            "relevance":         gen_agg("relevance",         "prefpo"),
            "completeness":      gen_agg("completeness",      "prefpo"),
            "overall":           gen_agg("overall",           "prefpo"),
        },
        "samples": results,
    }

    out_path = OUT_PATH
    out_path.write_text(json.dumps(output, indent=2))
    if PARTIAL_PATH.exists():
        try:
            PARTIAL_PATH.unlink()
        except OSError:
            pass

    print("\n── Results ─────────────────────────────────────")
    print(f"  Recall@1 : {retrieval_metrics['recall@1']:.3f}")
    print(f"  Recall@3 : {retrieval_metrics['recall@3']:.3f}")
    print(f"  Recall@5 : {retrieval_metrics['recall@5']:.3f}")
    print(f"  MRR@10   : {retrieval_metrics['mrr@10']:.3f}")
    print(f"  Standard  — citation_validity={output['standard']['citation_validity']:.3f}  "
          f"source_grounding={output['standard']['source_grounding']:.3f}  "
          f"faithfulness={output['standard']['faithfulness']:.3f}")
    print(f"  PrefPO    — citation_validity={output['prefpo']['citation_validity']:.3f}  "
          f"source_grounding={output['prefpo']['source_grounding']:.3f}  "
          f"faithfulness={output['prefpo']['faithfulness']:.3f}")
    print(f"\nSaved → {out_path.resolve()}")


if __name__ == "__main__":
    main()

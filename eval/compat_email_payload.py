"""Convert upgraded eval artifacts into the legacy email payload contract."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _latency_row(payload: dict[str, Any] | None) -> dict[str, float]:
    row = payload or {}
    return {
        "mean": _safe_float(row.get("mean")),
        "p50": _safe_float(row.get("p50")),
        "p95": _safe_float(row.get("p95")),
        "min": _safe_float(row.get("min")),
        "max": _safe_float(row.get("max")),
    }


def to_email_payload(
    summary_payload: dict[str, Any],
    manifest_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = manifest_payload or {}
    retrieval = (summary_payload.get("retrieval") or {}).get("doc") or {}
    by_difficulty = summary_payload.get("by_difficulty") or {}
    latency_ms = summary_payload.get("latency_ms") or {}

    difficulty_rows: dict[str, Any] = {}
    for label in ["Easy", "Medium", "Hard"]:
        row = by_difficulty.get(label) or {}
        doc = row.get("doc") or {}
        difficulty_rows[label] = {
            "count": int(doc.get("count") or row.get("count") or 0),
            "recall@1": _safe_float(doc.get("recall@1")),
            "recall@5": _safe_float(doc.get("recall@5")),
            "mrr@5": _safe_float(doc.get("mrr@10")),
            "ndcg@5": _safe_float(doc.get("ndcg@5")),
            "faithfulness": 0.0,
        }

    reranker_name = str(((manifest.get("reranker") or {}).get("name") or "cross-encoder/ms-marco-MiniLM-L-6-v2")).strip()

    payload = {
        "timestamp": manifest.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        "config": {
            "total_questions": int((summary_payload.get("retrieval") or {}).get("count") or 0),
            "reranker": reranker_name,
            "multi_query_expansion": False,
            "groq_available_for_expansion": False,
        },
        "retrieval_metrics": {
            "recall@1": _safe_float(retrieval.get("recall@1")),
            "mrr@1": _safe_float(retrieval.get("mrr@10")),
            "ndcg@1": _safe_float(retrieval.get("ndcg@1")),
            "recall@3": _safe_float(retrieval.get("recall@3")),
            "mrr@3": _safe_float(retrieval.get("mrr@10")),
            "ndcg@3": _safe_float(retrieval.get("ndcg@3")),
            "recall@5": _safe_float(retrieval.get("recall@5")),
            "mrr@5": _safe_float(retrieval.get("mrr@10")),
            "ndcg@5": _safe_float(retrieval.get("ndcg@5")),
        },
        "generation_metrics": {
            "faithfulness": 0.0,
            "relevance": 0.0,
            "completeness": 0.0,
            "overall": 0.0,
        },
        "citation_metrics": {
            "citation_validity": 0.0,
            "citation_coverage": 0.0,
            "citation_grounding": 0.0,
        },
        "latency_summary": {
            "embed_ms": _latency_row(latency_ms.get("embed") or latency_ms.get("embed_ms")),
            "search_ms": _latency_row(latency_ms.get("search") or latency_ms.get("search_ms")),
            "rerank_ms": _latency_row(latency_ms.get("rerank") or latency_ms.get("rerank_ms")),
            "generate_ms": _latency_row(latency_ms.get("generate") or latency_ms.get("generate_ms")),
        },
        "difficulty_breakdown": difficulty_rows,
    }
    return payload


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.compat_email_payload")
    parser.add_argument("--summary", required=True, help="Path to eval/run.py summary.json")
    parser.add_argument("--manifest", default=None, help="Optional path to eval_manifest.json")
    parser.add_argument("--out", required=True, help="Output JSON path in email-compatible schema")
    args = parser.parse_args(argv)

    summary_payload = _read_json(args.summary)
    manifest_payload = _read_json(args.manifest) if args.manifest else None
    email_payload = to_email_payload(summary_payload=summary_payload, manifest_payload=manifest_payload)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(email_payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

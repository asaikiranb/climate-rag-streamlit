"""Writers for per-query and summary evaluation outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.latency import QueryLatency, summarize_query_latencies
from eval.metrics import K_VALUES, aggregate_scored_metrics


def _group_rows(rows: list[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(field) or "Unknown")
        groups.setdefault(key, []).append(row)
    return groups


def _build_group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "doc": aggregate_scored_metrics(rows, "doc", K_VALUES),
        "page": aggregate_scored_metrics(rows, "page", K_VALUES),
    }


def build_summary(
    per_query_rows: list[dict[str, Any]],
    latencies: list[QueryLatency],
) -> dict[str, Any]:
    """Assemble the summary JSON payload."""

    by_difficulty = {
        key: _build_group_summary(group_rows)
        for key, group_rows in sorted(_group_rows(per_query_rows, "difficulty").items())
    }
    by_gold_sources = {
        key: _build_group_summary(group_rows)
        for key, group_rows in sorted(_group_rows(per_query_rows, "gold_sources").items())
    }

    metric_notes = {
        "doc_scoring": "Document metrics are scored only when gold_sources is present.",
        "page_scoring": (
            "Page metrics are scored only when gold_sources is present and at least one of "
            "page_range or anchor_text is present."
        ),
        "page_ndcg": (
            "Page nDCG uses graded relevance where rel=2 is an exact page hit or anchor-text "
            "fallback hit, and rel=1 is the correct document with the wrong or unknown page."
        ),
    }

    summary = {
        "retrieval": {
            "count": len(per_query_rows),
            "doc": aggregate_scored_metrics(per_query_rows, "doc", K_VALUES),
            "page": aggregate_scored_metrics(per_query_rows, "page", K_VALUES),
        },
        "metric_notes": metric_notes,
        "by_difficulty": by_difficulty,
        "by_gold_sources": by_gold_sources,
        "by_gold_source": by_gold_sources,
        "latency_ms": summarize_query_latencies(latencies),
        "index_stats": {
            "documents": "not_available",
            "chunks": "not_available",
            "vector_dim": "not_available",
        },
    }
    return summary


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Write the per-query JSONL file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    return output_path


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a JSON document."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return output_path


def print_console_summary(summary: dict[str, Any]) -> None:
    """Print a compact console summary."""

    retrieval = summary["retrieval"]
    latency = summary["latency_ms"]["total"]

    print("")
    print("Evaluation Summary")
    print(f"  Queries: {retrieval['count']}")
    print(f"  DOC scored: {retrieval['doc']['count']}")
    print(f"  PAGE scored: {retrieval['page']['count']}")

    for prefix in ("doc", "page"):
        metrics = retrieval[prefix]
        scored = metrics["count"]
        if not scored:
            print(f"  {prefix.upper()}: no scored rows")
            continue
        print(
            "  "
            f"{prefix.upper()}: Recall@1={metrics['recall@1']:.4f} "
            f"Recall@3={metrics['recall@3']:.4f} "
            f"Recall@5={metrics['recall@5']:.4f} "
            f"Recall@10={metrics['recall@10']:.4f} "
            f"MRR@10={metrics['mrr@10']:.4f}"
        )

    if latency:
        print(
            "  "
            f"Latency total ms: mean={latency['mean']:.2f} "
            f"p50={latency['p50']:.2f} p95={latency['p95']:.2f}"
        )

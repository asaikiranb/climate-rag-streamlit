"""Helpers for building stable evaluation report payloads."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np


K_VALUES = [1, 3, 5]


def summarize_latencies(samples: list[float]) -> dict[str, float]:
    """Return aggregate latency stats for a latency sample."""

    if not samples:
        return {}
    return {
        "mean": float(np.mean(samples)),
        "p50": float(np.percentile(samples, 50)),
        "p95": float(np.percentile(samples, 95)),
        "min": float(np.min(samples)),
        "max": float(np.max(samples)),
    }


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def build_summary_payload(
    *,
    results: list[dict[str, Any]],
    latencies: dict[str, list[float]],
    reranker_name: str,
    groq_available_for_expansion: bool,
    expansion_counts: list[int],
    expansion_fallback_count: int,
    retrieval_only: bool,
) -> dict[str, Any]:
    """Build the canonical evaluation summary payload."""

    summary: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "total_questions": len(results),
            "reranker": reranker_name,
            "multi_query_expansion": True,
            "groq_available_for_expansion": bool(groq_available_for_expansion),
        },
        "retrieval_metrics": {
            f"recall@{k}": _mean([row["retrieval_metrics"][f"recall@{k}"] for row in results])
            for k in K_VALUES
        },
        "latency_summary": {stage: summarize_latencies(samples) for stage, samples in latencies.items()},
        "expansion_summary": {
            "avg_queries_per_question": _mean(expansion_counts) if expansion_counts else 1.0,
            "fallback_single_query_count": expansion_fallback_count,
            "fallback_single_query_rate": (expansion_fallback_count / len(results)) if results else 0.0,
        },
        "difficulty_breakdown": {},
    }

    for k in K_VALUES:
        summary["retrieval_metrics"][f"mrr@{k}"] = _mean(
            [row["retrieval_metrics"][f"mrr@{k}"] for row in results]
        )
        summary["retrieval_metrics"][f"ndcg@{k}"] = _mean(
            [row["retrieval_metrics"][f"ndcg@{k}"] for row in results]
        )

    if not retrieval_only:
        summary["generation_metrics"] = {
            metric: _mean([row["generation_metrics"][metric] for row in results])
            for metric in ["faithfulness", "relevance", "completeness", "overall"]
        }
        summary["citation_metrics"] = {
            "citation_validity": _mean(
                [row["citation_metrics"]["citation_validity"]["score"] for row in results]
            ),
            "citation_coverage": _mean(
                [row["citation_metrics"]["citation_coverage"]["score"] for row in results]
            ),
            "citation_grounding": _mean(
                [row["citation_metrics"]["source_grounding"]["score"] for row in results]
            ),
        }

    for difficulty in ["Easy", "Medium", "Hard"]:
        subset = [row for row in results if row["difficulty"] == difficulty]
        if not subset:
            continue

        payload: dict[str, Any] = {
            "count": len(subset),
            "recall@1": _mean([row["retrieval_metrics"]["recall@1"] for row in subset]),
            "recall@5": _mean([row["retrieval_metrics"]["recall@5"] for row in subset]),
            "mrr@5": _mean([row["retrieval_metrics"]["mrr@5"] for row in subset]),
            "ndcg@5": _mean([row["retrieval_metrics"]["ndcg@5"] for row in subset]),
        }
        if not retrieval_only:
            payload["faithfulness"] = _mean([row["generation_metrics"]["faithfulness"] for row in subset])
        summary["difficulty_breakdown"][difficulty] = payload

    return summary


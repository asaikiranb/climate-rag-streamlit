"""Latency helpers for per-stage timing summaries."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable


@dataclass(frozen=True)
class QueryLatency:
    """Internal latency breakdown for a single evaluation row."""

    total_ms: float
    embed_ms: float | None
    search_ms: float | None
    rerank_ms: float | None
    generate_ms: float | None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if len(values) == 1:
        return values[0]

    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def summarize_latency(values: Iterable[float | None]) -> dict[str, float] | None:
    """Build summary stats for a latency series, or return null if empty."""

    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None

    return {
        "mean": round(mean(numbers), 3),
        "p50": round(_percentile(numbers, 0.5), 3),
        "p95": round(_percentile(numbers, 0.95), 3),
        "min": round(min(numbers), 3),
        "max": round(max(numbers), 3),
    }


def summarize_query_latencies(latencies: list[QueryLatency]) -> dict[str, dict[str, float] | None]:
    """Build the top-level latency summary section."""

    return {
        "total": summarize_latency(entry.total_ms for entry in latencies),
        "embed": summarize_latency(entry.embed_ms for entry in latencies),
        "search": summarize_latency(entry.search_ms for entry in latencies),
        "rerank": summarize_latency(entry.rerank_ms for entry in latencies),
        "generate": summarize_latency(entry.generate_ms for entry in latencies),
    }

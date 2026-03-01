"""Normalization utilities for retrieval outputs and matching."""

from __future__ import annotations

import os
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any


def normalize_filename(value: str | None) -> str:
    """Normalize a filename for robust comparisons."""

    text = unicodedata.normalize("NFKC", value or "")
    text = os.path.basename(text)
    return text.strip().casefold()


def normalize_text(value: str | None) -> str:
    """Normalize free-form text for fuzzy matching."""

    text = unicodedata.normalize("NFKC", value or "")
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def coerce_page_number(value: Any) -> int | None:
    """Convert common page metadata formats into an integer page number."""

    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        integer = int(value)
        return integer if integer > 0 and integer == value else None

    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        page = int(text)
        return page if page > 0 else None
    return None


def normalize_retrieval(hit: dict[str, Any], rank: int) -> dict[str, Any]:
    """Project a raw retrieval hit into the evaluation schema."""

    metadata = hit.get("metadata") or hit.get("meta") or {}
    filename = metadata.get("filename") or metadata.get("file") or ""
    page = (
        metadata.get("page_number")
        or metadata.get("page")
        or metadata.get("page_num")
        or metadata.get("pageNumber")
    )
    score = (
        hit.get("fused_score")
        if hit.get("fused_score") is not None
        else hit.get("rrf_score")
    )

    return {
        "rank": rank,
        "filename": str(filename).strip() or None,
        "page": coerce_page_number(page),
        "snippet": (hit.get("document") or hit.get("doc") or "").strip() or None,
        "score": float(score) if score is not None else None,
    }


def normalize_retrievals(hits: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    """Normalize the top-N retrievals."""

    return [normalize_retrieval(hit, rank) for rank, hit in enumerate(hits[:top_n], start=1)]


def anchor_similarity(anchor_text: str, snippet: str | None) -> float:
    """Compute a best-effort fuzzy match score from 0 to 100."""

    anchor = normalize_text(anchor_text)
    haystack = normalize_text(snippet)
    if not anchor or not haystack:
        return 0.0
    if anchor in haystack:
        return 100.0

    best = SequenceMatcher(None, anchor, haystack).ratio() * 100.0
    anchor_tokens = anchor.split()
    haystack_tokens = haystack.split()
    if not anchor_tokens or not haystack_tokens:
        return round(best, 2)

    target_width = len(anchor_tokens)
    window_sizes = {target_width}
    if target_width > 1:
        window_sizes.add(target_width - 1)
    window_sizes.add(target_width + 1)

    for window_size in sorted(window_sizes):
        if window_size <= 0 or window_size > len(haystack_tokens):
            continue
        for start in range(0, len(haystack_tokens) - window_size + 1):
            window = " ".join(haystack_tokens[start : start + window_size])
            score = SequenceMatcher(None, anchor, window).ratio() * 100.0
            if score > best:
                best = score
            if best >= 100.0:
                return 100.0

    return round(best, 2)


def anchor_matches(anchor_text: str, snippet: str | None, threshold: int = 80) -> bool:
    """Return whether the snippet passes the fuzzy anchor threshold."""

    return anchor_similarity(anchor_text, snippet) >= float(threshold)

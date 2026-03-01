"""Custom citation metrics and offline retrieval-eval scoring helpers."""

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from eval.normalize import anchor_matches, normalize_filename


def parse_citations(answer_text: str) -> List[int]:
    """Pull all [N] citation numbers from the answer."""
    return [int(m) for m in re.findall(r'\[(\d+)\]', answer_text)]


def citation_validity(answer_text: str, num_sources: int) -> Dict:
    """Check if every cited number is within the valid source range."""
    citations = parse_citations(answer_text)

    if not citations:
        return {
            "score": 0.0,
            "total_citations": 0,
            "valid_citations": 0,
            "invalid_citations": [],
            "note": "No citations found",
        }

    valid = [c for c in citations if 1 <= c <= num_sources]
    invalid = [c for c in citations if c < 1 or c > num_sources]

    return {
        "score": len(valid) / len(citations) if citations else 0.0,
        "total_citations": len(citations),
        "valid_citations": len(valid),
        "invalid_citations": invalid,
    }


def citation_coverage(answer_text: str) -> Dict:
    """What fraction of factual sentences have at least one [N] citation?"""
    sentences = re.split(r'(?<=[.!?])\s+', answer_text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if not sentences:
        return {"score": 0.0, "total_sentences": 0, "cited_sentences": 0}

    # Skip non-factual sentences
    skip_phrases = ["documents don't cover", "not enough information", "i don't know", "no information"]
    factual_sentences = [s for s in sentences if not any(p in s.lower() for p in skip_phrases)]

    if not factual_sentences:
        return {"score": 1.0, "total_sentences": 0, "cited_sentences": 0, "note": "No factual claims"}

    cited = [s for s in factual_sentences if re.search(r'\[\d+\]', s)]

    return {
        "score": len(cited) / len(factual_sentences),
        "total_sentences": len(factual_sentences),
        "cited_sentences": len(cited),
    }


STOPWORDS = {
    'this', 'that', 'with', 'from', 'have', 'been', 'were',
    'their', 'which', 'these', 'those', 'also', 'into', 'such',
    'more', 'than', 'them', 'only', 'some', 'each', 'other',
    'about', 'would', 'could', 'should', 'does', 'very',
}


def source_grounding(answer_text: str, results: List[Dict]) -> Dict:
    """
    For each [N] citation, check if key terms from that sentence
    actually appear in the corresponding source chunk. Fast lexical check.
    """
    sentences = re.split(r'(?<=[.!?])\s+', answer_text.strip())
    checks = []

    for sentence in sentences:
        citation_nums = [int(m) for m in re.findall(r'\[(\d+)\]', sentence)]
        if not citation_nums:
            continue

        # Get meaningful words (4+ chars, skip stopwords)
        clean = re.sub(r'\[\d+\]', '', sentence)
        words = set(
            w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', clean)
            if w.lower() not in STOPWORDS
        )

        for num in citation_nums:
            idx = num - 1
            if 0 <= idx < len(results):
                source_text = results[idx]['document'].lower()
                matched = [w for w in words if w in source_text]
                score = len(matched) / len(words) if words else 0.0
                checks.append({
                    "citation": num,
                    "score": round(score, 3),
                    "matched_terms": len(matched),
                    "total_terms": len(words),
                })

    if not checks:
        return {"score": 0.0, "per_citation": [], "note": "No citations to verify"}

    avg_score = sum(c['score'] for c in checks) / len(checks)
    return {
        "score": round(avg_score, 3),
        "per_citation": checks,
    }


def compute_custom_metrics(answer_text: str, results: List[Dict]) -> Dict:
    """Run all custom citation metrics."""
    return {
        "citation_validity": citation_validity(answer_text, len(results)),
        "citation_coverage": citation_coverage(answer_text),
        "source_grounding": source_grounding(answer_text, results),
    }


K_VALUES = (1, 3, 5, 10)


@dataclass(frozen=True, slots=True)
class RetrievalScore:
    """Per-row retrieval metrics for either doc or page evaluation."""

    scored: bool
    rr: float | None
    hits: dict[int, int | None]
    ndcg: dict[int, float | None]


def _rank_gain(rank: int, gain: int) -> float:
    return float(gain) / math.log2(rank + 1)


def _doc_first_match_rank(
    retrievals: list[dict[str, Any]],
    gold_source: str,
) -> int | None:
    target = normalize_filename(gold_source)
    if not target:
        return None

    for retrieval in retrievals:
        if normalize_filename(retrieval.get("filename")) == target:
            return int(retrieval["rank"])
    return None


def compute_doc_retrieval_scores(
    retrievals: list[dict[str, Any]],
    gold_source: str,
    k_values: tuple[int, ...] = K_VALUES,
) -> RetrievalScore:
    """Compute document-level metrics for a single row."""

    if not gold_source.strip():
        return RetrievalScore(
            scored=False,
            rr=None,
            hits={k: None for k in k_values},
            ndcg={k: None for k in k_values},
        )

    rank = _doc_first_match_rank(retrievals, gold_source)
    rr = (1.0 / rank) if rank is not None and rank <= 10 else 0.0

    hits = {
        k: 1 if rank is not None and rank <= k else 0
        for k in k_values
    }
    ndcg = {
        k: (_rank_gain(rank, 1) if rank is not None and rank <= k else 0.0)
        for k in k_values
    }

    return RetrievalScore(scored=True, rr=rr, hits=hits, ndcg=ndcg)


def _page_best_rank_and_gain(
    retrievals: list[dict[str, Any]],
    gold_source: str,
    gold_pages: list[int] | None,
    anchor_text: str,
    anchor_threshold: int,
) -> tuple[int | None, int]:
    target = normalize_filename(gold_source)
    if not target:
        return None, 0

    has_anchor = bool(anchor_text.strip())
    has_pages = bool(gold_pages)
    page_set = set(gold_pages or [])

    best_rank: int | None = None
    best_gain = 0

    for retrieval in retrievals:
        if normalize_filename(retrieval.get("filename")) != target:
            continue

        page_hit = has_pages and retrieval.get("page") in page_set
        anchor_hit = False
        if not page_hit and has_anchor:
            anchor_hit = anchor_matches(
                anchor_text,
                retrieval.get("snippet"),
                threshold=anchor_threshold,
            )

        gain = 2 if page_hit or anchor_hit else 1
        rank = int(retrieval["rank"])

        if gain > best_gain or (gain == best_gain and best_rank is not None and rank < best_rank):
            best_rank = rank
            best_gain = gain
        elif best_rank is None:
            best_rank = rank
            best_gain = gain

        if best_gain == 2 and best_rank == 1:
            break

    return best_rank, best_gain


def compute_page_retrieval_scores(
    retrievals: list[dict[str, Any]],
    gold_source: str,
    gold_pages: list[int] | None,
    anchor_text: str,
    anchor_threshold: int,
    k_values: tuple[int, ...] = K_VALUES,
) -> RetrievalScore:
    """Compute page-level metrics for a single row."""

    if not gold_source.strip() or (not gold_pages and not anchor_text.strip()):
        return RetrievalScore(
            scored=False,
            rr=None,
            hits={k: None for k in k_values},
            ndcg={k: None for k in k_values},
        )

    rank, gain = _page_best_rank_and_gain(
        retrievals=retrievals,
        gold_source=gold_source,
        gold_pages=gold_pages,
        anchor_text=anchor_text,
        anchor_threshold=anchor_threshold,
    )

    rr = (1.0 / rank) if rank is not None and gain == 2 and rank <= 10 else 0.0
    hits = {
        k: 1 if rank is not None and gain == 2 and rank <= k else 0
        for k in k_values
    }
    ndcg = {
        k: (_rank_gain(rank, gain) / 2.0 if rank is not None and rank <= k and gain > 0 else 0.0)
        for k in k_values
    }

    return RetrievalScore(scored=True, rr=rr, hits=hits, ndcg=ndcg)


def aggregate_scored_metrics(
    rows: list[dict[str, Any]],
    prefix: str,
    k_values: tuple[int, ...] = K_VALUES,
) -> dict[str, Any]:
    """Aggregate only the rows that were actually scored."""

    scored_rows = [row for row in rows if row.get(f"{prefix}_scored")]
    summary: dict[str, Any] = {"count": len(scored_rows)}

    if not scored_rows:
        summary["mrr@10"] = None
        for k in k_values:
            summary[f"recall@{k}"] = None
            summary[f"ndcg@{k}"] = None
        return summary

    summary["mrr@10"] = round(
        sum(float(row[f"{prefix}_rr"]) for row in scored_rows) / len(scored_rows),
        6,
    )
    for k in k_values:
        summary[f"recall@{k}"] = round(
            sum(int(row[f"{prefix}_hit@{k}"]) for row in scored_rows) / len(scored_rows),
            6,
        )
        summary[f"ndcg@{k}"] = round(
            sum(float(row[f"{prefix}_ndcg@{k}"]) for row in scored_rows) / len(scored_rows),
            6,
        )
    return summary

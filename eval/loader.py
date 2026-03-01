"""CSV loading helpers for offline retrieval evaluation."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPECTED_HEADERS = ["Question", "gold_sources", "metadata", "page_range", "anchor_text"]


@dataclass(frozen=True, slots=True)
class EvalRow:
    """Normalized evaluation input row."""

    row_index: int
    question_id: str
    question: str
    gold_sources: str
    metadata: dict[str, Any]
    metadata_raw: str
    difficulty: str
    gold_pages: list[int] | None
    anchor_text: str


def parse_page_range(raw_value: str) -> list[int] | None:
    """Parse a page range cell into a list of positive page numbers."""

    text = (raw_value or "").strip()
    if not text:
        return None

    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Invalid page_range value: {raw_value!r}") from exc

    if isinstance(value, int):
        candidates = [value]
    elif isinstance(value, (list, tuple)):
        candidates = list(value)
    else:
        raise ValueError(f"Unsupported page_range value: {raw_value!r}")

    pages: list[int] = []
    for candidate in candidates:
        if isinstance(candidate, bool) or not isinstance(candidate, int):
            raise ValueError(f"Non-integer page in page_range: {raw_value!r}")
        if candidate <= 0:
            raise ValueError(f"Page numbers must be positive: {raw_value!r}")
        pages.append(candidate)

    return pages or None


def build_question_id(row_index: int, question: str) -> str:
    """Build a stable question identifier from the row index and question text."""

    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]
    return f"q{row_index:05d}_{digest}"


def _parse_metadata(raw_value: str) -> dict[str, Any]:
    text = (raw_value or "").strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}

    if isinstance(parsed, dict):
        return parsed
    return {"value": parsed}


def _normalize_gold_source(raw_value: str) -> str:
    text = (raw_value or "").strip()
    if not text:
        return ""

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text

    if isinstance(parsed, list) and parsed:
        first = parsed[0]
        return str(first).strip()
    return str(parsed).strip()


def _validate_headers(headers: list[str] | None) -> None:
    if headers is None:
        raise ValueError("CSV is missing a header row.")
    normalized = list(headers)
    if normalized:
        normalized[0] = normalized[0].lstrip("\ufeff")
    if normalized != EXPECTED_HEADERS:
        raise ValueError(
            "CSV headers must exactly match "
            f"{EXPECTED_HEADERS}, received {normalized}."
        )


def load_golden_csv(path: str | Path, limit: int | None = None) -> list[EvalRow]:
    """Load and validate the golden CSV."""

    rows: list[EvalRow] = []
    csv_path = Path(path)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _validate_headers(reader.fieldnames)

        for row_index, raw_row in enumerate(reader, start=1):
            question = (raw_row["Question"] or "").strip()
            gold_sources = _normalize_gold_source(raw_row["gold_sources"])
            metadata_raw = raw_row["metadata"] or ""
            metadata = _parse_metadata(metadata_raw)
            difficulty = str(metadata.get("difficulty", "Unknown")).strip() or "Unknown"

            rows.append(
                EvalRow(
                    row_index=row_index,
                    question_id=build_question_id(row_index, question),
                    question=question,
                    gold_sources=gold_sources,
                    metadata=metadata,
                    metadata_raw=metadata_raw,
                    difficulty=difficulty,
                    gold_pages=parse_page_range(raw_row["page_range"] or ""),
                    anchor_text=(raw_row["anchor_text"] or "").strip(),
                )
            )

            if limit is not None and len(rows) >= limit:
                break

    return rows

"""Shared helpers for deterministic text-pipeline behavior."""

from __future__ import annotations

import re
import uuid


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def stable_chunk_id(
    filename: str,
    doc_hash: str,
    page_number: int,
    section_idx: int,
    chunk_idx: int,
) -> str:
    normalized_filename = normalize_whitespace((filename or "").lower())
    return f"{normalized_filename}:{doc_hash}:{page_number}:{section_idx}:{chunk_idx}"


def chunk_id_to_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

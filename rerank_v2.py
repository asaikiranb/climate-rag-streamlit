"""Single-stage calibrated reranker with optional doc-first aggregation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder

from config import (
    DOC_FIRST_AGGREGATE_DECAY,
    DOC_FIRST_AGGREGATE_TOP_K,
    DOC_FIRST_RERANK_ENABLED,
    MODEL_DEVICE,
    RERANK_STAGE1_MODEL,
    RERANK_STAGE1_POOL_SIZE,
)
from hf_local import resolve_local_snapshot


def _calibrate(scores: list[float]) -> list[float]:
    if not scores:
        return []
    values = np.array(scores, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    std = float(values.std())
    if std == 0.0:
        z = np.zeros_like(values)
    else:
        z = (values - float(values.mean())) / std
    low = float(z.min())
    high = float(z.max())
    if high == low:
        return [1.0 for _ in scores]
    normalized = (z - low) / (high - low)
    return [float(value) for value in normalized]


class TwoStageCalibratedReranker:
    """Backwards-compatible name; implementation is stage-1 only."""

    def __init__(
        self,
        stage1_model: str = RERANK_STAGE1_MODEL,
        stage1_pool_size: int = RERANK_STAGE1_POOL_SIZE,
        doc_first_enabled: bool = DOC_FIRST_RERANK_ENABLED,
        doc_first_aggregate_top_k: int = DOC_FIRST_AGGREGATE_TOP_K,
        doc_first_aggregate_decay: float = DOC_FIRST_AGGREGATE_DECAY,
        **_: Any,
    ) -> None:
        self.stage1_model_name = stage1_model
        self.model_name = stage1_model
        self.stage1_pool_size = max(1, int(stage1_pool_size))
        self.doc_first_enabled = bool(doc_first_enabled)
        self.doc_first_aggregate_top_k = max(1, int(doc_first_aggregate_top_k))
        self.doc_first_aggregate_decay = max(0.0, float(doc_first_aggregate_decay))

        self.stage1_model = self._load_cross_encoder(self.stage1_model_name)
        if self.stage1_model is None:
            raise RuntimeError(f"Unable to load reranker model: {self.stage1_model_name}")

    @staticmethod
    def _load_cross_encoder(model_name: str) -> CrossEncoder | None:
        local_snapshot = resolve_local_snapshot(model_name)
        try:
            if local_snapshot:
                return CrossEncoder(local_snapshot, device=MODEL_DEVICE)
        except Exception:
            pass

        try:
            return CrossEncoder(model_name, device=MODEL_DEVICE)
        except Exception:
            return None

    @staticmethod
    def _doc_group_key(row: dict[str, Any]) -> str:
        metadata = row.get("metadata") or {}
        for field in ("filename", "source_path", "title"):
            value = metadata.get(field)
            if value:
                return str(value)
        return str(row.get("id", ""))

    def _doc_aggregate_score(self, rows: list[dict[str, Any]]) -> float:
        score = 0.0
        weight = 1.0
        for row in rows[: self.doc_first_aggregate_top_k]:
            chunk_score = float(row.get("fused_score", 0.0) or 0.0)
            score += weight * chunk_score
            weight *= self.doc_first_aggregate_decay
        return float(score)

    def _apply_doc_first_rank(self, ranked_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.doc_first_enabled or not ranked_rows:
            return ranked_rows

        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in ranked_rows:
            key = self._doc_group_key(row)
            grouped.setdefault(key, []).append(row)

        for rows in grouped.values():
            rows.sort(
                key=lambda item: (item.get("fused_score", 0.0), str(item.get("id", ""))),
                reverse=True,
            )

        doc_rank = sorted(
            grouped.items(),
            key=lambda item: (
                self._doc_aggregate_score(item[1]),
                str(item[0]),
            ),
            reverse=True,
        )

        output: list[dict[str, Any]] = []
        max_len = max(len(rows) for _, rows in doc_rank)
        for depth in range(max_len):
            for _, rows in doc_rank:
                if depth < len(rows):
                    output.append(rows[depth])

        return output

    def rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []

        pool_size = min(self.stage1_pool_size, len(candidates))
        stage1_pool = [dict(row) for row in candidates[:pool_size]]
        remainder = [dict(row) for row in candidates[pool_size:]]

        pairs = [(query, row.get("document", "")) for row in stage1_pool]
        stage1_pred = self.stage1_model.predict(pairs) if pairs else []

        stage1_raw: list[float] = []
        stage1_finite_count = 0
        for value in stage1_pred:
            score = float(value)
            if np.isfinite(score):
                stage1_finite_count += 1
            else:
                score = 0.0
            stage1_raw.append(score)

        stage1_has_signal = stage1_finite_count > 0
        stage1_cal = _calibrate(stage1_raw) if stage1_has_signal else [0.0 for _ in stage1_raw]

        ranked: list[dict[str, Any]] = []

        if not stage1_has_signal:
            for row in candidates:
                out = dict(row)
                metadata = out.get("metadata") or {}
                retrieval_score = float(out.get("retrieval_score", out.get("fused_score", 0.0) or 0.0))
                meta_prior = float(metadata.get("metadata_prior", out.get("metadata_prior", 0.0) or 0.0))
                out["stage1_raw"] = 0.0
                out["stage1_score"] = 0.0
                out["fused_score"] = float(0.95 * retrieval_score + 0.05 * meta_prior)
                ranked.append(out)

            ranked.sort(key=lambda row: (row.get("fused_score", 0.0), str(row.get("id", ""))), reverse=True)
            return self._apply_doc_first_rank(ranked)

        for idx, row in enumerate(stage1_pool):
            out = dict(row)
            metadata = out.get("metadata") or {}
            retrieval_score = float(out.get("retrieval_score", out.get("fused_score", 0.0) or 0.0))
            meta_prior = float(metadata.get("metadata_prior", out.get("metadata_prior", 0.0) or 0.0))
            out["stage1_raw"] = stage1_raw[idx]
            out["stage1_score"] = stage1_cal[idx]
            out["fused_score"] = float(0.75 * out["stage1_score"] + 0.20 * retrieval_score + 0.05 * meta_prior)
            ranked.append(out)

        for row in remainder:
            out = dict(row)
            metadata = out.get("metadata") or {}
            retrieval_score = float(out.get("retrieval_score", out.get("fused_score", 0.0) or 0.0))
            meta_prior = float(metadata.get("metadata_prior", out.get("metadata_prior", 0.0) or 0.0))
            out["stage1_raw"] = 0.0
            out["stage1_score"] = 0.0
            out["fused_score"] = float(0.95 * retrieval_score + 0.05 * meta_prior)
            ranked.append(out)

        ranked.sort(key=lambda row: (row.get("fused_score", 0.0), str(row.get("id", ""))), reverse=True)
        return self._apply_doc_first_rank(ranked)

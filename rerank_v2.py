"""Two-stage calibrated reranker with local fallback handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder

from config import (
    RERANK_FINAL_META_WEIGHT,
    RERANK_FINAL_RETRIEVAL_WEIGHT,
    RERANK_FINAL_STAGE1_WEIGHT,
    RERANK_FINAL_STAGE2_WEIGHT,
    RERANK_STAGE1_MODEL,
    RERANK_STAGE1_POOL_SIZE,
    RERANK_STAGE2_ALT_MODEL,
    RERANK_STAGE2_FALLBACK_MODEL,
    RERANK_STAGE2_MODEL,
    RERANK_STAGE2_POOL_SIZE,
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


@dataclass(frozen=True, slots=True)
class Stage2ModelInfo:
    name: str
    source: str


class TwoStageCalibratedReranker:
    """Fast stage-1 reranking followed by stronger calibrated stage-2 reranking."""

    def __init__(
        self,
        stage1_model: str = RERANK_STAGE1_MODEL,
        stage2_model: str = RERANK_STAGE2_MODEL,
        stage2_fallback_model: str = RERANK_STAGE2_FALLBACK_MODEL,
        stage2_alt_model: str = RERANK_STAGE2_ALT_MODEL,
        stage1_pool_size: int = RERANK_STAGE1_POOL_SIZE,
        stage2_pool_size: int = RERANK_STAGE2_POOL_SIZE,
        use_alt_stage2: bool = False,
    ) -> None:
        self.stage1_model_name = stage1_model
        self.stage2_model_name = stage2_model
        self.stage2_fallback_model_name = stage2_fallback_model
        self.stage2_alt_model_name = stage2_alt_model
        self.stage1_pool_size = max(1, int(stage1_pool_size))
        self.stage2_pool_size = max(1, int(stage2_pool_size))
        self.use_alt_stage2 = bool(use_alt_stage2)

        self.stage1_model = self._load_cross_encoder(self.stage1_model_name)
        if self.stage1_model is None:
            raise RuntimeError(f"Unable to load stage-1 reranker model: {self.stage1_model_name}")
        self.stage2_model, self.stage2_info = self._load_stage2_model()

    def _load_cross_encoder(self, model_name: str) -> CrossEncoder | None:
        local_snapshot = resolve_local_snapshot(model_name)
        if local_snapshot:
            try:
                return CrossEncoder(local_snapshot)
            except Exception:
                pass
        try:
            return CrossEncoder(model_name)
        except Exception:
            return None

    def _load_stage2_model(self) -> tuple[CrossEncoder | None, Stage2ModelInfo]:
        if self.use_alt_stage2:
            alt_model = self._load_cross_encoder(self.stage2_alt_model_name)
            if alt_model is not None:
                return alt_model, Stage2ModelInfo(name=self.stage2_alt_model_name, source="alt")
            return None, Stage2ModelInfo(name="none", source="missing")

        primary = self._load_cross_encoder(self.stage2_model_name)
        if primary is not None:
            return primary, Stage2ModelInfo(name=self.stage2_model_name, source="primary")

        fallback = self._load_cross_encoder(self.stage2_fallback_model_name)
        if fallback is not None:
            return fallback, Stage2ModelInfo(name=self.stage2_fallback_model_name, source="fallback")

        alt = self._load_cross_encoder(self.stage2_alt_model_name)
        if alt is not None:
            return alt, Stage2ModelInfo(name=self.stage2_alt_model_name, source="alt")

        return None, Stage2ModelInfo(name="none", source="missing")

    def rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []

        stage1_pool = candidates[: min(self.stage1_pool_size, len(candidates))]
        remainder = candidates[len(stage1_pool) :]
        stage1_pairs = [(query, candidate.get("document", "")) for candidate in stage1_pool]
        stage1_pred = self.stage1_model.predict(stage1_pairs)
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

        stage1_rows = []
        for idx, candidate in enumerate(stage1_pool):
            row = dict(candidate)
            row["stage1_raw"] = stage1_raw[idx]
            row["stage1_score"] = stage1_cal[idx]
            stage1_rows.append(row)

        stage1_rows.sort(key=lambda row: row.get("stage1_score", 0.0), reverse=True)

        if not stage1_has_signal and self.stage2_model is None:
            retrieval_only_ranked: list[dict[str, Any]] = []
            for candidate in candidates:
                row = dict(candidate)
                metadata = row.get("metadata") or {}
                retrieval_score = float(row.get("retrieval_score", row.get("fused_score", 0.0) or 0.0))
                meta_prior = float(metadata.get("metadata_prior", row.get("metadata_prior", 0.0) or 0.0))
                fused_score = 0.95 * retrieval_score + 0.05 * meta_prior
                row["stage2_model"] = self.stage2_info.name
                row["stage2_source"] = self.stage2_info.source
                row["stage1_raw"] = 0.0
                row["stage1_score"] = 0.0
                row["stage2_raw"] = 0.0
                row["stage2_score"] = 0.0
                row["fused_score"] = float(fused_score)
                retrieval_only_ranked.append(row)

            retrieval_only_ranked.sort(
                key=lambda row: (row.get("fused_score", 0.0), str(row.get("id", ""))),
                reverse=True,
            )
            return retrieval_only_ranked

        if self.stage2_model is None:
            ranked_one_stage: list[dict[str, Any]] = []
            for row in stage1_rows:
                metadata = row.get("metadata") or {}
                retrieval_score = float(row.get("retrieval_score", row.get("fused_score", 0.0) or 0.0))
                meta_prior = float(metadata.get("metadata_prior", row.get("metadata_prior", 0.0) or 0.0))
                final_score = (
                    0.75 * float(row.get("stage1_score", 0.0))
                    + 0.20 * retrieval_score
                    + 0.05 * meta_prior
                )
                out = dict(row)
                out["stage2_model"] = self.stage2_info.name
                out["stage2_source"] = self.stage2_info.source
                out["stage2_raw"] = 0.0
                out["stage2_score"] = 0.0
                out["fused_score"] = float(final_score)
                ranked_one_stage.append(out)

            for row in remainder:
                out = dict(row)
                metadata = out.get("metadata") or {}
                retrieval_score = float(out.get("retrieval_score", out.get("fused_score", 0.0) or 0.0))
                meta_prior = float(metadata.get("metadata_prior", out.get("metadata_prior", 0.0) or 0.0))
                out["stage2_model"] = self.stage2_info.name
                out["stage2_source"] = self.stage2_info.source
                out["stage1_raw"] = 0.0
                out["stage1_score"] = 0.0
                out["stage2_raw"] = 0.0
                out["stage2_score"] = 0.0
                out["fused_score"] = float(0.95 * retrieval_score + 0.05 * meta_prior)
                ranked_one_stage.append(out)

            ranked_one_stage.sort(
                key=lambda row: (row.get("fused_score", 0.0), str(row.get("id", ""))),
                reverse=True,
            )
            return ranked_one_stage

        stage2_pool = stage1_rows[: min(self.stage2_pool_size, len(stage1_rows))]

        if self.stage2_model is not None and stage2_pool:
            stage2_pairs = [(query, row.get("document", "")) for row in stage2_pool]
            stage2_raw = [float(np.nan_to_num(float(value), nan=0.0, posinf=0.0, neginf=0.0)) for value in self.stage2_model.predict(stage2_pairs)]
            stage2_cal = _calibrate(stage2_raw)
        else:
            stage2_raw = [0.0 for _ in stage2_pool]
            stage2_cal = [0.0 for _ in stage2_pool]

        ranked: list[dict[str, Any]] = []
        for idx, row in enumerate(stage2_pool):
            metadata = row.get("metadata") or {}
            retrieval_score = float(row.get("retrieval_score", row.get("fused_score", 0.0) or 0.0))
            meta_prior = float(metadata.get("metadata_prior", row.get("metadata_prior", 0.0) or 0.0))
            final_score = (
                RERANK_FINAL_STAGE2_WEIGHT * stage2_cal[idx]
                + RERANK_FINAL_STAGE1_WEIGHT * float(row.get("stage1_score", 0.0))
                + RERANK_FINAL_RETRIEVAL_WEIGHT * retrieval_score
                + RERANK_FINAL_META_WEIGHT * meta_prior
            )
            out = dict(row)
            out["stage2_model"] = self.stage2_info.name
            out["stage2_source"] = self.stage2_info.source
            out["stage2_raw"] = stage2_raw[idx]
            out["stage2_score"] = stage2_cal[idx]
            out["fused_score"] = float(final_score)
            ranked.append(out)

        overflow = stage1_rows[len(stage2_pool) :]
        for row in overflow:
            metadata = row.get("metadata") or {}
            retrieval_score = float(row.get("retrieval_score", row.get("fused_score", 0.0) or 0.0))
            meta_prior = float(metadata.get("metadata_prior", row.get("metadata_prior", 0.0) or 0.0))
            fallback_score = (
                0.45 * float(row.get("stage1_score", 0.0))
                + 0.45 * retrieval_score
                + 0.10 * meta_prior
            )
            out = dict(row)
            out["stage2_model"] = self.stage2_info.name
            out["stage2_source"] = self.stage2_info.source
            out["stage2_raw"] = 0.0
            out["stage2_score"] = 0.0
            out["fused_score"] = float(fallback_score)
            ranked.append(out)

        for row in remainder:
            out = dict(row)
            out["stage2_model"] = self.stage2_info.name
            out["stage2_source"] = self.stage2_info.source
            out["stage2_raw"] = 0.0
            out["stage2_score"] = 0.0
            out["fused_score"] = float(out.get("retrieval_score", out.get("fused_score", 0.0) or 0.0))
            ranked.append(out)

        ranked.sort(key=lambda row: (row.get("fused_score", 0.0), str(row.get("id", ""))), reverse=True)
        return ranked

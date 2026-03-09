"""Two-stage calibrated reranker with local fallback handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder

from config import (
    CONTEXTUAL_STAGE2_MODEL_IDS,
    DOC_FIRST_AGGREGATE_DECAY,
    DOC_FIRST_AGGREGATE_TOP_K,
    DOC_FIRST_RERANK_ENABLED,
    MODEL_DEVICE,
    REQUIRE_STAGE2,
    RERANK_FINAL_META_WEIGHT,
    RERANK_FINAL_RETRIEVAL_WEIGHT,
    RERANK_FINAL_STAGE1_WEIGHT,
    RERANK_FINAL_STAGE2_WEIGHT,
    RERANK_STAGE1_MODEL,
    RERANK_STAGE1_POOL_SIZE,
    RERANK_STAGE2_ALT_MODEL,
    RERANK_STAGE2_BACKEND,
    RERANK_STAGE2_FALLBACK_MODEL,
    RERANK_STAGE2_MODEL,
    RERANK_STAGE2_POOL_SIZE,
)
from hf_local import resolve_local_snapshot
from rerank_contextual import ContextualRerankerBackend


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
    backend: str
    available: bool


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
        require_stage2: bool = REQUIRE_STAGE2,
        stage2_backend: str = RERANK_STAGE2_BACKEND,
        disable_stage2: bool = False,
        doc_first_enabled: bool = DOC_FIRST_RERANK_ENABLED,
        doc_first_aggregate_top_k: int = DOC_FIRST_AGGREGATE_TOP_K,
        doc_first_aggregate_decay: float = DOC_FIRST_AGGREGATE_DECAY,
    ) -> None:
        self.stage1_model_name = stage1_model
        self.stage2_model_name = stage2_model
        self.stage2_fallback_model_name = stage2_fallback_model
        self.stage2_alt_model_name = stage2_alt_model
        self.stage1_pool_size = max(1, int(stage1_pool_size))
        self.stage2_pool_size = max(1, int(stage2_pool_size))
        self.use_alt_stage2 = bool(use_alt_stage2)
        self.require_stage2 = bool(require_stage2)
        self.stage2_backend = (stage2_backend or "cross_encoder").strip().lower()
        self.disable_stage2 = bool(disable_stage2)
        self.doc_first_enabled = bool(doc_first_enabled)
        self.doc_first_aggregate_top_k = max(1, int(doc_first_aggregate_top_k))
        self.doc_first_aggregate_decay = max(0.0, float(doc_first_aggregate_decay))

        if self.disable_stage2 and self.require_stage2:
            raise RuntimeError("Invalid reranker config: disable_stage2=true is incompatible with require_stage2=true")

        self.stage1_model = self._load_cross_encoder(self.stage1_model_name)
        if self.stage1_model is None:
            raise RuntimeError(f"Unable to load stage-1 reranker model: {self.stage1_model_name}")
        if self.disable_stage2:
            self.stage2_model = None
            self.stage2_info = Stage2ModelInfo(
                name="none",
                source="disabled",
                backend="disabled",
                available=False,
            )
        else:
            self.stage2_model, self.stage2_info = self._load_stage2_model()
        if self.require_stage2 and not self.stage2_info.available:
            raise RuntimeError(
                "stage2_model_unavailable: strict mode enabled and no stage-2 reranker could be loaded"
            )

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
            grouped.setdefault(key, []).append(dict(row))

        doc_entries: list[tuple[str, float, list[dict[str, Any]]]] = []
        for key, rows in grouped.items():
            rows.sort(key=lambda item: (float(item.get("fused_score", 0.0) or 0.0), str(item.get("id", ""))), reverse=True)
            doc_score = self._doc_aggregate_score(rows)
            doc_entries.append((key, doc_score, rows))

        doc_entries.sort(key=lambda item: (item[1], item[0]), reverse=True)

        reordered: list[dict[str, Any]] = []
        for doc_rank, (_key, doc_score, rows) in enumerate(doc_entries, start=1):
            top_row = dict(rows[0])
            top_row["doc_rank"] = doc_rank
            top_row["doc_score"] = float(doc_score)
            top_row["doc_chunk_rank"] = 1
            reordered.append(top_row)

        chunk_offset = 1
        while True:
            any_added = False
            for doc_rank, (_key, doc_score, rows) in enumerate(doc_entries, start=1):
                if chunk_offset >= len(rows):
                    continue
                row = dict(rows[chunk_offset])
                row["doc_rank"] = doc_rank
                row["doc_score"] = float(doc_score)
                row["doc_chunk_rank"] = chunk_offset + 1
                reordered.append(row)
                any_added = True
            if not any_added:
                break
            chunk_offset += 1

        return reordered

    def _with_stage2_metadata(self, row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        metadata = dict(out.get("metadata") or {})
        metadata["stage2_model"] = self.stage2_info.name
        metadata["stage2_source"] = self.stage2_info.source
        metadata["stage2_backend"] = self.stage2_info.backend
        metadata["stage2_available"] = self.stage2_info.available
        out["metadata"] = metadata
        return out

    def _load_cross_encoder(self, model_name: str) -> CrossEncoder | None:
        local_snapshot = resolve_local_snapshot(model_name)
        if local_snapshot:
            try:
                return CrossEncoder(local_snapshot, device=MODEL_DEVICE)
            except Exception:
                pass
        try:
            return CrossEncoder(model_name, device=MODEL_DEVICE)
        except Exception:
            return None

    def _load_contextual_backend(self, model_name: str) -> ContextualRerankerBackend | None:
        try:
            return ContextualRerankerBackend(model_name=model_name, backend=self.stage2_backend)
        except Exception:
            return None

    @staticmethod
    def _dedup_model_names(values: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for value in values:
            model = value.strip()
            if not model or model in seen:
                continue
            seen.add(model)
            out.append(model)
        return out

    def _load_stage2_model(self) -> tuple[Any | None, Stage2ModelInfo]:
        if self.use_alt_stage2:
            alt_model = self._load_cross_encoder(self.stage2_alt_model_name)
            if alt_model is not None:
                return alt_model, Stage2ModelInfo(
                    name=self.stage2_alt_model_name,
                    source="alt",
                    backend="cross_encoder",
                    available=True,
                )
            return None, Stage2ModelInfo(name="none", source="missing", backend="cross_encoder", available=False)

        if self.stage2_backend in {"contextual_hf", "contextual_vllm"}:
            contextual_candidates = self._dedup_model_names(
                [
                    self.stage2_model_name,
                    self.stage2_fallback_model_name,
                    *CONTEXTUAL_STAGE2_MODEL_IDS,
                ]
            )
            for index, model_name in enumerate(contextual_candidates):
                model = self._load_contextual_backend(model_name)
                if model is None:
                    continue
                source = "primary" if index == 0 else ("fallback" if index == 1 else "config_list")
                return model, Stage2ModelInfo(
                    name=model_name,
                    source=source,
                    backend=self.stage2_backend,
                    available=True,
                )

            alt = self._load_cross_encoder(self.stage2_alt_model_name)
            if alt is not None:
                return alt, Stage2ModelInfo(
                    name=self.stage2_alt_model_name,
                    source="alt",
                    backend="cross_encoder",
                    available=True,
                )
            return None, Stage2ModelInfo(name="none", source="missing", backend=self.stage2_backend, available=False)

        primary = self._load_cross_encoder(self.stage2_model_name)
        if primary is not None:
            return primary, Stage2ModelInfo(
                name=self.stage2_model_name,
                source="primary",
                backend="cross_encoder",
                available=True,
            )

        fallback = self._load_cross_encoder(self.stage2_fallback_model_name)
        if fallback is not None:
            return fallback, Stage2ModelInfo(
                name=self.stage2_fallback_model_name,
                source="fallback",
                backend="cross_encoder",
                available=True,
            )

        alt = self._load_cross_encoder(self.stage2_alt_model_name)
        if alt is not None:
            return alt, Stage2ModelInfo(
                name=self.stage2_alt_model_name,
                source="alt",
                backend="cross_encoder",
                available=True,
            )

        return None, Stage2ModelInfo(name="none", source="missing", backend="cross_encoder", available=False)

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
                row = self._with_stage2_metadata(candidate)
                metadata = row.get("metadata") or {}
                retrieval_score = float(row.get("retrieval_score", row.get("fused_score", 0.0) or 0.0))
                meta_prior = float(metadata.get("metadata_prior", row.get("metadata_prior", 0.0) or 0.0))
                fused_score = 0.95 * retrieval_score + 0.05 * meta_prior
                row["stage2_model"] = self.stage2_info.name
                row["stage2_source"] = self.stage2_info.source
                row["stage2_backend"] = self.stage2_info.backend
                row["stage2_available"] = self.stage2_info.available
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
            return self._apply_doc_first_rank(retrieval_only_ranked)

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
                out = self._with_stage2_metadata(row)
                out["stage2_model"] = self.stage2_info.name
                out["stage2_source"] = self.stage2_info.source
                out["stage2_backend"] = self.stage2_info.backend
                out["stage2_available"] = self.stage2_info.available
                out["stage2_raw"] = 0.0
                out["stage2_score"] = 0.0
                out["fused_score"] = float(final_score)
                ranked_one_stage.append(out)

            for row in remainder:
                out = self._with_stage2_metadata(row)
                metadata = out.get("metadata") or {}
                retrieval_score = float(out.get("retrieval_score", out.get("fused_score", 0.0) or 0.0))
                meta_prior = float(metadata.get("metadata_prior", out.get("metadata_prior", 0.0) or 0.0))
                out["stage2_model"] = self.stage2_info.name
                out["stage2_source"] = self.stage2_info.source
                out["stage2_backend"] = self.stage2_info.backend
                out["stage2_available"] = self.stage2_info.available
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
            return self._apply_doc_first_rank(ranked_one_stage)

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
            out = self._with_stage2_metadata(row)
            out["stage2_model"] = self.stage2_info.name
            out["stage2_source"] = self.stage2_info.source
            out["stage2_backend"] = self.stage2_info.backend
            out["stage2_available"] = self.stage2_info.available
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
            out = self._with_stage2_metadata(row)
            out["stage2_model"] = self.stage2_info.name
            out["stage2_source"] = self.stage2_info.source
            out["stage2_backend"] = self.stage2_info.backend
            out["stage2_available"] = self.stage2_info.available
            out["stage2_raw"] = 0.0
            out["stage2_score"] = 0.0
            out["fused_score"] = float(fallback_score)
            ranked.append(out)

        for row in remainder:
            out = self._with_stage2_metadata(row)
            out["stage2_model"] = self.stage2_info.name
            out["stage2_source"] = self.stage2_info.source
            out["stage2_backend"] = self.stage2_info.backend
            out["stage2_available"] = self.stage2_info.available
            out["stage2_raw"] = 0.0
            out["stage2_score"] = 0.0
            out["fused_score"] = float(out.get("retrieval_score", out.get("fused_score", 0.0) or 0.0))
            ranked.append(out)

        ranked.sort(key=lambda row: (row.get("fused_score", 0.0), str(row.get("id", ""))), reverse=True)
        return self._apply_doc_first_rank(ranked)

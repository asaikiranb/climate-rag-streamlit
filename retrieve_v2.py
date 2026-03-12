"""Improved hybrid retriever with analyzer upgrades, sparse modes, and metadata priors."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

try:
    import chromadb
except Exception:  # pragma: no cover
    chromadb = None

try:
    from qdrant_client import QdrantClient, models
except Exception:  # pragma: no cover
    QdrantClient = None
    models = None

try:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
except Exception:  # pragma: no cover
    torch = None
    AutoModelForMaskedLM = None
    AutoTokenizer = None

from config import (
    CHROMA_COLLECTION,
    CHROMA_PATH,
    DENSE_FUSION_WEIGHT,
    EARLY_PAGE_PRIOR,
    INGEST_EMBEDDING_MODEL,
    MODEL_DEVICE,
    METADATA_FILENAME_WEIGHT,
    METADATA_PRIOR_WEIGHT,
    METADATA_SECTION_WEIGHT,
    METADATA_TITLE_WEIGHT,
    QDRANT_COLLECTION,
    QDRANT_PATH,
    RETRIEVAL_CANDIDATE_K,
    SPARSE_FUSION_WEIGHT,
    SPARSE_MODE,
    SPLADE_MAX_LENGTH,
    SPLADE_MAX_TERMS,
    SPLADE_MODEL,
    VECTOR_DB_BACKEND,
)
from hf_local import resolve_local_snapshot

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_/][a-z0-9]+)?")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "to",
    "with",
    "without",
}

TEMPORAL_QUERY_TERMS = {
    "when",
    "year",
    "date",
    "timeline",
    "ratified",
    "ratify",
    "since",
}

PROCEDURAL_QUERY_TERMS = {
    "how",
    "step",
    "steps",
    "procedure",
    "process",
    "installation",
    "install",
    "repair",
    "servicing",
    "service",
}

PROCEDURAL_SECTION_TERMS = {
    "step",
    "steps",
    "procedure",
    "process",
    "installation",
    "install",
    "service",
    "servicing",
    "maintenance",
}

YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")


def _runtime_device() -> str:
    requested = (MODEL_DEVICE or "cpu").strip().lower()
    if torch is None:
        return "cpu"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if requested == "mps" and mps_backend and mps_backend.is_available():
        return "mps"
    return "cpu"


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    cleaned = [0.0 if not math.isfinite(value) else float(value) for value in values]
    low = min(cleaned)
    high = max(cleaned)
    if math.isclose(low, high):
        return [1.0 for _ in cleaned]
    return [(value - low) / (high - low) for value in cleaned]


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    document: str
    metadata: dict[str, Any]
    dense_raw: float
    sparse_raw: float
    metadata_prior: float


class Analyzer:
    """Tokenizer + stemmer based analyzer."""

    def __init__(self) -> None:
        self.stemmer = PorterStemmer()

    def tokens(self, text: str) -> list[str]:
        words = TOKEN_PATTERN.findall((text or "").lower())
        out: list[str] = []
        for word in words:
            if word in STOPWORDS or len(word) < 2:
                continue
            out.append(self.stemmer.stem(word))
        return out


class SpladeSparseEncoder:
    """Optional SPLADE-like sparse encoder for query/document scoring."""

    def __init__(self, model_name: str = SPLADE_MODEL, max_terms: int = SPLADE_MAX_TERMS, max_length: int = SPLADE_MAX_LENGTH) -> None:
        self.model_name = model_name
        self.max_terms = max(8, int(max_terms))
        self.max_length = max(32, int(max_length))
        self.device = _runtime_device()
        self.available = False
        self.error: str | None = None
        self.tokenizer = None
        self.model = None

        if torch is None or AutoTokenizer is None or AutoModelForMaskedLM is None:
            self.error = "torch/transformers unavailable"
            return

        token = os.getenv("HF_TOKEN")
        local_snapshot = resolve_local_snapshot(self.model_name)
        try:
            model_ref = local_snapshot or self.model_name
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_ref,
                token=token,
                local_files_only=bool(local_snapshot),
            )
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_ref,
                token=token,
                local_files_only=bool(local_snapshot),
            )
            self.model.to(self.device)
            self.model.eval()
            self.available = True
        except Exception as exc:  # pragma: no cover
            self.error = str(exc)
            self.tokenizer = None
            self.model = None

    def encode(self, text: str) -> dict[int, float]:
        if not self.available or self.tokenizer is None or self.model is None or torch is None:
            return {}
        if not text.strip():
            return {}

        with torch.no_grad():
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            logits = self.model(**encoded).logits
            weights = torch.log1p(torch.relu(logits))
            mask = encoded["attention_mask"].unsqueeze(-1)
            masked = weights * mask
            pooled = torch.max(masked, dim=1).values.squeeze(0)
            if pooled.numel() == 0:
                return {}

            top_k = min(self.max_terms, int(pooled.numel()))
            values, indices = torch.topk(pooled, k=top_k)

        sparse: dict[int, float] = {}
        for index, value in zip(indices.tolist(), values.tolist()):
            score = float(value)
            if score <= 0.0 or not math.isfinite(score):
                continue
            sparse[int(index)] = score
        return sparse


class HybridRetrieverV2:
    """Dense + sparse fusion retriever with metadata priors and backend adapter support."""

    def __init__(
        self,
        backend: str = VECTOR_DB_BACKEND,
        embedding_model: str = INGEST_EMBEDDING_MODEL,
        sparse_mode: str = SPARSE_MODE,
        sparse_weight_override: float | None = None,
        chroma_path: str = CHROMA_PATH,
        chroma_collection: str = CHROMA_COLLECTION,
        qdrant_path: str = QDRANT_PATH,
        qdrant_collection: str = QDRANT_COLLECTION,
    ) -> None:
        self.backend = backend.strip().lower()
        if self.backend not in {"chroma", "qdrant"}:
            raise ValueError(f"Unsupported backend: {backend}")

        self.embedding_model = embedding_model
        self.device = _runtime_device()
        local_snapshot = resolve_local_snapshot(self.embedding_model)
        token = os.getenv("HF_TOKEN")
        if local_snapshot:
            try:
                self.encoder = SentenceTransformer(
                    local_snapshot,
                    local_files_only=True,
                    token=token,
                    device=self.device,
                )
            except Exception:
                self.encoder = SentenceTransformer(self.embedding_model, token=token, device=self.device)
        else:
            self.encoder = SentenceTransformer(self.embedding_model, token=token, device=self.device)

        self.sparse_mode = (sparse_mode or "none").strip().lower()
        if self.sparse_mode not in {"none", "bm42", "splade"}:
            raise ValueError(f"Unsupported sparse_mode: {sparse_mode}")

        self.sparse_weight = float(SPARSE_FUSION_WEIGHT if sparse_weight_override is None else sparse_weight_override)
        self.analyzer = Analyzer()
        self.bm25: BM25Okapi | None = None
        self.corpus_documents: list[str] = []
        self.corpus_metadata: list[dict[str, Any]] = []
        self.corpus_ids: list[str] = []
        self.tokenized_documents: list[list[str]] = []
        self.splade_documents: list[dict[int, float]] = []

        self.splade_encoder: SpladeSparseEncoder | None = None
        if self.sparse_mode == "splade":
            self.splade_encoder = SpladeSparseEncoder()
            if not self.splade_encoder.available:
                reason = self.splade_encoder.error or "unknown"
                raise RuntimeError(f"SPLADE unavailable: {reason}")

        self.chroma_collection = None
        self.qdrant_client = None
        self.qdrant_collection = qdrant_collection

        if self.backend == "chroma":
            if chromadb is None:
                raise ImportError("chromadb is required for backend='chroma'")
            db = chromadb.PersistentClient(path=chroma_path)
            self.chroma_collection = db.get_collection(name=chroma_collection)
        else:
            if QdrantClient is None or models is None:
                raise ImportError("qdrant-client is required for backend='qdrant'")
            self.qdrant_client = QdrantClient(path=qdrant_path)

    def _load_documents(self, filename_filter: str | None = None) -> None:
        docs: list[str] = []
        metas: list[dict[str, Any]] = []
        ids: list[str] = []

        if self.backend == "chroma":
            assert self.chroma_collection is not None
            res = self.chroma_collection.get(include=["documents", "metadatas"])
            docs = [str(document or "") for document in res.get("documents", [])]
            metas = [dict(meta or {}) for meta in res.get("metadatas", [])]
            ids = [str(value) for value in res.get("ids", [])]
        else:
            assert self.qdrant_client is not None
            offset = None
            while True:
                points, offset = self.qdrant_client.scroll(
                    collection_name=self.qdrant_collection,
                    scroll_filter=None,
                    with_payload=True,
                    with_vectors=False,
                    limit=1024,
                    offset=offset,
                )
                for point in points:
                    payload = dict(point.payload or {})
                    docs.append(str(payload.get("document") or ""))
                    metas.append(payload)
                    ids.append(str(payload.get("chunk_id") or point.id))
                if offset is None:
                    break

        if filename_filter:
            needle = filename_filter.lower()
            keep = [needle in str(meta.get("filename", "")).lower() for meta in metas]
            docs = [doc for idx, doc in enumerate(docs) if keep[idx]]
            metas = [meta for idx, meta in enumerate(metas) if keep[idx]]
            ids = [value for idx, value in enumerate(ids) if keep[idx]]

        self.corpus_documents = docs
        self.corpus_metadata = metas
        self.corpus_ids = ids

        self.tokenized_documents = [self.analyzer.tokens(document) for document in docs]
        self.bm25 = BM25Okapi(self.tokenized_documents) if self.tokenized_documents else None

        if self.sparse_mode == "splade" and self.splade_encoder is not None:
            self.splade_documents = [self.splade_encoder.encode(document) for document in docs]
        else:
            self.splade_documents = []

    def _dense_search(self, query: str, top_k: int, filename_filter: str | None = None) -> list[dict[str, Any]]:
        query_vector = self.encoder.encode(query, show_progress_bar=False).tolist()
        if self.backend == "chroma":
            assert self.chroma_collection is not None
            where = None
            if filename_filter:
                where = {"filename": {"$contains": filename_filter.lower()}}
            res = self.chroma_collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            ids = res.get("ids", [[]])[0]
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            distances = res.get("distances", [[]])[0]
            out = []
            for idx in range(len(ids)):
                dist = float(distances[idx])
                score = 1.0 / (1.0 + max(dist, 0.0))
                out.append(
                    {
                        "id": str(ids[idx]),
                        "document": str(docs[idx]),
                        "metadata": dict(metas[idx] or {}),
                        "dense_raw": score,
                    }
                )
            return out

        assert self.qdrant_client is not None
        query_filter = None
        if filename_filter:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="filename",
                        match=models.MatchText(text=filename_filter.lower()),
                    )
                ]
            )

        if hasattr(self.qdrant_client, "query_points"):
            response = self.qdrant_client.query_points(
                collection_name=self.qdrant_collection,
                query=query_vector,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
                limit=top_k,
            )
            points = list(getattr(response, "points", response))
        else:
            points = self.qdrant_client.search(
                collection_name=self.qdrant_collection,
                query_vector=query_vector,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
                limit=top_k,
            )
        return [
            {
                "id": str(point.payload.get("chunk_id") or point.id),
                "document": str((point.payload or {}).get("document") or ""),
                "metadata": dict(point.payload or {}),
                "dense_raw": float(point.score),
            }
            for point in points
        ]

    def _top_sparse_scores(self, scores: np.ndarray, top_k: int) -> dict[str, float]:
        if len(scores) == 0:
            return {}
        out: dict[str, float] = {}
        top_indices = np.argsort(scores)[::-1][:top_k]
        for index in top_indices:
            score = float(scores[index])
            if score <= 0.0 or not math.isfinite(score):
                continue
            out[self.corpus_ids[int(index)]] = score
        return out

    def _sparse_scores(self, query: str, query_tokens: list[str], top_k: int) -> dict[str, float]:
        if self.bm25 is None:
            return {}

        if self.sparse_mode == "none":
            bm25_scores = self.bm25.get_scores(query_tokens)
            return self._top_sparse_scores(bm25_scores, top_k)

        if self.sparse_mode == "bm42":
            bm25_scores = self.bm25.get_scores(query_tokens)
            query_token_set = set(query_tokens)
            combined = np.zeros(len(self.corpus_ids), dtype=float)
            for idx, bm25 in enumerate(bm25_scores):
                token_set = set(self.tokenized_documents[idx])
                overlap = (len(query_token_set.intersection(token_set)) / len(query_token_set)) if query_token_set else 0.0
                combined[idx] = 0.70 * float(bm25) + 0.30 * float(overlap) * (1.0 + max(float(bm25), 0.0))
            return self._top_sparse_scores(combined, top_k)

        if self.sparse_mode == "splade":
            if self.splade_encoder is None:
                return {}
            query_sparse = self.splade_encoder.encode(query)
            if not query_sparse:
                return {}
            scores = np.zeros(len(self.corpus_ids), dtype=float)
            for idx, document_sparse in enumerate(self.splade_documents):
                if not document_sparse:
                    continue
                score = 0.0
                for token_id, weight in query_sparse.items():
                    score += weight * float(document_sparse.get(token_id, 0.0))
                scores[idx] = score
            return self._top_sparse_scores(scores, top_k)

        return {}

    def _overlap(self, query_tokens: set[str], value: str) -> float:
        field_tokens = set(self.analyzer.tokens(value))
        if not query_tokens or not field_tokens:
            return 0.0
        return len(query_tokens.intersection(field_tokens)) / len(query_tokens)

    def _metadata_prior(
        self,
        query_tokens: set[str],
        query_text: str,
        document: str,
        metadata: dict[str, Any],
    ) -> float:
        title = str(metadata.get("title") or "")
        section = str(metadata.get("section_title") or "")
        filename = str(metadata.get("filename") or "")
        page = metadata.get("page_number")

        prior = 0.0
        prior += METADATA_TITLE_WEIGHT * self._overlap(query_tokens, title)
        prior += METADATA_SECTION_WEIGHT * self._overlap(query_tokens, section)
        prior += METADATA_FILENAME_WEIGHT * self._overlap(query_tokens, filename)

        if isinstance(page, int) and page <= 3:
            prior += EARLY_PAGE_PRIOR
        elif isinstance(page, str) and page.isdigit() and int(page) <= 3:
            prior += EARLY_PAGE_PRIOR

        lowered_query = query_text.lower()
        is_temporal = any(term in lowered_query for term in TEMPORAL_QUERY_TERMS)
        if is_temporal:
            temporal_text = " ".join([title, section, filename, document[:500]])
            if YEAR_PATTERN.search(temporal_text):
                prior += 0.05

        is_procedural = any(term in lowered_query for term in PROCEDURAL_QUERY_TERMS)
        if is_procedural:
            lowered_section = f"{title} {section}".lower()
            if any(term in lowered_section for term in PROCEDURAL_SECTION_TERMS):
                prior += 0.05

        return min(1.0, max(0.0, prior))

    def search(self, query: str, top_k: int = RETRIEVAL_CANDIDATE_K, brand: str | None = None) -> list[dict[str, Any]]:
        if self.bm25 is None or brand:
            self._load_documents(filename_filter=brand)

        if not self.corpus_documents:
            return []

        dense_hits = self._dense_search(query=query, top_k=top_k, filename_filter=brand)
        query_tokens = self.analyzer.tokens(query)
        sparse_rank_map = self._sparse_scores(query=query, query_tokens=query_tokens, top_k=top_k)

        dense_map: dict[str, dict[str, Any]] = {}
        for hit in dense_hits:
            dense_map[str(hit["id"])] = hit

        dense_ids = list(dense_map.keys())
        sparse_ids = [candidate_id for candidate_id in sparse_rank_map.keys() if candidate_id not in dense_map]
        candidate_ids = dense_ids + sparse_ids
        if not candidate_ids:
            return []

        candidates: list[Candidate] = []
        query_token_set = set(query_tokens)
        corpus_id_to_idx = {value: idx for idx, value in enumerate(self.corpus_ids)}

        for candidate_id in candidate_ids:
            dense_hit = dense_map.get(candidate_id)
            dense_raw = float((dense_hit or {}).get("dense_raw", 0.0))
            sparse_raw = float(sparse_rank_map.get(candidate_id, 0.0))

            if not math.isfinite(dense_raw):
                dense_raw = 0.0
            if not math.isfinite(sparse_raw):
                sparse_raw = 0.0

            if dense_hit:
                document = str(dense_hit.get("document") or "")
                metadata = dict(dense_hit.get("metadata") or {})
            else:
                index = corpus_id_to_idx.get(candidate_id)
                if index is None:
                    continue
                document = self.corpus_documents[index]
                metadata = dict(self.corpus_metadata[index])

            prior = self._metadata_prior(
                query_tokens=query_token_set,
                query_text=query,
                document=document,
                metadata=metadata,
            )
            candidates.append(
                Candidate(
                    candidate_id=candidate_id,
                    document=document,
                    metadata=metadata,
                    dense_raw=dense_raw,
                    sparse_raw=sparse_raw,
                    metadata_prior=prior,
                )
            )

        dense_norm = minmax([candidate.dense_raw for candidate in candidates])
        sparse_norm = minmax([candidate.sparse_raw for candidate in candidates])

        results: list[dict[str, Any]] = []
        for idx, candidate in enumerate(candidates):
            retrieval_score = (
                DENSE_FUSION_WEIGHT * dense_norm[idx]
                + self.sparse_weight * sparse_norm[idx]
                + METADATA_PRIOR_WEIGHT * candidate.metadata_prior
            )
            if not math.isfinite(retrieval_score):
                retrieval_score = 0.0

            enriched_meta = dict(candidate.metadata)
            enriched_meta["metadata_prior"] = candidate.metadata_prior
            enriched_meta["dense_norm"] = dense_norm[idx]
            enriched_meta["sparse_norm"] = sparse_norm[idx]
            enriched_meta["sparse_mode"] = self.sparse_mode

            results.append(
                {
                    "id": candidate.candidate_id,
                    "document": candidate.document,
                    "metadata": enriched_meta,
                    "fused_score": float(retrieval_score),
                    "retrieval_score": float(retrieval_score),
                }
            )

        results.sort(key=lambda row: (row["fused_score"], row["id"]), reverse=True)
        return results[:top_k]

    def close(self) -> None:
        if self.qdrant_client is not None:
            try:
                self.qdrant_client.close()
            except Exception:
                pass

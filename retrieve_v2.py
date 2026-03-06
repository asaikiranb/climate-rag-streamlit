"""Improved hybrid retriever with analyzer upgrades and metadata priors."""

from __future__ import annotations

import math
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

from config import (
    CHROMA_COLLECTION,
    CHROMA_PATH,
    DENSE_FUSION_WEIGHT,
    EARLY_PAGE_PRIOR,
    INGEST_EMBEDDING_MODEL,
    METADATA_FILENAME_WEIGHT,
    METADATA_PRIOR_WEIGHT,
    METADATA_SECTION_WEIGHT,
    METADATA_TITLE_WEIGHT,
    QDRANT_COLLECTION,
    QDRANT_PATH,
    RETRIEVAL_CANDIDATE_K,
    SPARSE_FUSION_WEIGHT,
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


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    document: str
    metadata: dict[str, Any]
    dense_raw: float
    sparse_raw: float
    metadata_prior: float


class Analyzer:
    """Tokenizer + stemmer based BM25 analyzer."""

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


class HybridRetrieverV2:
    """Dense+BM25 fusion retriever with metadata priors and backend adapter support."""

    def __init__(
        self,
        backend: str = VECTOR_DB_BACKEND,
        embedding_model: str = INGEST_EMBEDDING_MODEL,
        chroma_path: str = CHROMA_PATH,
        chroma_collection: str = CHROMA_COLLECTION,
        qdrant_path: str = QDRANT_PATH,
        qdrant_collection: str = QDRANT_COLLECTION,
    ) -> None:
        self.backend = backend.strip().lower()
        if self.backend not in {"chroma", "qdrant"}:
            raise ValueError(f"Unsupported backend: {backend}")

        self.embedding_model = embedding_model
        local_snapshot = resolve_local_snapshot(self.embedding_model)
        if local_snapshot:
            try:
                self.encoder = SentenceTransformer(local_snapshot, local_files_only=True)
            except Exception:
                self.encoder = SentenceTransformer(self.embedding_model)
        else:
            self.encoder = SentenceTransformer(self.embedding_model)
        self.analyzer = Analyzer()
        self.bm25: BM25Okapi | None = None
        self.corpus_documents: list[str] = []
        self.corpus_metadata: list[dict[str, Any]] = []
        self.corpus_ids: list[str] = []

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

        tokenized = [self.analyzer.tokens(document) for document in docs]
        self.bm25 = BM25Okapi(tokenized)

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

    def _overlap(self, query_tokens: set[str], value: str) -> float:
        field_tokens = set(self.analyzer.tokens(value))
        if not query_tokens or not field_tokens:
            return 0.0
        return len(query_tokens.intersection(field_tokens)) / len(query_tokens)

    def _metadata_prior(self, query_tokens: set[str], metadata: dict[str, Any]) -> float:
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

        return min(1.0, max(0.0, prior))

    def search(self, query: str, top_k: int = RETRIEVAL_CANDIDATE_K, brand: str | None = None) -> list[dict[str, Any]]:
        if self.bm25 is None or brand:
            self._load_documents(filename_filter=brand)

        if not self.corpus_documents:
            return []

        dense_hits = self._dense_search(query=query, top_k=top_k, filename_filter=brand)
        query_tokens = self.analyzer.tokens(query)
        sparse_scores = self.bm25.get_scores(query_tokens) if self.bm25 is not None else np.array([])

        sparse_indices = np.argsort(sparse_scores)[::-1][:top_k] if len(sparse_scores) else []
        sparse_rank_map: dict[str, float] = {}
        for idx in sparse_indices:
            score = float(sparse_scores[idx])
            if score <= 0:
                continue
            sparse_rank_map[self.corpus_ids[int(idx)]] = score

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

            prior = self._metadata_prior(query_token_set, metadata)
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
                + SPARSE_FUSION_WEIGHT * sparse_norm[idx]
                + METADATA_PRIOR_WEIGHT * candidate.metadata_prior
            )
            if not math.isfinite(retrieval_score):
                retrieval_score = 0.0
            enriched_meta = dict(candidate.metadata)
            enriched_meta["metadata_prior"] = candidate.metadata_prior
            enriched_meta["dense_norm"] = dense_norm[idx]
            enriched_meta["sparse_norm"] = sparse_norm[idx]
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

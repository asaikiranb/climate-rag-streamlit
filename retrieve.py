"""Hybrid retriever combining semantic vector search and BM25 keyword matching."""

import os
from typing import List, Dict
import numpy as np
from rank_bm25 import BM25Okapi
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

from config import (
    CHROMA_PATH, 
    CHROMA_COLLECTION, 
    RETRIEVAL_CANDIDATE_K, 
    RRF_K,
    VECTOR_WEIGHT,
    BM25_WEIGHT,
)

class HybridRetriever:
    """Combines semantic vector search with keyword-based BM25 search."""

    def __init__(self):
        # Use local SentenceTransformer only (no HuggingFace API). Same model as ingestion.
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-m3"
        )
        self.db = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.db.get_collection(name=CHROMA_COLLECTION)
        
        self.bm25 = None
        self.documents = []
        self.metadata = []
        self.ids = []

    def _initialize_bm25(self, filter_brand: str = None):
        """Build a BM25 index from document collection."""
        res = self.collection.get(include=["documents", "metadatas"])
        docs, metas, ids = res["documents"], res["metadatas"], res["ids"]

        if filter_brand:
            mask = [filter_brand.lower() in m.get("filename", "").lower() for m in metas]
            docs = [d for i, d in enumerate(docs) if mask[i]]
            metas = [m for i, m in enumerate(metas) if mask[i]]
            ids = [id for i, id in enumerate(ids) if mask[i]]

        self.documents, self.metadata, self.ids = docs, metas, ids
        tokenized = [d.lower().split() for d in docs]
        self.bm25 = BM25Okapi(tokenized)

    def _normalize_query(self, query: str) -> str:
        """Light normalization to improve retrieval match (no extra latency)."""
        if not query or not query.strip():
            return query
        return " ".join(query.strip().lower().split())

    def search(self, query: str, top_k: int = RETRIEVAL_CANDIDATE_K, brand: str = None) -> List[Dict]:
        """Execute hybrid search using semantic and keyword strategies."""
        if not self.bm25 or brand:
            self._initialize_bm25(brand)

        # 1. Semantic Vector Search
        emb = self.ef([query])[0]
        v_res = self.collection.query(
            query_embeddings=[emb],
            n_results=top_k,
            where={"filename": {"$contains": brand.lower()}} if brand else None,
            include=["documents", "metadatas", "distances"]
        )
        
        vector_hits = [
            {"id": v_res["ids"][0][i], "doc": v_res["documents"][0][i], "meta": v_res["metadatas"][0][i], "rank": i+1}
            for i in range(len(v_res["ids"][0]))
        ]

        # 2. Keyword BM25 Search
        scores = self.bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[::-1][:top_k]
        bm25_hits = [
            {"id": self.ids[i], "doc": self.documents[i], "meta": self.metadata[i], "rank": j+1}
            for j, i in enumerate(top_idx) if scores[i] > 0
        ]

        # 3. Reciprocal Rank Fusion (RRF)
        return self._fuse(vector_hits, bm25_hits)

    def _fuse(self, vector_hits: List[Dict], bm25_hits: List[Dict]) -> List[Dict]:
        """Combine results using weighted reciprocal rank fusion."""
        scores = {}
        for hit in vector_hits:
            scores[hit["id"]] = {"score": VECTOR_WEIGHT / (RRF_K + hit["rank"]), "doc": hit["doc"], "meta": hit["meta"]}

        for hit in bm25_hits:
            rrf = BM25_WEIGHT / (RRF_K + hit["rank"])
            if hit["id"] in scores:
                scores[hit["id"]]["score"] += rrf
            else:
                scores[hit["id"]] = {"score": rrf, "doc": hit["doc"], "meta": hit["meta"]}

        sorted_hits = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
        return [
            {"id": k, "document": v["doc"], "metadata": v["meta"], "rrf_score": v["score"]}
            for k, v in sorted_hits
        ]

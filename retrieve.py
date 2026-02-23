"""Hybrid retrieval: vector search + BM25 + RRF + optional cross-encoder reranking."""

import os
import time as _time
from typing import List, Dict

import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb

from config import RETRIEVAL_TOP_K, RETRIEVAL_CANDIDATE_K

load_dotenv()


class HybridRetriever:
    def __init__(self):
        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2", token=os.environ.get("HF_TOKEN")
        )

        chroma_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)

        self.collection_name = os.getenv("CHROMA_COLLECTION_NAME", "hvac_documents")
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            print(f"Connected to collection: {self.collection_name}")
        except Exception as e:
            raise Exception(f"Could not connect to collection '{self.collection_name}': {e}")

        self.bm25 = None
        self.all_documents = None
        self.all_metadatas = None
        self.all_ids = None

    def _load_bm25_index(self, brand_filter: str = None):
        """Build BM25 index from ChromaDB documents, with optional filename filter."""
        print("Loading documents for BM25 indexing...")

        results = self.collection.get(include=["documents", "metadatas"])
        documents = results["documents"]
        metadatas = results["metadatas"]
        ids = results["ids"]

        if brand_filter:
            filtered = [
                (doc, meta, doc_id)
                for doc, meta, doc_id in zip(documents, metadatas, ids)
                if brand_filter.lower() in meta.get("filename", "").lower()
            ]
            if filtered:
                documents, metadatas, ids = zip(*filtered)
                documents, metadatas, ids = list(documents), list(metadatas), list(ids)
            else:
                print(f"Warning: no documents found for brand '{brand_filter}'")
                documents, metadatas, ids = [], [], []

        self.all_documents = documents
        self.all_metadatas = metadatas
        self.all_ids = ids

        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        print(f"BM25 index built with {len(documents)} documents")

    def vector_search(self, query: str, top_k: int = RETRIEVAL_CANDIDATE_K, brand_filter: str = None) -> List[Dict]:
        """Semantic search via ChromaDB embeddings."""
        query_embedding = self.embedding_model.encode(query).tolist()

        where_filter = None
        if brand_filter:
            where_filter = {"filename": {"$contains": brand_filter.lower()}}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        return [
            {
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],
                "method": "vector",
            }
            for i in range(len(results["documents"][0]))
        ]

    def bm25_search(self, query: str, top_k: int = RETRIEVAL_CANDIDATE_K) -> List[Dict]:
        """Keyword search via BM25."""
        if not self.bm25 or not self.all_documents:
            raise Exception("BM25 index not loaded.")

        scores = self.bm25.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "id": self.all_ids[idx],
                "document": self.all_documents[idx],
                "metadata": self.all_metadatas[idx],
                "score": float(scores[idx]),
                "method": "bm25",
            }
            for idx in top_indices
            if scores[idx] > 0
        ]

    def reciprocal_rank_fusion(
        self,
        vector_results: List[Dict],
        bm25_results: List[Dict],
        k: int = 60,
    ) -> List[Dict]:
        """Merge vector and BM25 results. score = sum(1 / (k + rank)) across both lists."""
        rrf_scores: Dict[str, Dict] = {}

        for rank, result in enumerate(vector_results, start=1):
            doc_id = result["id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {"score": 0, "document": result["document"], "metadata": result["metadata"]}
            rrf_scores[doc_id]["score"] += 1 / (k + rank)

        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result["id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {"score": 0, "document": result["document"], "metadata": result["metadata"]}
            rrf_scores[doc_id]["score"] += 1 / (k + rank)

        return [
            {"id": doc_id, "document": data["document"], "metadata": data["metadata"], "rrf_score": data["score"]}
            for doc_id, data in sorted(rrf_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        ]

    def hybrid_search(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        brand_filter: str = None,
        reranker=None,
    ) -> List[Dict]:
        """
        Run vector + BM25 search, merge with RRF, optionally rerank with cross-encoder.
        Pass a loaded ContextualReranker to reranker to enable reranking.
        """
        if self.bm25 is None or brand_filter:
            self._load_bm25_index(brand_filter)

        if not self.all_documents:
            return []

        vector_results = self.vector_search(query, top_k=RETRIEVAL_CANDIDATE_K, brand_filter=brand_filter)
        bm25_results = self.bm25_search(query, top_k=RETRIEVAL_CANDIDATE_K)
        candidates = self.reciprocal_rank_fusion(vector_results, bm25_results)

        if reranker is not None:
            from rerank import rerank
            candidates = rerank(query, candidates, reranker)

        return candidates[:top_k]

    def hybrid_search_timed(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        brand_filter: str = None,
        reranker=None,
    ) -> Dict:
        """Same as hybrid_search but returns timing data alongside results."""
        if self.bm25 is None or brand_filter:
            self._load_bm25_index(brand_filter)

        if not self.all_documents:
            return {"results": [], "timings": {"embed_ms": 0, "search_ms": 0, "rerank_ms": 0}}

        embed_start = _time.perf_counter()
        query_embedding = self.embedding_model.encode(query).tolist()
        embed_ms = (_time.perf_counter() - embed_start) * 1000

        where_filter = {"filename": {"$contains": brand_filter.lower()}} if brand_filter else None

        search_start = _time.perf_counter()
        vs_raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=RETRIEVAL_CANDIDATE_K,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
        vector_results = [
            {
                "id": vs_raw["ids"][0][i],
                "document": vs_raw["documents"][0][i],
                "metadata": vs_raw["metadatas"][0][i],
                "score": 1 - vs_raw["distances"][0][i],
                "method": "vector",
            }
            for i in range(len(vs_raw["documents"][0]))
        ]
        bm25_results = self.bm25_search(query, top_k=RETRIEVAL_CANDIDATE_K)
        search_ms = (_time.perf_counter() - search_start) * 1000

        rerank_start = _time.perf_counter()
        candidates = self.reciprocal_rank_fusion(vector_results, bm25_results)
        if reranker is not None:
            from rerank import rerank
            candidates = rerank(query, candidates, reranker)
        final_results = candidates[:top_k]
        rerank_ms = (_time.perf_counter() - rerank_start) * 1000

        return {
            "results": final_results,
            "timings": {
                "embed_ms": round(embed_ms, 2),
                "search_ms": round(search_ms, 2),
                "rerank_ms": round(rerank_ms, 2),
            },
        }

    def get_available_brands(self) -> List[str]:
        """Return sorted unique brand names derived from filenames in the collection."""
        results = self.collection.get(include=["metadatas"])
        filenames = {meta["filename"] for meta in results["metadatas"]}
        brands = {fn.replace(".pdf", "").split("_")[0].split()[0] for fn in filenames}
        return sorted(brands)


if __name__ == "__main__":
    retriever = HybridRetriever()
    query = "How do I troubleshoot a refrigerant leak?"
    results = retriever.hybrid_search(query, top_k=5)
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r['metadata']['filename']} p{r['metadata']['page_number']}: {r['document'][:120]}...")

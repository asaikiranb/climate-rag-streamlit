"""Hybrid retrieval: vector search + BM25 + RRF + optional cross-encoder reranking.

Retrieval strategy:
  1. Multi-query expansion — generate 2 alternative phrasings per query via Groq LLM,
     embed all 3, and fuse their vector results via RRF before merging with BM25. This
     significantly improves recall for source-specific questions ("according to the HPMP
     poster...") where a single embedding misses semantically-close-but-wrong documents.
  2. Embedding: BAAI/bge-m3 (1024-dim, 8192-token context) via HF Inference API when
     HF_TOKEN is available; local SentenceTransformer fallback otherwise.
  3. Hybrid BM25 + vector RRF — keyword and semantic signals are complementary.
  4. BM25 with NLTK word_tokenize for proper punctuation/compound-term handling.
  5. Optional reranking — Contextual AI ctxl-rerank-v2 for final ordering.
"""

import os
import time as _time
from typing import List, Dict, Optional

import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import chromadb

from config import RETRIEVAL_TOP_K, RETRIEVAL_CANDIDATE_K

load_dotenv()

import nltk
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import word_tokenize


def _expand_query(query: str) -> List[str]:
    """
    Generate alternative phrasings for a query to improve recall.

    Uses Groq (llama-3.1-8b-instant) when GROQ_API_KEY is set — LLM-based expansion
    produces true semantic paraphrases with different vocabulary. Falls back to a single
    query when no API key is available.
    """
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        return [query]

    from groq import Groq
    client = Groq(api_key=groq_key)
    prompt = (
        "Generate 4 alternative phrasings of this question for document retrieval. "
        "Focus on semantic diversity — use different vocabulary, synonyms, and phrasing while preserving the original intent. "
        "Include the document or source name if mentioned in the question. "
        "Output only the 4 alternatives, one per line, no numbering, no explanations.\n\n"
        f"Question: {query}"
    )
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=250,
        )
        lines = [l.strip() for l in resp.choices[0].message.content.strip().split("\n") if l.strip()]
        seen = {query.lower().rstrip("?. ")}
        deduped = [query]
        for a in lines[:4]:
            key = a.lower().rstrip("?. ")
            if key not in seen and len(a) > 10:
                seen.add(key)
                deduped.append(a)
        return deduped[:5]
    except Exception:
        return [query]  # graceful fallback — never let expansion break retrieval


class HybridRetriever:
    def __init__(self):
        hf_token = os.environ.get("HF_TOKEN")
        self.embedding_model_id = "BAAI/bge-m3"

        # Always use local SentenceTransformer for bge-m3.
        # bge-m3 is NOT available on the HF free Inference API (returns 403),
        # so we download/cache the weights and run inference locally.
        # Passing token= allows authenticated downloads in CI without rate limits.
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer(self.embedding_model_id, token=hf_token or None)
        print(f"Embedding: {self.embedding_model_id} (local SentenceTransformer)")

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

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Embed query texts using bge-m3 (local SentenceTransformer).

        Prepends 'query: ' prefix to align with the document-context enriched
        embeddings stored in ChromaDB (which have 'Document: <title>\n\n' prefixed).
        """
        prefixed = ["query: " + t for t in texts]
        return self.embedding_model.encode(prefixed).tolist()

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

        tokenized = [word_tokenize(doc.lower()) for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        print(f"BM25 index built with {len(documents)} documents")

    def vector_search(
        self,
        query: str,
        top_k: int = RETRIEVAL_CANDIDATE_K,
        brand_filter: str = None,
        extra_queries: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Semantic search via ChromaDB embeddings.

        If extra_queries is provided, embeds all queries and fuses their results
        via RRF before returning — multi-query expansion for better recall.
        """
        all_queries = [query] + (extra_queries or [])
        embeddings = self._embed(all_queries)

        where_filter = None
        if brand_filter:
            where_filter = {"filename": {"$contains": brand_filter.lower()}}

        # Single-query fast path
        if len(embeddings) == 1:
            results = self.collection.query(
                query_embeddings=embeddings,
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

        # Multi-query: retrieve per-query, fuse with RRF
        rrf_scores: Dict[str, Dict] = {}
        rrf_k = 60
        for emb in embeddings:
            res = self.collection.query(
                query_embeddings=[emb],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
            for rank, (doc_id, doc, meta, dist) in enumerate(zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            ), start=1):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = {"score": 0.0, "document": doc, "metadata": meta,
                                          "best_sim": 0.0}
                rrf_scores[doc_id]["score"] += 1.0 / (rrf_k + rank)
                sim = 1 - dist
                if sim > rrf_scores[doc_id]["best_sim"]:
                    rrf_scores[doc_id]["best_sim"] = sim

        fused = sorted(rrf_scores.items(), key=lambda x: x[1]["score"], reverse=True)[:top_k]
        return [
            {
                "id": doc_id,
                "document": data["document"],
                "metadata": data["metadata"],
                "score": data["best_sim"],
                "method": "vector_multi",
            }
            for doc_id, data in fused
        ]

    def bm25_search(self, query: str, top_k: int = RETRIEVAL_CANDIDATE_K) -> List[Dict]:
        """Keyword search via BM25."""
        if not self.bm25 or not self.all_documents:
            raise Exception("BM25 index not loaded.")

        scores = self.bm25.get_scores(word_tokenize(query.lower()))
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
        Uses multi-query expansion on the vector side for improved recall.
        Pass a loaded ContextualReranker to reranker to enable reranking.
        """
        if self.bm25 is None or brand_filter:
            self._load_bm25_index(brand_filter)

        if not self.all_documents:
            return []

        expanded = _expand_query(query)
        extra = expanded[1:] if len(expanded) > 1 else None
        vector_results = self.vector_search(query, top_k=RETRIEVAL_CANDIDATE_K,
                                            brand_filter=brand_filter, extra_queries=extra)
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

        # Multi-query expansion via Groq LLM, then embed all queries
        embed_start = _time.perf_counter()
        expanded = _expand_query(query)
        extra_queries = expanded[1:] if len(expanded) > 1 else None
        embed_ms = (_time.perf_counter() - embed_start) * 1000

        where_filter = {"filename": {"$contains": brand_filter.lower()}} if brand_filter else None

        search_start = _time.perf_counter()
        vector_results = self.vector_search(
            query, top_k=RETRIEVAL_CANDIDATE_K,
            brand_filter=brand_filter, extra_queries=extra_queries
        )
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
    query = "What is the India Cooling Action Plan?"
    results = retriever.hybrid_search(query, top_k=5)
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r['metadata']['filename']} p{r['metadata']['page_number']}: {r['document'][:120]}...")

"""Cross-Encoder reranking module; model is configurable via config.RERANKER_MODEL."""

import os
from typing import List, Dict
from sentence_transformers import CrossEncoder

from config import RERANK_POOL_SIZE, RERANKER_WEIGHT, RETRIEVER_WEIGHT, RRF_K, RERANKER_MODEL
from config import USE_DIVERSITY_TOP_K, RETRIEVAL_TOP_K_FOR_DIVERSITY

class CrossEncoderReranker:
    """Reranks retrieved candidates using a Cross-Encoder and weighted Rank-Based Fusion."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or RERANKER_MODEL
        self.model = CrossEncoder(self.model_name)

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Optimize candidate ranking using a cross-encoder."""
        if not candidates:
            return candidates

        pool_size = min(RERANK_POOL_SIZE, len(candidates))
        pool = candidates[:pool_size]
        remainder = candidates[pool_size:]

        # 1. Score with Cross-Encoder
        docs = [c["document"] for c in pool]
        ce_scores = self.model.predict([(query, d) for d in docs])

        # 2. Assign CE Ranks
        ranked_pool = sorted(
            zip(pool, ce_scores), 
            key=lambda x: x[1], 
            reverse=True
        )

        # 3. Apply Weighted Rank Fusion
        for ce_rank, (candidate, _) in enumerate(ranked_pool, start=1):
            retriever_rank = candidates.index(candidate) + 1
            
            # Weighted reciprocal rank fusion
            fused_score = (
                RETRIEVER_WEIGHT * (1.0 / (RRF_K + retriever_rank)) + 
                RERANKER_WEIGHT * (1.0 / (RRF_K + ce_rank))
            )
            candidate["fused_score"] = fused_score

        pool.sort(key=lambda x: x["fused_score"], reverse=True)

        if USE_DIVERSITY_TOP_K and len(pool) >= RETRIEVAL_TOP_K_FOR_DIVERSITY:
            pool = self._diversify_top_k(pool, RETRIEVAL_TOP_K_FOR_DIVERSITY)

        return pool + remainder

    def _diversify_top_k(self, pool: List[Dict], k: int) -> List[Dict]:
        """Reorder so top-k are best chunk per document (raises recall@1 when gold is 2nd chunk of a doc)."""
        by_doc = {}
        for c in pool:
            fn = c.get("metadata", {}).get("filename", "")
            if fn not in by_doc or c["fused_score"] > by_doc[fn]["fused_score"]:
                by_doc[fn] = c
        best_per_doc = sorted(by_doc.values(), key=lambda x: x["fused_score"], reverse=True)
        top_diverse = best_per_doc[:k]
        seen_ids = {c["id"] for c in top_diverse}
        rest = [c for c in pool if c["id"] not in seen_ids]
        return top_diverse + rest

"""Cross-Encoder reranking module using BAAI/bge-reranker-v2-m3."""

from typing import List, Dict
from sentence_transformers import CrossEncoder

from config import RERANK_POOL_SIZE, RERANKER_WEIGHT, RETRIEVER_WEIGHT, RRF_K

class CrossEncoderReranker:
    """Reranks retrieved candidates using a Cross-Encoder and weighted Rank-Based Fusion."""

    def __init__(self):
        self.model = CrossEncoder("BAAI/bge-reranker-v2-m3")

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
        return pool + remainder

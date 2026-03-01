"""
Comprehensive RAG evaluation metrics suite.

Metrics computed:
  - Retrieval: recall@k, MRR@k, NDCG@k for k=1,3,5
  - Latency: mean/p50/p95/min/max

Usage:
    python run_contextual_eval.py --retrieval-only --output results.json
"""

import os
import json
import argparse
import time
import numpy as np
from datetime import datetime
from typing import List, Dict

from retrieve import HybridRetriever
from rerank import CrossEncoderReranker
from llm import GenerationClient
from eval.retrieval_metrics import compute_retrieval_metrics_at_k
from eval.metrics import compute_custom_metrics
from eval.generation_metrics_ollama import judge_generation

K_VALUES = [1, 3, 5]

class EvaluationEngine:
    """Orchestrates full RAG quality measurement (retrieval, generation, citations, judging)."""

    def __init__(self, use_reranker: bool = True):
        self.retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker() if use_reranker else None
        self.generator = GenerationClient()

    def run(self, dataset_path: str, output_path: str, limit: int = None):
        """Execute evaluation over the dataset."""
        dataset = self._load_jsonl(dataset_path)
        if limit:
            dataset = dataset[:limit]
        print(f"Loaded {len(dataset)} evaluation items.")

        results = []
        latencies = {"embed_ms": [], "search_ms": [], "rerank_ms": [], "generate_ms": []}

        for i, entry in enumerate(dataset, 1):
            question = entry["question"]
            gold = entry.get("gold_sources", [])
            difficulty = entry.get("metadata", {}).get("difficulty", "Unknown")
            print(f"[{i}/{len(dataset)}] Evaluating ({difficulty}): {question[:60]}...")

            # 1. Retrieval
            t0 = time.perf_counter()
            hits = self.retriever.search(question)
            latencies["search_ms"].append((time.perf_counter() - t0) * 1000)

            # 2. Reranking
            t0 = time.perf_counter()
            if self.reranker:
                hits = self.reranker.rerank(question, hits)
            latencies["rerank_ms"].append((time.perf_counter() - t0) * 1000)

            # 3. Generation
            t0 = time.perf_counter()
            answer = self.generator.generate(question, hits[:5])
            latencies["generate_ms"].append((time.perf_counter() - t0) * 1000)

            # 4. Metrics
            ret_met = compute_retrieval_metrics_at_k(
                [h["metadata"]["filename"] for h in hits[:5]], gold, k_values=K_VALUES
            )
            cit_met = compute_custom_metrics(answer, hits[:5])
            gen_met = judge_generation(question, self.generator._build_context(hits[:5]), answer)

            results.append({
                "question": question,
                "difficulty": difficulty,
                "gold_sources": gold,
                "answer": answer,
                "retrieval_metrics": ret_met,
                "citation_metrics": cit_met,
                "generation_metrics": gen_met
            })

        self._save(output_path, results, latencies)

    def _load_jsonl(self, path: str) -> List[Dict]:
        with open(path, "r") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _summarize_latencies(self, lats: List[float]) -> Dict:
        if not lats: return {}
        return {
            "mean": float(np.mean(lats)),
            "p50": float(np.percentile(lats, 50)),
            "p95": float(np.percentile(lats, 95)),
            "min": float(np.min(lats)),
            "max": float(np.max(lats))
        }

    def _save(self, path: str, results, latencies):
        # Aggregate logic for format_email.py
        summary = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "total_questions": len(results),
                "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2" if self.reranker else "None"
            },
            "retrieval_metrics": {
                f"recall@{k}": float(np.mean([r["retrieval_metrics"][f"recall@{k}"] for r in results]))
                for k in K_VALUES
            },
            "generation_metrics": {
                m: float(np.mean([r["generation_metrics"][m] for r in results]))
                for m in ["faithfulness", "relevance", "completeness", "overall"]
            },
            "citation_metrics": {
                "citation_validity": float(np.mean([r["citation_metrics"]["citation_validity"]["score"] for r in results])),
                "citation_coverage": float(np.mean([r["citation_metrics"]["citation_coverage"]["score"] for r in results])),
                "citation_grounding": float(np.mean([r["citation_metrics"]["source_grounding"]["score"] for r in results]))
            },
            "latency_summary": {k: self._summarize_latencies(v) for k, v in latencies.items()},
            "difficulty_breakdown": {}
        }

        # Add MRR/NDCG to retrieval summary
        for k in K_VALUES:
            summary["retrieval_metrics"][f"mrr@{k}"] = float(np.mean([r["retrieval_metrics"][f"mrr@{k}"] for r in results]))
            summary["retrieval_metrics"][f"ndcg@{k}"] = float(np.mean([r["retrieval_metrics"][f"ndcg@{k}"] for r in results]))

        # Difficulty breakdown
        for diff in ["Easy", "Medium", "Hard"]:
            subset = [r for r in results if r["difficulty"] == diff]
            if not subset: continue
            summary["difficulty_breakdown"][diff] = {
                "count": len(subset),
                "recall@1": float(np.mean([r["retrieval_metrics"]["recall@1"] for r in subset])),
                "recall@5": float(np.mean([r["retrieval_metrics"]["recall@5"] for r in subset])),
                "mrr@5": float(np.mean([r["retrieval_metrics"]["mrr@5"] for r in subset])),
                "ndcg@5": float(np.mean([r["retrieval_metrics"]["ndcg@5"] for r in subset])),
                "faithfulness": float(np.mean([r["generation_metrics"]["faithfulness"] for r in subset]))
            }

        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Evaluation complete. Results: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="eval_results_comprehensive.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    EvaluationEngine().run("contextual_eval_dataset.jsonl", args.output, limit=args.limit)

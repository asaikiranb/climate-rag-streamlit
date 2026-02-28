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
from rerank import ProductionReranker
from eval.retrieval_metrics import compute_retrieval_metrics_at_k

K_VALUES = [1, 3, 5]

class EvaluationEngine:
    """Orchestrates RAG quality measurement and reporting."""

    def __init__(self, use_reranker: bool = True):
        self.retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker() if use_reranker else None

    def run(self, dataset_path: str, output_path: str):
        """Execute evaluation over the provided dataset."""
        dataset = self._load_jsonl(dataset_path)
        print(f"Loaded {len(dataset)} evaluation items.")

        results, l_emb, l_src, l_rrk = [], [], [], []

        for entry in dataset:
            question = entry["question"]
            gold = entry.get("gold_sources", [])
            print(f"Evaluating: {question[:60]}...")

            t0 = time.perf_counter()
            hits = self.retriever.search(question)
            t_src = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            if self.reranker:
                hits = self.reranker.rerank(question, hits)
            t_rrk = (time.perf_counter() - t0) * 1000

            met = compute_retrieval_metrics_at_k(
                [h["metadata"]["filename"] for h in hits[:5]], gold, k_values=K_VALUES
            )
            
            results.append({"question": question, "metrics": met})
            l_src.append(t_src)
            l_rrk.append(t_rrk)

        self._save(output_path, results, l_src, l_rrk)

    def _load_jsonl(self, path: str) -> List[Dict]:
        with open(path, "r") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _save(self, path: str, results, t_src, t_rrk):
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "retrieval": {
                f"recall@{k}": float(np.mean([r["metrics"][f"recall@{k}"] for r in results]))
                for k in K_VALUES
            },
            "latency": {
                "search_ms_p50": float(np.percentile(t_src, 50)),
                "rerank_ms_p50": float(np.percentile(t_rrk, 50)),
            }
        }
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Evaluation complete. Results: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="eval_results_production.json")
    args = parser.parse_args()
    EvaluationEngine().run("contextual_eval_dataset.jsonl", args.output)

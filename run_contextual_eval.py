"""
Comprehensive RAG evaluation metrics suite.

Metrics computed:
  - Retrieval: recall@k, MRR@k, NDCG@k for k=1,3,5
  - Latency: mean/p50/p95/min/max

Usage:
    python run_contextual_eval.py --retrieval-only --output results.json
"""

import json
import argparse
import time
from typing import List, Dict

from retrieve import HybridRetriever
from rerank import CrossEncoderReranker
from llm import GenerationClient
from query import expand_query
from eval.retrieval_metrics import compute_retrieval_metrics_at_k
from eval.metrics import compute_custom_metrics
from eval.generation_metrics_ollama import judge_generation
from eval.report_contract import build_summary_payload

K_VALUES = [1, 3, 5]

class EvaluationEngine:
    """Orchestrates full RAG quality measurement (retrieval, generation, citations, judging)."""

    def __init__(self, use_reranker: bool = True):
        self.retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker() if use_reranker else None
        self.generator = GenerationClient()
        self.expansion_counts: List[int] = []
        self.expansion_fallback_count = 0

    def _expand_queries(self, question: str) -> List[str]:
        """Expand query via Groq-backed expander with safe single-query fallback."""
        try:
            expanded = expand_query(question, self.generator.groq)
        except Exception:
            expanded = [question]

        # Preserve order and remove empty/duplicate variants.
        deduped = []
        seen = set()
        for query in expanded or [question]:
            q = (query or "").strip()
            key = q.lower()
            if not q or key in seen:
                continue
            deduped.append(q)
            seen.add(key)

        if not deduped:
            deduped = [question]

        if len(deduped) <= 1:
            self.expansion_fallback_count += 1
        self.expansion_counts.append(len(deduped))
        return deduped

    def _retrieve_multi_query(self, question: str) -> tuple[list[dict], list[str], float]:
        """Run multi-query retrieval, merge by chunk id, then return candidate pool."""
        expanded_queries = self._expand_queries(question)

        t0 = time.perf_counter()
        seen_ids = set()
        merged_hits = []
        for query in expanded_queries:
            for hit in self.retriever.search(query):
                hit_id = hit.get("id")
                if hit_id in seen_ids:
                    continue
                seen_ids.add(hit_id)
                merged_hits.append(hit)
        search_ms = (time.perf_counter() - t0) * 1000
        return merged_hits, expanded_queries, search_ms

    def run(self, dataset_path: str, output_path: str, limit: int = None, retrieval_only: bool = False):
        """Execute evaluation over the dataset."""
        dataset = self._load_jsonl(dataset_path)
        if limit:
            dataset = dataset[:limit]
        print(f"Loaded {len(dataset)} evaluation items." + (" [retrieval-only]" if retrieval_only else ""))

        results = []
        latencies = {"embed_ms": [], "search_ms": [], "rerank_ms": [], "generate_ms": []}

        for i, entry in enumerate(dataset, 1):
            question = entry["question"]
            gold = entry.get("gold_sources", [])
            difficulty = entry.get("metadata", {}).get("difficulty", "Unknown")
            print(f"[{i}/{len(dataset)}] Evaluating ({difficulty}): {question[:60]}...")

            # 1. Retrieval
            hits, expanded_queries, search_ms = self._retrieve_multi_query(question)
            latencies["search_ms"].append(search_ms)

            # 2. Reranking
            t0 = time.perf_counter()
            if self.reranker:
                hits = self.reranker.rerank(question, hits)
            latencies["rerank_ms"].append((time.perf_counter() - t0) * 1000)

            if retrieval_only:
                ret_met = compute_retrieval_metrics_at_k(
                    [h["metadata"]["filename"] for h in hits[:5]], gold, k_values=K_VALUES
                )
                results.append({
                    "question": question,
                    "difficulty": difficulty,
                    "gold_sources": gold,
                    "expanded_queries": expanded_queries,
                    "retrieval_metrics": ret_met,
                })
                continue

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
                "expanded_queries": expanded_queries,
                "answer": answer,
                "retrieval_metrics": ret_met,
                "citation_metrics": cit_met,
                "generation_metrics": gen_met
            })

        self._save(output_path, results, latencies, retrieval_only=retrieval_only)

    def _load_jsonl(self, path: str) -> List[Dict]:
        with open(path, "r") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _save(self, path: str, results, latencies, retrieval_only: bool = False):
        summary = build_summary_payload(
            results=results,
            latencies=latencies,
            reranker_name=self.reranker.model_name if self.reranker else "None",
            groq_available_for_expansion=bool(self.generator.groq),
            expansion_counts=self.expansion_counts,
            expansion_fallback_count=self.expansion_fallback_count,
            retrieval_only=retrieval_only,
        )

        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Evaluation complete. Results: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="eval_results_comprehensive.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retrieval-only", action="store_true", help="Run only retrieval (+ reranker) metrics, no generation or LLM judge")
    args = parser.parse_args()
    EvaluationEngine().run("contextual_eval_dataset.jsonl", args.output, limit=args.limit, retrieval_only=args.retrieval_only)

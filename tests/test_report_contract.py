"""Tests for retrieval-only report contract and email rendering."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.report_contract import build_summary_payload
from format_email import build_html


class ReportContractTests(unittest.TestCase):
    def test_retrieval_only_summary_uses_locked_top_level_keys(self) -> None:
        results = [
            {
                "difficulty": "Easy",
                "retrieval_metrics": {
                    "recall@1": 0.8,
                    "recall@3": 0.9,
                    "recall@5": 1.0,
                    "mrr@1": 0.8,
                    "mrr@3": 0.85,
                    "mrr@5": 0.88,
                    "ndcg@1": 0.8,
                    "ndcg@3": 0.84,
                    "ndcg@5": 0.9,
                },
            },
            {
                "difficulty": "Hard",
                "retrieval_metrics": {
                    "recall@1": 0.6,
                    "recall@3": 0.7,
                    "recall@5": 0.8,
                    "mrr@1": 0.6,
                    "mrr@3": 0.66,
                    "mrr@5": 0.7,
                    "ndcg@1": 0.6,
                    "ndcg@3": 0.68,
                    "ndcg@5": 0.74,
                },
            },
        ]
        latencies = {
            "embed_ms": [],
            "search_ms": [10.0, 20.0],
            "rerank_ms": [30.0, 40.0],
            "generate_ms": [],
        }

        summary = build_summary_payload(
            results=results,
            latencies=latencies,
            reranker_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            groq_available_for_expansion=True,
            expansion_counts=[3, 4],
            expansion_fallback_count=0,
            retrieval_only=True,
        )

        self.assertEqual(
            list(summary.keys()),
            [
                "timestamp",
                "config",
                "retrieval_metrics",
                "latency_summary",
                "expansion_summary",
                "difficulty_breakdown",
            ],
        )
        self.assertTrue(summary["config"]["multi_query_expansion"])
        self.assertTrue(summary["config"]["groq_available_for_expansion"])
        self.assertNotIn("generation_metrics", summary)
        self.assertNotIn("citation_metrics", summary)

    def test_email_hides_generation_and_citation_in_retrieval_only_mode(self) -> None:
        retrieval_only_summary = {
            "timestamp": "2026-03-06T07:47:11.344399",
            "config": {
                "total_questions": 117,
                "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "multi_query_expansion": True,
                "groq_available_for_expansion": True,
            },
            "retrieval_metrics": {
                "recall@1": 0.69,
                "recall@3": 0.78,
                "recall@5": 0.88,
                "mrr@1": 0.69,
                "ndcg@1": 0.69,
                "mrr@3": 0.73,
                "ndcg@3": 0.74,
                "mrr@5": 0.75,
                "ndcg@5": 0.78,
            },
            "latency_summary": {
                "embed_ms": {},
                "search_ms": {"mean": 1000, "p50": 900, "p95": 1500, "min": 600, "max": 3000},
                "rerank_ms": {"mean": 5000, "p50": 4900, "p95": 5600, "min": 4700, "max": 5900},
                "generate_ms": {},
            },
            "expansion_summary": {
                "avg_queries_per_question": 3.3,
                "fallback_single_query_count": 0,
                "fallback_single_query_rate": 0.0,
            },
            "difficulty_breakdown": {
                "Easy": {"count": 39, "recall@1": 0.70, "recall@5": 0.87, "mrr@5": 0.75, "ndcg@5": 0.78},
                "Medium": {
                    "count": 39,
                    "recall@1": 0.74,
                    "recall@5": 0.90,
                    "mrr@5": 0.79,
                    "ndcg@5": 0.81,
                },
                "Hard": {"count": 39, "recall@1": 0.64, "recall@5": 0.87, "mrr@5": 0.72, "ndcg@5": 0.76},
            },
        }

        html = build_html(
            retrieval_only_summary,
            branch="feature/test",
            commit="0123456789abcdef",
            repo="asaikiranb/RAG-climate",
        )

        self.assertIn("Expansion Summary", html)
        self.assertNotIn("Generation Metrics", html)
        self.assertNotIn("Citation Metrics", html)
        self.assertNotIn("Faithfulness", html)


if __name__ == "__main__":
    unittest.main()


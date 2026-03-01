"""Tests for the additive offline evaluation pipeline."""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.latency import QueryLatency
from eval.loader import parse_page_range
from eval.metrics import compute_doc_retrieval_scores, compute_page_retrieval_scores
from eval.writers import build_summary


class EvalPipelineTests(unittest.TestCase):
    def test_parse_page_range(self) -> None:
        self.assertEqual(parse_page_range("[19]"), [19])
        self.assertEqual(parse_page_range("[29,30,31]"), [29, 30, 31])
        self.assertIsNone(parse_page_range(""))

    def test_doc_and_page_metrics(self) -> None:
        retrievals = [
            {"rank": 1, "filename": "other.pdf", "page": 3, "snippet": "irrelevant"},
            {"rank": 2, "filename": "Manual.PDF", "page": 10, "snippet": "compressor safety section"},
        ]

        doc_scores = compute_doc_retrieval_scores(retrievals, "manual.pdf")
        self.assertTrue(doc_scores.scored)
        self.assertEqual(doc_scores.hits[1], 0)
        self.assertEqual(doc_scores.hits[3], 1)
        self.assertAlmostEqual(doc_scores.rr or 0.0, 0.5)
        self.assertAlmostEqual(doc_scores.ndcg[3] or 0.0, 1.0 / math.log2(3))

        page_scores = compute_page_retrieval_scores(
            retrievals=retrievals,
            gold_source="manual.pdf",
            gold_pages=[7],
            anchor_text="",
            anchor_threshold=80,
        )
        self.assertTrue(page_scores.scored)
        self.assertEqual(page_scores.hits[3], 0)
        self.assertEqual(page_scores.rr, 0.0)
        self.assertAlmostEqual(page_scores.ndcg[3] or 0.0, 0.5 / math.log2(3))

    def test_unrated_rows_are_excluded_from_aggregation(self) -> None:
        rows = [
            {
                "difficulty": "Easy",
                "gold_sources": "manual.pdf",
                "doc_scored": True,
                "page_scored": False,
                "doc_rr": 1.0,
                "page_rr": None,
                "doc_hit@1": 1,
                "doc_hit@3": 1,
                "doc_hit@5": 1,
                "doc_hit@10": 1,
                "page_hit@1": None,
                "page_hit@3": None,
                "page_hit@5": None,
                "page_hit@10": None,
                "doc_ndcg@1": 1.0,
                "doc_ndcg@3": 1.0,
                "doc_ndcg@5": 1.0,
                "doc_ndcg@10": 1.0,
                "page_ndcg@1": None,
                "page_ndcg@3": None,
                "page_ndcg@5": None,
                "page_ndcg@10": None,
            },
            {
                "difficulty": "Easy",
                "gold_sources": "",
                "doc_scored": False,
                "page_scored": False,
                "doc_rr": None,
                "page_rr": None,
                "doc_hit@1": None,
                "doc_hit@3": None,
                "doc_hit@5": None,
                "doc_hit@10": None,
                "page_hit@1": None,
                "page_hit@3": None,
                "page_hit@5": None,
                "page_hit@10": None,
                "doc_ndcg@1": None,
                "doc_ndcg@3": None,
                "doc_ndcg@5": None,
                "doc_ndcg@10": None,
                "page_ndcg@1": None,
                "page_ndcg@3": None,
                "page_ndcg@5": None,
                "page_ndcg@10": None,
            },
        ]
        summary = build_summary(
            rows,
            [QueryLatency(total_ms=10.0, embed_ms=None, search_ms=5.0, rerank_ms=3.0, generate_ms=None)],
        )

        self.assertEqual(summary["retrieval"]["doc"]["count"], 1)
        self.assertEqual(summary["retrieval"]["page"]["count"], 0)
        self.assertEqual(summary["retrieval"]["doc"]["recall@1"], 1.0)
        self.assertIsNone(summary["retrieval"]["page"]["recall@1"])

    def test_summary_includes_page_ndcg_note(self) -> None:
        summary = build_summary([], [])
        note = summary["metric_notes"]["page_ndcg"]
        self.assertIn("rel=2", note)
        self.assertIn("rel=1", note)


if __name__ == "__main__":
    unittest.main()

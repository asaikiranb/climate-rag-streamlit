"""Unit tests for doc-first ingestion and benchmark helpers."""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmark_doc_first import EXPECTED_HEADERS, metric_gate, write_stratified_subset
from pipeline_utils import normalize_whitespace, stable_chunk_id


class DocFirstPipelineTests(unittest.TestCase):
    def test_stable_chunk_id_is_deterministic(self) -> None:
        cid1 = stable_chunk_id("abc123", 5, 2, 7)
        cid2 = stable_chunk_id("abc123", 5, 2, 7)
        cid3 = stable_chunk_id("abc123", 5, 2, 8)
        self.assertEqual(cid1, cid2)
        self.assertNotEqual(cid1, cid3)

    def test_normalize_whitespace(self) -> None:
        text = " A\n\n  B\t\tC "
        self.assertEqual(normalize_whitespace(text), "A B C")

    def test_metric_gate_requires_four_and_hard_pair(self) -> None:
        summary = {
            "retrieval": {
                "doc": {
                    "recall@1": 0.80,
                    "recall@3": 0.85,
                    "recall@5": 0.90,
                    "recall@10": 0.90,
                    "ndcg@5": 0.80,
                    "mrr@10": 0.79,
                }
            }
        }
        gate = metric_gate(summary)
        self.assertTrue(gate["overall_pass"])
        self.assertTrue(gate["required_pair"])
        self.assertGreaterEqual(gate["passed_count"], 4)

    def test_write_stratified_subset_respects_size(self) -> None:
        rows = []
        for idx in range(15):
            rows.append(
                {
                    "Question": f"Q{idx}",
                    "gold_sources": "doc.pdf",
                    "metadata": '{"difficulty":"Easy"}' if idx < 5 else ('{"difficulty":"Medium"}' if idx < 10 else '{"difficulty":"Hard"}'),
                    "page_range": "[1]",
                    "anchor_text": "anchor",
                }
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, "input.csv")
            output_path = os.path.join(tmp_dir, "output.csv")

            with open(input_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=EXPECTED_HEADERS)
                writer.writeheader()
                writer.writerows(rows)

            write_stratified_subset(
                input_csv=Path(input_path),
                output_csv=Path(output_path),
                sample_size=9,
            )

            with open(output_path, "r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                output_rows = list(reader)

        self.assertEqual(len(output_rows), 9)


if __name__ == "__main__":
    unittest.main()

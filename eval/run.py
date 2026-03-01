"""Offline evaluation runner for the Groq-backed RAG pipeline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from eval.latency import QueryLatency
from eval.loader import EvalRow, load_golden_csv
from eval.metrics import (
    K_VALUES,
    compute_doc_retrieval_scores,
    compute_page_retrieval_scores,
)
from eval.normalize import normalize_retrievals
from eval.writers import build_summary, print_console_summary, write_json, write_jsonl


class OfflineGroqEvalRunner:
    """Run the local RAG pipeline over a golden CSV and score retrieval quality."""

    def __init__(self, top_k: int, anchor_threshold: int) -> None:
        self.output_top_n = max(top_k, max(K_VALUES))
        self.anchor_threshold = anchor_threshold
        from rerank import CrossEncoderReranker
        from retrieve import HybridRetriever

        self.retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker()
        self.generator: Any | None = None
        self.expand_query_fn: Any | None = None
        self.groq_available = False

        try:
            from llm import GenerationClient

            self.generator = GenerationClient()
            self.groq_available = bool(self.generator.groq)
            if self.groq_available:
                try:
                    from query import expand_query as expand_query_fn

                    self.expand_query_fn = expand_query_fn
                except Exception:
                    self.expand_query_fn = None
        except Exception:
            self.generator = None
            self.expand_query_fn = None
            self.groq_available = False

    def _retrieve(self, question: str) -> tuple[list[dict[str, Any]], float | None, float | None]:
        search_started = time.perf_counter()
        expanded_queries = [question]
        if self.groq_available and self.expand_query_fn is not None and self.generator is not None:
            expanded_queries = self.expand_query_fn(question, self.generator.groq)

        seen_ids: set[str] = set()
        candidates: list[dict[str, Any]] = []
        for expanded_query in expanded_queries:
            for result in self.retriever.search(expanded_query):
                result_id = str(result["id"])
                if result_id in seen_ids:
                    continue
                seen_ids.add(result_id)
                candidates.append(result)

        search_ms = (time.perf_counter() - search_started) * 1000.0

        rerank_started = time.perf_counter()
        reranked = self.reranker.rerank(question, candidates)
        rerank_ms = (time.perf_counter() - rerank_started) * 1000.0
        return reranked, search_ms, rerank_ms

    def _generate_answer(self, question: str, hits: list[dict[str, Any]]) -> tuple[str, float | None]:
        if not hits or not self.groq_available or self.generator is None:
            return "", None

        generate_started = time.perf_counter()
        answer = self.generator.generate(question, hits[:5], use_fallback=False)
        generate_ms = (time.perf_counter() - generate_started) * 1000.0
        return answer, generate_ms

    def _score_row(
        self,
        row: EvalRow,
        retrievals: list[dict[str, Any]],
        answer_text: str,
        latency: QueryLatency,
        error_type: str | None,
        error_message: str | None,
    ) -> dict[str, Any]:
        doc_scores = compute_doc_retrieval_scores(retrievals, row.gold_sources)
        page_scores = compute_page_retrieval_scores(
            retrievals=retrievals,
            gold_source=row.gold_sources,
            gold_pages=row.gold_pages,
            anchor_text=row.anchor_text,
            anchor_threshold=self.anchor_threshold,
        )

        payload: dict[str, Any] = {
            "question_id": row.question_id,
            "Question": row.question,
            "difficulty": row.difficulty,
            "gold_sources": row.gold_sources or None,
            "gold_pages": row.gold_pages,
            "anchor_text": row.anchor_text or None,
            "answer_text": answer_text,
            "normalized_retrieval_topN": retrievals,
            "doc_scored": doc_scores.scored,
            "page_scored": page_scores.scored,
            "doc_rr": round(doc_scores.rr, 6) if doc_scores.rr is not None else None,
            "page_rr": round(page_scores.rr, 6) if page_scores.rr is not None else None,
            "latency_ms": round(latency.total_ms, 3),
            "error_type": error_type,
            "error_message": error_message,
            "contextual_conversation_id": None,
            "contextual_message_id": None,
        }

        for k in K_VALUES:
            doc_hit = doc_scores.hits[k]
            page_hit = page_scores.hits[k]
            doc_ndcg = doc_scores.ndcg[k]
            page_ndcg = page_scores.ndcg[k]
            payload[f"doc_hit@{k}"] = doc_hit
            payload[f"page_hit@{k}"] = page_hit
            payload[f"doc_ndcg@{k}"] = round(doc_ndcg, 6) if doc_ndcg is not None else None
            payload[f"page_ndcg@{k}"] = round(page_ndcg, 6) if page_ndcg is not None else None

        return payload

    def run(
        self,
        input_csv: str,
        output_dir: str,
        limit: int | None = None,
    ) -> tuple[Path, Path]:
        """Execute the evaluation and write the result artifacts."""

        rows = load_golden_csv(input_csv, limit=limit)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Loaded {len(rows)} rows from {input_csv}")
        if not self.groq_available:
            print("GROQ_API_KEY not set; answer_text will be empty and generation latency will be null.")

        per_query_rows: list[dict[str, Any]] = []
        latencies: list[QueryLatency] = []

        for index, row in enumerate(rows, start=1):
            started = time.perf_counter()
            normalized_retrievals: list[dict[str, Any]] = []
            answer_text = ""
            error_type: str | None = None
            error_message: str | None = None
            search_ms: float | None = None
            rerank_ms: float | None = None
            generate_ms: float | None = None

            print(f"[{index}/{len(rows)}] {row.question[:80]}")

            try:
                raw_hits, search_ms, rerank_ms = self._retrieve(row.question)
                normalized_retrievals = normalize_retrievals(raw_hits, self.output_top_n)
                answer_text, generate_ms = self._generate_answer(row.question, raw_hits)
            except Exception as exc:
                error_type = type(exc).__name__
                error_message = str(exc)

            total_ms = (time.perf_counter() - started) * 1000.0
            latency = QueryLatency(
                total_ms=total_ms,
                embed_ms=None,
                search_ms=search_ms,
                rerank_ms=rerank_ms,
                generate_ms=generate_ms,
            )
            latencies.append(latency)
            per_query_rows.append(
                self._score_row(
                    row=row,
                    retrievals=normalized_retrievals,
                    answer_text=answer_text,
                    latency=latency,
                    error_type=error_type,
                    error_message=error_message,
                )
            )

        per_query_path = write_jsonl(output_path / "per_query.jsonl", per_query_rows)
        summary_payload = build_summary(per_query_rows, latencies)
        summary_path = write_json(output_path / "summary.json", summary_payload)
        print_console_summary(summary_payload)
        print("")
        print(f"Per-query JSONL: {per_query_path}")
        print(f"Summary JSON: {summary_path}")

        return per_query_path, summary_path


def add_eval_subcommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the eval CLI subcommand."""

    parser = subparsers.add_parser("eval", help="Run the offline retrieval evaluation.")
    parser.add_argument("--input", required=True, help="Path to the golden CSV.")
    parser.add_argument("--out", required=True, help="Output directory for eval artifacts.")
    parser.add_argument("--top-k", type=int, default=10, help="Normalize at least this many retrievals.")
    parser.add_argument(
        "--anchor-threshold",
        type=int,
        default=80,
        help="Fuzzy anchor match threshold from 0 to 100.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit rows for smoke tests.")
    parser.set_defaults(handler=run_from_args)


def run_from_args(args: argparse.Namespace) -> int:
    """CLI adapter."""

    runner = OfflineGroqEvalRunner(top_k=args.top_k, anchor_threshold=args.anchor_threshold)
    runner.run(input_csv=args.input, output_dir=args.out, limit=args.limit)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Standalone eval entrypoint."""

    parser = argparse.ArgumentParser(prog="python -m eval.run")
    parser.add_argument("--input", required=True, help="Path to the golden CSV.")
    parser.add_argument("--out", required=True, help="Output directory for eval artifacts.")
    parser.add_argument("--top-k", type=int, default=10, help="Normalize at least this many retrievals.")
    parser.add_argument(
        "--anchor-threshold",
        type=int,
        default=80,
        help="Fuzzy anchor match threshold from 0 to 100.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit rows for smoke tests.")
    args = parser.parse_args(argv)
    return run_from_args(args)


if __name__ == "__main__":
    raise SystemExit(main())

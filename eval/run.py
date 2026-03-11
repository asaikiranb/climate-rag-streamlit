"""Offline evaluation runner for the Groq-backed RAG pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from config import RETRIEVAL_CANDIDATE_K
from config import EVAL_EMAIL_COMPAT_MODE, EVAL_PRIMARY_CONTRACT
from eval.compat_email_payload import to_email_payload
from eval.latency import QueryLatency
from eval.loader import EvalRow, load_golden_csv
from eval.metrics import (
    K_VALUES,
    compute_doc_retrieval_scores,
    compute_page_retrieval_scores,
)
from eval.normalize import normalize_retrievals
from eval.writers import build_summary, print_console_summary, write_json, write_jsonl

SENSITIVE_TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{20,}")


def _redact_sensitive(value: Any) -> str:
    text = str(value)
    token = os.getenv("HF_TOKEN", "").strip()
    if token:
        text = text.replace(token, "[REDACTED_HF_TOKEN]")
    return SENSITIVE_TOKEN_PATTERN.sub("[REDACTED_HF_TOKEN]", text)


def _sha1_file(path: str | Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    env_sha = os.getenv("GITHUB_SHA", "").strip()
    if env_sha:
        return env_sha
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


class OfflineGroqEvalRunner:
    """Run the local RAG pipeline over a golden CSV and score retrieval quality."""

    def __init__(
        self,
        top_k: int,
        anchor_threshold: int,
        profile: str = "baseline",
        backend: str = "qdrant",
        embedding_model: str | None = None,
        sparse_mode: str = "none",
        retrieval_candidate_k: int | None = None,
        stage1_pool_size: int | None = None,
        chroma_path: str | None = None,
        chroma_collection: str | None = None,
        qdrant_path: str | None = None,
        qdrant_collection: str | None = None,
    ) -> None:
        self.output_top_n = max(top_k, max(K_VALUES))
        self.anchor_threshold = anchor_threshold
        self.profile = profile.strip().lower()
        self.backend = backend
        self.embedding_model = embedding_model
        self.sparse_mode = sparse_mode
        self.contract = EVAL_PRIMARY_CONTRACT
        self.retrieval_candidate_k = max(1, int(retrieval_candidate_k or RETRIEVAL_CANDIDATE_K))
        self.stage1_pool_size = stage1_pool_size
        self.chroma_path = chroma_path
        self.chroma_collection = chroma_collection
        self.qdrant_path = qdrant_path
        self.qdrant_collection = qdrant_collection

        if self.profile == "upgraded":
            from rerank_v2 import TwoStageCalibratedReranker
            from retrieve_v2 import HybridRetrieverV2

            retriever_kwargs: dict[str, Any] = {
                "backend": backend,
                "sparse_mode": sparse_mode,
            }
            if embedding_model:
                retriever_kwargs["embedding_model"] = embedding_model
            if chroma_path:
                retriever_kwargs["chroma_path"] = chroma_path
            if chroma_collection:
                retriever_kwargs["chroma_collection"] = chroma_collection
            if qdrant_path:
                retriever_kwargs["qdrant_path"] = qdrant_path
            if qdrant_collection:
                retriever_kwargs["qdrant_collection"] = qdrant_collection

            self.retriever = HybridRetrieverV2(**retriever_kwargs)
            reranker_kwargs: dict[str, Any] = {}
            if stage1_pool_size is not None:
                reranker_kwargs["stage1_pool_size"] = int(stage1_pool_size)
            self.reranker = TwoStageCalibratedReranker(**reranker_kwargs)
        else:
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
            if self.groq_available and self.profile == "baseline":
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
        if (
            self.profile == "baseline"
            and self.groq_available
            and self.expand_query_fn is not None
            and self.generator is not None
        ):
            expanded_queries = self.expand_query_fn(question, self.generator.groq)

        seen_ids: set[str] = set()
        candidates: list[dict[str, Any]] = []
        for expanded_query in expanded_queries:
            for result in self.retriever.search(expanded_query, top_k=self.retrieval_candidate_k):
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

    def _build_manifest(self, input_csv: str, rows_count: int, limit: int | None) -> dict[str, Any]:
        retriever_backend = getattr(self.retriever, "backend", self.backend)
        retriever_sparse = getattr(self.retriever, "sparse_mode", self.sparse_mode)
        reranker_name = getattr(self.reranker, "model_name", None)
        doc_first_enabled = getattr(self.reranker, "doc_first_enabled", None)
        doc_first_aggregate_top_k = getattr(self.reranker, "doc_first_aggregate_top_k", None)
        doc_first_aggregate_decay = getattr(self.reranker, "doc_first_aggregate_decay", None)
        if not reranker_name:
            reranker_name = getattr(self.reranker, "stage1_model_name", None)

        manifest: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "contract": self.contract,
            "profile": self.profile,
            "git_commit": _git_commit(),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "input": {
                "path": str(input_csv),
                "sha1": _sha1_file(input_csv),
                "rows": int(rows_count),
                "limit": limit,
            },
            "retriever": {
                "backend": str(retriever_backend),
                "embedding_model": str(getattr(self.retriever, "embedding_model", self.embedding_model) or ""),
                "sparse_mode": str(retriever_sparse),
                "top_n": self.output_top_n,
                "candidate_k": self.retrieval_candidate_k,
                "chroma_path": self.chroma_path,
                "chroma_collection": self.chroma_collection,
                "qdrant_path": self.qdrant_path,
                "qdrant_collection": self.qdrant_collection,
            },
            "reranker": {
                "name": reranker_name,
                "mode": "single_stage_calibrated",
                "stage1_pool_size": self.stage1_pool_size,
                "doc_first_enabled": doc_first_enabled,
                "doc_first_aggregate_top_k": doc_first_aggregate_top_k,
                "doc_first_aggregate_decay": doc_first_aggregate_decay,
            },
            "k_values": list(K_VALUES),
            "anchor_threshold": self.anchor_threshold,
        }
        return manifest

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
                error_message = _redact_sensitive(exc)

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
        manifest_payload = self._build_manifest(input_csv=input_csv, rows_count=len(rows), limit=limit)
        manifest_path = write_json(output_path / "eval_manifest.json", manifest_payload)

        if EVAL_EMAIL_COMPAT_MODE:
            compat_payload = to_email_payload(summary_payload=summary_payload, manifest_payload=manifest_payload)
            write_json(output_path / "eval_results_comprehensive.json", compat_payload)

        print_console_summary(summary_payload)
        print("")
        print(f"Per-query JSONL: {per_query_path}")
        print(f"Summary JSON: {summary_path}")
        print(f"Manifest JSON: {manifest_path}")

        if hasattr(self.retriever, "close"):
            try:
                self.retriever.close()
            except Exception:
                pass

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
    parser.add_argument(
        "--profile",
        choices=["baseline", "upgraded"],
        default="baseline",
        help="Choose baseline (current) or upgraded retrieval+reranking profile.",
    )
    parser.add_argument(
        "--backend",
        choices=["chroma", "qdrant"],
        default="qdrant",
        help="Vector backend for upgraded profile.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Override embedding model for upgraded profile.",
    )
    parser.add_argument(
        "--sparse-mode",
        choices=["none", "bm42", "splade"],
        default="none",
        help="Sparse branch mode for upgraded profile.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="Override retrieval candidate pool size for upgraded profile.",
    )
    parser.add_argument(
        "--stage1-pool-size",
        type=int,
        default=None,
        help="Override stage-1 reranker pool size for upgraded profile.",
    )
    parser.add_argument("--chroma-path", default=None, help="Override Chroma path for upgraded profile.")
    parser.add_argument(
        "--chroma-collection",
        default=None,
        help="Override Chroma collection for upgraded profile.",
    )
    parser.add_argument("--qdrant-path", default=None, help="Override Qdrant path for upgraded profile.")
    parser.add_argument(
        "--qdrant-collection",
        default=None,
        help="Override Qdrant collection for upgraded profile.",
    )
    parser.set_defaults(handler=run_from_args)


def run_from_args(args: argparse.Namespace) -> int:
    """CLI adapter."""

    runner = OfflineGroqEvalRunner(
        top_k=args.top_k,
        anchor_threshold=args.anchor_threshold,
        profile=args.profile,
        backend=args.backend,
        embedding_model=args.embedding_model,
        sparse_mode=args.sparse_mode,
        retrieval_candidate_k=args.candidate_k,
        stage1_pool_size=args.stage1_pool_size,
        chroma_path=args.chroma_path,
        chroma_collection=args.chroma_collection,
        qdrant_path=args.qdrant_path,
        qdrant_collection=args.qdrant_collection,
    )
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
    parser.add_argument(
        "--profile",
        choices=["baseline", "upgraded"],
        default="baseline",
        help="Choose baseline (current) or upgraded retrieval+reranking profile.",
    )
    parser.add_argument(
        "--backend",
        choices=["chroma", "qdrant"],
        default="qdrant",
        help="Vector backend for upgraded profile.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Override embedding model for upgraded profile.",
    )
    parser.add_argument(
        "--sparse-mode",
        choices=["none", "bm42", "splade"],
        default="none",
        help="Sparse branch mode for upgraded profile.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="Override retrieval candidate pool size for upgraded profile.",
    )
    parser.add_argument(
        "--stage1-pool-size",
        type=int,
        default=None,
        help="Override stage-1 reranker pool size for upgraded profile.",
    )
    parser.add_argument("--chroma-path", default=None, help="Override Chroma path for upgraded profile.")
    parser.add_argument(
        "--chroma-collection",
        default=None,
        help="Override Chroma collection for upgraded profile.",
    )
    parser.add_argument("--qdrant-path", default=None, help="Override Qdrant path for upgraded profile.")
    parser.add_argument(
        "--qdrant-collection",
        default=None,
        help="Override Qdrant collection for upgraded profile.",
    )
    args = parser.parse_args(argv)
    return run_from_args(args)


if __name__ == "__main__":
    raise SystemExit(main())

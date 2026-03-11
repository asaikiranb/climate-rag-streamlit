"""Formats evaluation results as a structured HTML email for stakeholder reporting."""

import argparse
import json
from datetime import datetime


def metric_color(value: float) -> str:
    if value >= 0.8:
        return "#2ea043"
    if value >= 0.6:
        return "#d29922"
    return "#cf222e"


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_payload(data: dict) -> dict:
    required = {
        "config",
        "retrieval_metrics",
        "latency_summary",
        "difficulty_breakdown",
    }
    if required.issubset(set(data.keys())):
        return data

    candidate = data.get("summary") if isinstance(data.get("summary"), dict) else data
    if required.issubset(set(candidate.keys())):
        return candidate

    if "retrieval" in candidate:
        try:
            from eval.compat_email_payload import to_email_payload

            manifest = data.get("manifest") if isinstance(data.get("manifest"), dict) else None
            return to_email_payload(summary_payload=candidate, manifest_payload=manifest)
        except Exception:
            return candidate

    return candidate


def build_html(data: dict, branch: str, commit: str, repo: str) -> str:
    data = _normalize_payload(data)

    config = data.get("config", {})
    retrieval = data.get("retrieval_metrics", {})
    latency = data.get("latency_summary", {})
    difficulty = data.get("difficulty_breakdown", {})
    generation = data.get("generation_metrics") or {}
    citation = data.get("citation_metrics") or {}

    commit_short = commit[:7]
    commit_url = f"https://github.com/{repo}/commit/{commit}"
    branch_url = f"https://github.com/{repo}/tree/{branch}"
    timestamp = str(data.get("timestamp", datetime.now().isoformat()))[:19]

    html_parts: list[str] = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<style>",
        "  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #24292f; max-width: 760px; margin: 0 auto; padding: 20px; }",
        "  h1 { font-size: 22px; border-bottom: 2px solid #d0d7de; padding-bottom: 8px; }",
        "  h2 { font-size: 16px; color: #57606a; margin-top: 24px; }",
        "  table { border-collapse: collapse; width: 100%; margin: 8px 0 16px; font-size: 14px; }",
        "  th { background: #f6f8fa; text-align: left; padding: 8px 12px; border: 1px solid #d0d7de; font-weight: 600; }",
        "  td { padding: 8px 12px; border: 1px solid #d0d7de; }",
        "  .metric { font-weight: 600; font-family: monospace; }",
        "  .meta { font-size: 13px; color: #57606a; }",
        "  .footer { margin-top: 24px; padding-top: 12px; border-top: 1px solid #d0d7de; font-size: 12px; color: #8b949e; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>RAG Evaluation Results</h1>",
        (
            f"<p class=\"meta\">"
            f"Branch: <a href=\"{branch_url}\"><strong>{branch}</strong></a> &nbsp;|&nbsp; "
            f"Commit: <a href=\"{commit_url}\"><code>{commit_short}</code></a> &nbsp;|&nbsp; "
            f"{timestamp} &nbsp;|&nbsp; "
            f"{int(_to_float(config.get('total_questions'), 0))} questions &nbsp;|&nbsp; "
            f"Reranker: {config.get('reranker', 'N/A')}"
            f"</p>"
        ),
        "<h2>Retrieval Metrics</h2>",
        "<table>",
        "  <tr><th>k</th><th>Recall</th><th>MRR</th><th>NDCG</th></tr>",
    ]

    for k in [1, 3, 5]:
        recall = _to_float(retrieval.get(f"recall@{k}"))
        mrr = _to_float(retrieval.get(f"mrr@{k}"))
        ndcg = _to_float(retrieval.get(f"ndcg@{k}"))
        html_parts.extend(
            [
                "  <tr>",
                f"    <td><strong>{k}</strong></td>",
                f"    <td class=\"metric\" style=\"color:{metric_color(recall)}\">{recall:.4f}</td>",
                f"    <td class=\"metric\" style=\"color:{metric_color(mrr)}\">{mrr:.4f}</td>",
                f"    <td class=\"metric\" style=\"color:{metric_color(ndcg)}\">{ndcg:.4f}</td>",
                "  </tr>",
            ]
        )

    html_parts.extend(
        [
            "</table>",
            "<h2>Generation Metrics</h2>",
            "<table>",
            "  <tr><th>Metric</th><th>Score</th></tr>",
        ]
    )

    for metric_name, label in [
        ("faithfulness", "Faithfulness"),
        ("relevance", "Relevance"),
        ("completeness", "Completeness"),
        ("overall", "Overall"),
    ]:
        score = _to_float(generation.get(metric_name), 0.0)
        html_parts.append(
            f"  <tr><td>{label}</td><td class=\"metric\" style=\"color:{metric_color(score)}\">{score:.4f}</td></tr>"
        )

    html_parts.extend(
        [
            "</table>",
            "<h2>Citation Metrics</h2>",
            "<table>",
            "  <tr><th>Metric</th><th>Score</th></tr>",
        ]
    )

    for key, label in [
        ("citation_validity", "Validity"),
        ("citation_coverage", "Coverage"),
        ("citation_grounding", "Grounding"),
    ]:
        score = _to_float(citation.get(key), 0.0)
        html_parts.append(
            f"  <tr><td>{label}</td><td class=\"metric\" style=\"color:{metric_color(score)}\">{score:.4f}</td></tr>"
        )

    html_parts.extend(
        [
            "</table>",
            "<h2>Latency Summary</h2>",
            "<table>",
            "  <tr><th>Stage</th><th>Mean</th><th>P50</th><th>P95</th><th>Min</th><th>Max</th></tr>",
        ]
    )

    for stage in ["embed_ms", "search_ms", "rerank_ms", "generate_ms"]:
        stats = latency.get(stage, {}) or {}
        html_parts.extend(
            [
                "  <tr>",
                f"    <td>{stage}</td>",
                f"    <td>{_to_float(stats.get('mean')):.1f}ms</td>",
                f"    <td>{_to_float(stats.get('p50')):.1f}ms</td>",
                f"    <td>{_to_float(stats.get('p95')):.1f}ms</td>",
                f"    <td>{_to_float(stats.get('min')):.1f}ms</td>",
                f"    <td>{_to_float(stats.get('max')):.1f}ms</td>",
                "  </tr>",
            ]
        )

    html_parts.extend(
        [
            "</table>",
            "<h2>By Difficulty</h2>",
            "<table>",
            "  <tr><th>Difficulty</th><th>Count</th><th>Top-1 Acc</th><th>Recall@5</th><th>MRR@5</th><th>NDCG@5</th><th>Faithfulness</th></tr>",
        ]
    )

    for level in ["Easy", "Medium", "Hard"]:
        metrics = difficulty.get(level, {}) or {}
        recall_at_1 = _to_float(metrics.get("recall@1"), 0.0)
        top1_pct = f"{recall_at_1 * 100:.1f}%"
        html_parts.append(
            "  <tr>"
            f"<td><strong>{level}</strong></td>"
            f"<td>{int(_to_float(metrics.get('count'), 0.0))}</td>"
            f"<td class=\"metric\" style=\"color:{metric_color(recall_at_1)}\">{top1_pct}</td>"
            f"<td class=\"metric\">{_to_float(metrics.get('recall@5'), 0.0):.4f}</td>"
            f"<td class=\"metric\">{_to_float(metrics.get('mrr@5'), 0.0):.4f}</td>"
            f"<td class=\"metric\">{_to_float(metrics.get('ndcg@5'), 0.0):.4f}</td>"
            f"<td class=\"metric\">{_to_float(metrics.get('faithfulness'), 0.0):.4f}</td>"
            "</tr>"
        )

    html_parts.extend(
        [
            "</table>",
            "</body>",
            "</html>",
        ]
    )

    return "\n".join(html_parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to eval_results_comprehensive.json")
    parser.add_argument("--branch", default="unknown")
    parser.add_argument("--commit", default="unknown")
    parser.add_argument("--repo", default="asaikiranb/RAG-climate")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as infile:
        payload = json.load(infile)

    print(build_html(payload, args.branch, args.commit, args.repo))

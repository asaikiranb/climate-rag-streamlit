"""Formats evaluation results as a structured HTML email for stakeholder reporting."""

import json
import argparse
from datetime import datetime


def metric_color(value: float) -> str:
    """Return a CSS color based on metric value."""
    if value >= 0.8:
        return "#2ea043"  # green
    elif value >= 0.6:
        return "#d29922"  # amber
    else:
        return "#cf222e"  # red


def build_html(data: dict, branch: str, commit: str, repo: str) -> str:
    config = data.get("config", {})
    ret = data.get("retrieval_metrics", {})
    gen = data.get("generation_metrics", {})
    cit = data.get("citation_metrics", {})
    lat = data.get("latency_summary", {})
    diff = data.get("difficulty_breakdown", {})

    commit_short = commit[:7]
    commit_url = f"https://github.com/{repo}/commit/{commit}"
    branch_url = f"https://github.com/{repo}/tree/{branch}"
    timestamp = data.get("timestamp", datetime.now().isoformat())

    html = f"""<!DOCTYPE html>
<html>
<head>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #24292f; max-width: 700px; margin: 0 auto; padding: 20px; }}
  h1 {{ font-size: 22px; border-bottom: 2px solid #d0d7de; padding-bottom: 8px; }}
  h2 {{ font-size: 16px; color: #57606a; margin-top: 24px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 8px 0 16px; font-size: 14px; }}
  th {{ background: #f6f8fa; text-align: left; padding: 8px 12px; border: 1px solid #d0d7de; font-weight: 600; }}
  td {{ padding: 8px 12px; border: 1px solid #d0d7de; }}
  .metric {{ font-weight: 600; font-family: monospace; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 600; color: white; }}
  .meta {{ font-size: 13px; color: #57606a; }}
  .footer {{ margin-top: 24px; padding-top: 12px; border-top: 1px solid #d0d7de; font-size: 12px; color: #8b949e; }}
</style>
</head>
<body>

<h1>📊 RAG Evaluation Results</h1>

<p class="meta">
  Branch: <a href="{branch_url}"><strong>{branch}</strong></a> &nbsp;|&nbsp;
  Commit: <a href="{commit_url}"><code>{commit_short}</code></a> &nbsp;|&nbsp;
  {timestamp[:19]} &nbsp;|&nbsp;
  {config.get('total_questions', 126)} questions &nbsp;|&nbsp;
  Reranker: {config.get('reranker', 'N/A')}
</p>

<h2>Retrieval Metrics</h2>
<table>
  <tr><th>k</th><th>Recall</th><th>MRR</th><th>NDCG</th></tr>"""

    for k in [1, 3, 5]:
        r = ret.get(f"recall@{k}", 0)
        m = ret.get(f"mrr@{k}", 0)
        n = ret.get(f"ndcg@{k}", 0)
        html += f"""
  <tr>
    <td><strong>{k}</strong></td>
    <td class="metric" style="color:{metric_color(r)}">{r:.4f}</td>
    <td class="metric" style="color:{metric_color(m)}">{m:.4f}</td>
    <td class="metric" style="color:{metric_color(n)}">{n:.4f}</td>
  </tr>"""

    html += """
</table>

<h2>Generation Metrics</h2>
<table>
  <tr><th>Metric</th><th>Score</th></tr>"""

    for key in ["faithfulness", "relevance", "completeness", "overall"]:
        v = gen.get(key, 0)
        html += f"""
  <tr>
    <td>{key.capitalize()}</td>
    <td class="metric" style="color:{metric_color(v)}">{v:.4f}</td>
  </tr>"""

    html += """
</table>

<h2>Citation Metrics</h2>
<table>
  <tr><th>Metric</th><th>Score</th></tr>"""

    for key in ["citation_validity", "citation_coverage", "citation_grounding"]:
        v = cit.get(key, 0)
        label = key.replace("citation_", "").capitalize()
        html += f"""
  <tr>
    <td>{label}</td>
    <td class="metric" style="color:{metric_color(v)}">{v:.4f}</td>
  </tr>"""

    html += """
</table>

<h2>Latency Summary</h2>
<table>
  <tr><th>Stage</th><th>Mean</th><th>P50</th><th>P95</th><th>Min</th><th>Max</th></tr>"""

    for stage in ["embed_ms", "search_ms", "rerank_ms", "generate_ms"]:
        s = lat.get(stage, {})
        html += f"""
  <tr>
    <td>{stage}</td>
    <td>{s.get('mean', 0):.1f}ms</td>
    <td>{s.get('p50', 0):.1f}ms</td>
    <td>{s.get('p95', 0):.1f}ms</td>
    <td>{s.get('min', 0):.1f}ms</td>
    <td>{s.get('max', 0):.1f}ms</td>
  </tr>"""

    html += """
</table>

<h2>By Difficulty</h2>
<table>
  <tr><th>Difficulty</th><th>Count</th><th>Top-1 Acc</th><th>Recall@5</th><th>MRR@5</th><th>NDCG@5</th><th>Faithfulness</th></tr>"""

    for d in ["Easy", "Medium", "Hard"]:
        dd = diff.get(d, {})
        acc1 = dd.get('recall@1', 0)
        html += f"""
  <tr>
    <td><strong>{d}</strong></td>
    <td>{dd.get('count', 0)}</td>
    <td class="metric" style="color:{metric_color(acc1)}">{acc1:.1%}</td>
    <td class="metric" style="color:{metric_color(dd.get('recall@5', 0))}">{dd.get('recall@5', 0):.4f}</td>
    <td class="metric">{dd.get('mrr@5', 0):.4f}</td>
    <td class="metric">{dd.get('ndcg@5', 0):.4f}</td>
    <td class="metric" style="color:{metric_color(dd.get('faithfulness', 0))}">{dd.get('faithfulness', 0):.4f}</td>
  </tr>"""

    html += f"""
</table>

<p class="footer">
  Full results attached as JSON. View the run at
  <a href="https://github.com/{repo}/actions">GitHub Actions</a>.
</p>

</body>
</html>"""

    return html


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to eval_results_comprehensive.json")
    parser.add_argument("--branch", default="unknown")
    parser.add_argument("--commit", default="unknown")
    parser.add_argument("--repo", default="asaikiranb/RAG-climate")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    print(build_html(data, args.branch, args.commit, args.repo))

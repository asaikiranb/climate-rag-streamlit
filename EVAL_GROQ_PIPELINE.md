## Groq Offline Evaluation Pipeline

This repository now includes an additive offline evaluation pipeline that mirrors the requested retrieval methodology from the `contextual-hvac-rag` eval flow, but it runs against this repository's existing local RAG stack:

- query expansion via `query.expand_query`
- hybrid retrieval via `HybridRetriever`
- reranking via `CrossEncoderReranker`
- answer generation via Groq when `GROQ_API_KEY` is set

If `GROQ_API_KEY` is not set, retrieval still runs and `answer_text` is emitted as an empty string.

### CSV Contract

The input file must be a CSV with this exact header order:

1. `Question`
2. `gold_sources`
3. `metadata`
4. `page_range`
5. `anchor_text`

Notes:

- `page_range` is parsed safely from string literals such as `[19]` or `[29,30,31]`.
- `metadata` is read as JSON when possible. `difficulty` is pulled from `metadata["difficulty"]` and defaults to `Unknown`.
- `question_id` is stable per row and built from the 1-based row index plus a hash of the question text.

### Missing Gold Policy

Rows are always queried and logged, even when gold annotations are incomplete.

- DOC scoring is skipped when `gold_sources` is blank.
- PAGE scoring is skipped when `gold_sources` is blank, or when both `page_range` and `anchor_text` are blank.
- When scoring is skipped, the corresponding `*_scored` field is `false`, the metric fields are `null`, and the row is excluded from the aggregate for that metric family.

### Metric Definitions

The evaluator computes metrics for `k in {1,3,5,10}`.

DOC:

- `Recall@k`: `1` if the first matching normalized filename appears in the top-`k`, else `0`
- `MRR@10`: reciprocal rank of the first matching normalized filename in the top 10
- `nDCG@k`: binary gain for the first matching normalized filename

PAGE:

- exact hit: same normalized filename and retrieved page is in `page_range`
- fallback hit: if exact page does not hit, fuzzy match `anchor_text` against the retrieved snippet
- `Recall@k`: `1` if an exact-page or anchor fallback hit appears in the top-`k`, else `0`
- `MRR@10`: reciprocal rank of the first exact-page or anchor fallback hit in the top 10
- `nDCG@k`: graded relevance with partial credit

### Page nDCG Partial Credit

Page nDCG intentionally preserves partial credit:

- `rel=2`: exact page hit or anchor-text fallback hit
- `rel=1`: correct document, but wrong page or unknown page
- `rel=0`: incorrect document

The implementation uses linear graded gain, so correct-document / wrong-page retrievals still improve page nDCG without counting as page hits.

### Direct ACL Eval Mode

The original Contextual API-specific direct ACL mode does not apply in this repository. This add-on evaluator does not call `agents/{id}/query` or `agents/{id}/query/acl`; it evaluates the local Groq-backed RAG pipeline instead.

### Outputs

Running the evaluator writes:

- `per_query.jsonl`
- `summary.json`
- a compact console summary

The summary includes:

- aggregated DOC and PAGE retrieval metrics
- `metric_notes`, including the page nDCG graded-relevance explanation
- `by_difficulty`
- `by_gold_sources` and `by_gold_source` (compat alias)
- latency summaries
- placeholder `index_stats`

### Run Command

Windows wrapper:

```powershell
.\contextual-hvac-rag.cmd eval --input .\golden.csv --out .\eval_out --top-k 10
```

Python entrypoint:

```powershell
python .\contextual_hvac_rag.py eval --input .\golden.csv --out .\eval_out --top-k 10
```

Smoke test example:

```powershell
python .\contextual_hvac_rag.py eval --input .\golden.csv --out .\eval_out_smoke --top-k 10 --limit 5
```

### Caveat

Anchor-text fallback depends on the retrieved snippet text. If a retrieved hit has no snippet, or the chunk text omits the gold anchor span, PAGE fallback matching can miss even when the correct document is retrieved.

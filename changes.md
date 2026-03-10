# Changes from v1 to v4 (Text Pipeline)

## Ingestion / Indexing
- New ingestion pipeline is page-aware with OCR fallback and section splitting (`ingest_v2.py:152`, `ingest_v2.py:167`).
- Chunk IDs are deterministic and filename-aware (`pipeline_utils.py:13`, `pipeline_utils.py:21`, `ingest_v2.py:265`) to prevent cross-file collisions.
- Hash-registry upsert flow:
  - skip unchanged docs (`ingest_v2.py:354`),
  - delete prior chunks when changed (`ingest_v2.py:357`),
  - re-upsert cleanly (`ingest_v2.py:365`).
- Qdrant collection creation/upsert added (`ingest_v2.py:281`, `ingest_v2.py:338`).

## Retrieval
- New retriever class supports `chroma|qdrant` and sparse modes (`retrieve_v2.py:251`).
- BM42 sparse branch added (improved BM25 + overlap blending) (`retrieve_v2.py:462`).
- Metadata priors added (title/section/filename/page boosts) (`retrieve_v2.py:496`).
- Final weighted fusion uses dense+sparse+metadata contributions (`retrieve_v2.py:601`).

## Reranking
- Reranker now uses a single-stage calibrated CE scoring path before fusion (`rerank_v2.py:162`).

## Scoring + Reporting
- Uses golden CSV contract with manifest + per-query outputs (`.github/workflows/eval-ci.yml:55`, `.github/workflows/eval-ci.yml:70`).
- Eval scoring includes doc + page scoring in normalized contract (`eval/run.py:224`).
- CI output includes:
  - `summary.json`,
  - `per_query.jsonl`,
  - `eval_manifest.json`,
  - `eval_results_comprehensive.json` (`.github/workflows/eval-ci.yml:69`).

## Model / Backend Continuity
- Embedding model remains `BAAI/bge-m3`.
- Reranking model remains `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Backend is now implemented on Qdrant for v4 CI path (instead of Chroma in v1 CI path).

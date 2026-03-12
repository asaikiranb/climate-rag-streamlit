"""Entry point for v2 page-aware ingestion (Chroma or Qdrant)."""

from ingestion.ingest_v2 import main

if __name__ == "__main__":
    raise SystemExit(main())

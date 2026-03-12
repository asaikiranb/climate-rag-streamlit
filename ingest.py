"""Entry point for original document ingestion (ChromaDB)."""

from ingestion.ingest import IngestionPipeline

if __name__ == "__main__":
    IngestionPipeline().run("./Eval Dataset")

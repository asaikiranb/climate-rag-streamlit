"""Improved ingestion pipeline with page-aware chunking and hash-based upserts."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
import pytesseract
import tiktoken
from PIL import Image, ImageFilter
from sentence_transformers import SentenceTransformer

try:
    import chromadb
except Exception:  # pragma: no cover
    chromadb = None

try:
    from qdrant_client import QdrantClient, models
except Exception:  # pragma: no cover
    QdrantClient = None
    models = None

from config import (
    CHROMA_COLLECTION,
    CHROMA_PATH,
    INGEST_CHUNK_OVERLAP,
    INGEST_CHUNK_SIZE,
    INGEST_DOC_REGISTRY_PATH,
    INGEST_EMBEDDING_MODEL,
    INGEST_MIN_TEXT_CHARS,
    INGEST_SOURCE_DIR,
    MODEL_DEVICE,
    QDRANT_COLLECTION,
    QDRANT_PATH,
    VECTOR_DB_BACKEND,
)
from hf_local import resolve_local_snapshot
from pipeline_utils import chunk_id_to_uuid, normalize_whitespace, stable_chunk_id

TESSERACT_CONFIG = r"--oem 3 --psm 6"
OCR_DPI = 300
HEADING_PATTERN = re.compile(r"^(?:\d+(?:\.\d+)*[.)-]?\s+)?[A-Z][A-Z0-9\s,/:()\-&]{6,}$")


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    chunk_id: str
    text: str
    metadata: dict[str, Any]

class IngestionPipelineV2:
    """Builds deterministic chunks and upserts them into Chroma or Qdrant."""

    def __init__(
        self,
        backend: str = VECTOR_DB_BACKEND,
        source_dir: str = INGEST_SOURCE_DIR,
        embedding_model: str = INGEST_EMBEDDING_MODEL,
        chunk_size: int = INGEST_CHUNK_SIZE,
        chunk_overlap: int = INGEST_CHUNK_OVERLAP,
        min_text_chars: int = INGEST_MIN_TEXT_CHARS,
        chroma_path: str = CHROMA_PATH,
        chroma_collection: str = CHROMA_COLLECTION,
        qdrant_path: str = QDRANT_PATH,
        qdrant_collection: str = QDRANT_COLLECTION,
        doc_registry_path: str = INGEST_DOC_REGISTRY_PATH,
    ) -> None:
        self.backend = backend.strip().lower()
        if self.backend not in {"chroma", "qdrant"}:
            raise ValueError(f"Unsupported backend: {backend}")

        self.source_dir = Path(source_dir)
        self.embedding_model = embedding_model
        self.chunk_size = max(64, int(chunk_size))
        self.chunk_overlap = max(0, int(chunk_overlap))
        self.min_text_chars = max(1, int(min_text_chars))
        self.chroma_path = chroma_path
        self.chroma_collection_name = chroma_collection
        self.qdrant_path = qdrant_path
        self.qdrant_collection_name = qdrant_collection
        self.doc_registry_path = Path(doc_registry_path)

        self.tokenizer = self._load_tokenizer()
        self.encoder = self._load_encoder(self.embedding_model)
        self.registry = self._load_registry()

        self.chroma_collection = None
        self.qdrant_client = None
        self.vector_dim = None

        if self.backend == "chroma":
            if chromadb is None:
                raise ImportError("chromadb is required for backend='chroma'")
            db = chromadb.PersistentClient(path=self.chroma_path)
            self.chroma_collection = db.get_or_create_collection(
                name=self.chroma_collection_name,
                metadata={"description": "HVAC technical chunks (v2)"},
            )
        else:
            if QdrantClient is None or models is None:
                raise ImportError("qdrant-client is required for backend='qdrant'")
            self.qdrant_client = QdrantClient(path=self.qdrant_path)

    def _load_encoder(self, model_name: str) -> SentenceTransformer:
        token = os.environ.get("HF_TOKEN")
        local_snapshot = resolve_local_snapshot(model_name)
        if local_snapshot:
            try:
                return SentenceTransformer(
                    local_snapshot,
                    token=token,
                    local_files_only=True,
                    device=MODEL_DEVICE,
                )
            except Exception:
                pass
        return SentenceTransformer(model_name, token=token, device=MODEL_DEVICE)

    def _load_tokenizer(self):
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None

    def _load_registry(self) -> dict[str, Any]:
        if not self.doc_registry_path.exists():
            return {}
        try:
            return json.loads(self.doc_registry_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_registry(self) -> None:
        self.doc_registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.doc_registry_path.write_text(json.dumps(self.registry, indent=2), encoding="utf-8")

    def _ocr_page(self, page: fitz.Page) -> str:
        pix = page.get_pixmap(dpi=OCR_DPI)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
        img = img.filter(ImageFilter.SHARPEN)
        return pytesseract.image_to_string(img, config=TESSERACT_CONFIG)

    def _extract_pages(self, path: Path) -> list[dict[str, Any]]:
        doc = fitz.open(path)
        pages: list[dict[str, Any]] = []
        try:
            for idx, page in enumerate(doc, start=1):
                text = page.get_text() or ""
                if len(text.strip()) < self.min_text_chars:
                    text = self._ocr_page(page)
                normalized = normalize_whitespace(text)
                if normalized:
                    pages.append({"page_number": idx, "text": normalized})
        finally:
            doc.close()
        return pages

    def _split_sections(self, page_text: str) -> list[tuple[str, str]]:
        blocks = [blk.strip() for blk in re.split(r"\n\s*\n", page_text) if blk.strip()]
        sections: list[tuple[str, str]] = []
        current_title = ""
        current_lines: list[str] = []

        def flush() -> None:
            text = normalize_whitespace("\n".join(current_lines))
            if text:
                sections.append((current_title or "General", text))

        for block in blocks:
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            if not lines:
                continue
            first_line = lines[0]
            if HEADING_PATTERN.match(first_line):
                if current_lines:
                    flush()
                    current_lines = []
                current_title = normalize_whitespace(first_line)
                remainder = lines[1:]
                if remainder:
                    current_lines.extend(remainder)
                continue
            current_lines.extend(lines)

        if current_lines:
            flush()

        if not sections:
            sections.append(("General", page_text))
        return sections

    def _chunk_section_text(self, text: str) -> list[str]:
        stride = max(1, self.chunk_size - self.chunk_overlap)
        if self.tokenizer is None:
            words = text.split()
            if not words:
                return []
            out_words: list[str] = []
            for start in range(0, len(words), stride):
                end = start + self.chunk_size
                chunk_text = normalize_whitespace(" ".join(words[start:end]))
                if chunk_text:
                    out_words.append(chunk_text)
                if end >= len(words):
                    break
            return out_words

        tokens = self.tokenizer.encode(text)
        if not tokens:
            return []
        out_tokens: list[str] = []
        for start in range(0, len(tokens), stride):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = normalize_whitespace(self.tokenizer.decode(chunk_tokens))
            if chunk_text:
                out_tokens.append(chunk_text)
            if end >= len(tokens):
                break
        return out_tokens

    def _doc_hash(self, pages: list[dict[str, Any]]) -> str:
        joined = "\n".join(page["text"] for page in pages)
        normalized = normalize_whitespace(joined)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _index_signature(self) -> str:
        collection_name = self.chroma_collection_name if self.backend == "chroma" else self.qdrant_collection_name
        payload = ":".join(
            [
                self.backend,
                collection_name,
                self.embedding_model,
                str(self.chunk_size),
                str(self.chunk_overlap),
            ]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _title_from_filename(self, filename: str) -> str:
        stem = filename.rsplit(".pdf", 1)[0]
        if "_" in stem and stem.split("_", 1)[0].isdigit():
            stem = stem.split("_", 1)[1]
        return normalize_whitespace(stem.replace("_", " ").replace("-", " "))

    def _build_chunks(self, filename: str, source_path: str, pages: list[dict[str, Any]], doc_hash: str) -> list[ChunkRecord]:
        title = self._title_from_filename(filename)
        records: list[ChunkRecord] = []

        for page in pages:
            page_number = int(page["page_number"])
            sections = self._split_sections(page["text"])
            for section_idx, (section_title, section_text) in enumerate(sections, start=1):
                chunk_texts = self._chunk_section_text(section_text)
                for chunk_idx, chunk_text in enumerate(chunk_texts, start=1):
                    cid = stable_chunk_id(doc_hash, page_number, section_idx, chunk_idx)
                    metadata = {
                        "filename": filename,
                        "source_path": source_path,
                        "doc_hash": doc_hash,
                        "page_number": page_number,
                        "section_title": section_title,
                        "chunk_idx": chunk_idx,
                        "title": title,
                        "embedding_model": self.embedding_model,
                        "chunk_id": cid,
                        "backend": self.backend,
                    }
                    records.append(ChunkRecord(chunk_id=cid, text=chunk_text, metadata=metadata))
        return records

    def _ensure_qdrant_collection(self, vector_dim: int) -> None:
        assert self.qdrant_client is not None
        existing = {c.name for c in self.qdrant_client.get_collections().collections}
        if self.qdrant_collection_name not in existing:
            self.qdrant_client.create_collection(
                collection_name=self.qdrant_collection_name,
                vectors_config=models.VectorParams(size=vector_dim, distance=models.Distance.COSINE),
            )

    def _delete_existing_doc(self, filename: str) -> None:
        if self.backend == "chroma":
            assert self.chroma_collection is not None
            self.chroma_collection.delete(where={"filename": filename})
            return

        assert self.qdrant_client is not None
        filename_filter = models.Filter(
            must=[models.FieldCondition(key="filename", match=models.MatchValue(value=filename))]
        )
        self.qdrant_client.delete(
            collection_name=self.qdrant_collection_name,
            points_selector=models.FilterSelector(filter=filename_filter),
            wait=True,
        )

    def _upsert_chunks(self, records: list[ChunkRecord]) -> None:
        if not records:
            return

        texts = [record.text for record in records]
        enriched = [
            f"Title: {record.metadata.get('title', '')}\nSection: {record.metadata.get('section_title', '')}\n\n{record.text}"
            for record in records
        ]
        vectors = self.encoder.encode(enriched, show_progress_bar=False, batch_size=16)
        vector_dim = int(len(vectors[0]))

        if self.backend == "chroma":
            assert self.chroma_collection is not None
            self.chroma_collection.upsert(
                ids=[record.chunk_id for record in records],
                documents=texts,
                embeddings=[vector.tolist() for vector in vectors],
                metadatas=[record.metadata for record in records],
            )
            return

        self._ensure_qdrant_collection(vector_dim)
        assert self.qdrant_client is not None
        points = [
            models.PointStruct(
                id=chunk_id_to_uuid(record.chunk_id),
                vector=vectors[idx].tolist(),
                payload={**record.metadata, "document": record.text},
            )
            for idx, record in enumerate(records)
        ]
        self.qdrant_client.upsert(collection_name=self.qdrant_collection_name, points=points, wait=True)

    def ingest_file(self, path: Path) -> dict[str, Any]:
        filename = path.name
        pages = self._extract_pages(path)
        if not pages:
            return {"filename": filename, "status": "skipped_empty", "chunks": 0}

        doc_hash = self._doc_hash(pages)
        index_signature = self._index_signature()
        existing = self.registry.get(filename)
        if (
            existing
            and existing.get("doc_hash") == doc_hash
            and existing.get("index_signature") == index_signature
        ):
            return {"filename": filename, "status": "skipped_unchanged", "chunks": int(existing.get("chunks", 0))}

        if existing:
            self._delete_existing_doc(filename)

        records = self._build_chunks(
            filename=filename,
            source_path=str(path),
            pages=pages,
            doc_hash=doc_hash,
        )
        self._upsert_chunks(records)

        self.registry[filename] = {
            "doc_hash": doc_hash,
            "index_signature": index_signature,
            "chunks": len(records),
            "pages": len(pages),
            "source_path": str(path),
            "backend": self.backend,
            "embedding_model": self.embedding_model,
        }
        self._save_registry()
        return {"filename": filename, "status": "indexed", "chunks": len(records)}

    def run(self) -> list[dict[str, Any]]:
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")

        files = sorted(path for path in self.source_dir.iterdir() if path.suffix.lower() == ".pdf")
        results: list[dict[str, Any]] = []
        for path in files:
            print(f"Ingesting {path.name}...")
            outcome = self.ingest_file(path)
            print(f"  -> {outcome['status']} ({outcome['chunks']} chunks)")
            results.append(outcome)
        return results

    def close(self) -> None:
        if self.qdrant_client is not None:
            try:
                self.qdrant_client.close()
            except Exception:
                pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python ingest_v2.py")
    parser.add_argument("--source-dir", default=INGEST_SOURCE_DIR)
    parser.add_argument("--backend", default=VECTOR_DB_BACKEND, choices=["chroma", "qdrant"])
    parser.add_argument("--embedding-model", default=INGEST_EMBEDDING_MODEL)
    parser.add_argument("--chunk-size", type=int, default=INGEST_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=INGEST_CHUNK_OVERLAP)
    parser.add_argument("--registry-path", default=INGEST_DOC_REGISTRY_PATH)
    parser.add_argument("--chroma-path", default=CHROMA_PATH)
    parser.add_argument("--chroma-collection", default=CHROMA_COLLECTION)
    parser.add_argument("--qdrant-path", default=QDRANT_PATH)
    parser.add_argument("--qdrant-collection", default=QDRANT_COLLECTION)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pipeline = IngestionPipelineV2(
        backend=args.backend,
        source_dir=args.source_dir,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        doc_registry_path=args.registry_path,
        chroma_path=args.chroma_path,
        chroma_collection=args.chroma_collection,
        qdrant_path=args.qdrant_path,
        qdrant_collection=args.qdrant_collection,
    )
    outcomes = pipeline.run()
    indexed = sum(1 for row in outcomes if row["status"] == "indexed")
    skipped = len(outcomes) - indexed
    print(f"Ingestion complete: indexed={indexed}, skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

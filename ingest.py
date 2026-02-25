import os
import io
import fitz  # PyMuPDF
import chromadb
import tiktoken
from dotenv import load_dotenv
from typing import List, Dict, Tuple
import uuid

from PIL import Image, ImageFilter
import pytesseract

load_dotenv()

# ---------------------------------------------------------------------------
# OCR Configuration
# ---------------------------------------------------------------------------
# Tesseract OCR mode 3 = default (auto-detect), PSM 6 = block of text
TESSERACT_CONFIG = r"--oem 3 --psm 6"
OCR_DPI = 300          # render at 300 DPI for high-quality OCR
MIN_TEXT_CHARS = 50    # pages with fewer chars than this get OCR treatment


class PDFIngestion:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Embedding: BAAI/bge-m3 (1024-dim) via local SentenceTransformer.
        # bge-m3 is NOT on the HF free Inference API (returns 403), so we always
        # use local weights. Passing HF_TOKEN allows authenticated model downloads
        # in CI. MUST match the embedding model used in retrieve.py.
        hf_token = os.environ.get("HF_TOKEN")
        self.embedding_model_id = "BAAI/bge-m3"
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer(self.embedding_model_id, token=hf_token or None)
        print(f"Embedding model: {self.embedding_model_id} (local SentenceTransformer)")

        # Initialize local ChromaDB client
        chroma_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)

        # Get or create collection
        self.collection_name = os.getenv("CHROMA_COLLECTION_NAME", "hvac_documents")
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            print(f"Using existing collection: {self.collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "HVAC technical documents"}
            )
            print(f"Created new collection: {self.collection_name}")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 128) -> List[str]:
        """Split text into token-counted chunks with overlap.

        Smaller chunks (512 tokens) produce sharper, more discriminative
        embeddings — critical for matching specific factual queries.
        """
        tokens = self.tokenizer.encode(text)
        chunks = []

        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            start = end - overlap

            if end >= len(tokens):
                break

        return chunks

    # ------------------------------------------------------------------
    # OCR helpers
    # ------------------------------------------------------------------
    def _ocr_page(self, page: fitz.Page) -> str:
        """Run Tesseract OCR on a rendered page image.

        Pipeline:
          1. Render page at 300 DPI (high resolution for accurate OCR)
          2. Convert to grayscale (reduces noise)
          3. Apply sharpening filter (improves character edges)
          4. Run Tesseract in best-accuracy mode (OEM 3, PSM 6)
        """
        pix = page.get_pixmap(dpi=OCR_DPI)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        # Preprocessing: grayscale + sharpen for cleaner OCR
        img = img.convert("L")  # grayscale
        img = img.filter(ImageFilter.SHARPEN)

        text = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
        return text

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text from each page of a PDF, with OCR fallback.

        For each page:
          - First try PyMuPDF text extraction (fast, high quality for digital PDFs)
          - If < 50 chars extracted, fall back to Tesseract OCR (handles scanned/image PDFs)

        This dual strategy ensures we capture text from ALL document types:
        digital PDFs, scanned documents, posters, catalogs, and standards.
        """
        doc = fitz.open(pdf_path)
        pages_data = []
        ocr_pages = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()

            # OCR fallback for scanned / image-based pages
            if len(text.strip()) < MIN_TEXT_CHARS:
                ocr_text = self._ocr_page(page)
                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
                    ocr_pages += 1

            if text.strip():
                pages_data.append({
                    'text': text,
                    'page_number': page_num + 1,
                    'filename': os.path.basename(pdf_path)
                })

        if ocr_pages > 0:
            print(f"  OCR applied to {ocr_pages}/{len(doc)} pages")

        doc.close()
        return pages_data

    def _build_doc_title(self, filename: str) -> str:
        """Convert filename to a readable document title for context enrichment.

        '0034_HPMP-Poster-1.pdf' → 'HPMP Poster 1'
        'ISO-5149-2.pdf' → 'ISO 5149 2'
        """
        title = filename.replace('.pdf', '')
        # Strip leading numeric prefix (e.g., '0034_')
        parts = title.split('_', 1)
        if len(parts) > 1 and parts[0].isdigit():
            title = parts[1]
        # Clean up separators
        title = title.replace('-', ' ').replace('_', ' ')
        return title

    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text from a PDF, chunk it, return chunks with metadata.

        Strategy:
          1. Extract text from ALL pages (with OCR fallback)
          2. Concatenate all page text into a single document string
             (preserves cross-page context that per-page chunking misses)
          3. Chunk the full document text (512 tokens, 128 overlap)
          4. Map each chunk back to its source page via character offsets
        """
        print(f"Processing: {pdf_path}")
        pages_data = self.extract_text_from_pdf(pdf_path)

        if not pages_data:
            print(f"  No text extracted from {pdf_path}")
            return []

        filename = os.path.basename(pdf_path)

        # Build page offset map: [(start_char, end_char, page_number), ...]
        page_offsets: List[Tuple[int, int, int]] = []
        full_text_parts = []
        offset = 0
        for page_data in pages_data:
            text = page_data['text']
            start = offset
            full_text_parts.append(text)
            offset += len(text) + 1  # +1 for the newline separator
            page_offsets.append((start, offset - 1, page_data['page_number']))

        full_text = "\n".join(full_text_parts)

        # Chunk the full document text
        chunks = self.chunk_text(full_text)

        # Map each chunk to its primary page
        all_chunks = []
        char_pos = 0
        for chunk_idx, chunk in enumerate(chunks):
            # Find the chunk in full_text to determine page
            chunk_start = full_text.find(chunk[:100], max(0, char_pos - 200))
            if chunk_start == -1:
                chunk_start = char_pos

            # Find which page this chunk primarily belongs to
            page_number = pages_data[0]['page_number']  # default to first page
            for start, end, pn in page_offsets:
                if start <= chunk_start < end:
                    page_number = pn
                    break

            char_pos = chunk_start + len(chunk) // 2  # advance for next search

            all_chunks.append({
                'text': chunk,
                'filename': filename,
                'page_number': page_number,
                'chunk_index': chunk_idx,
            })

        print(f"  Created {len(all_chunks)} chunks from {len(pages_data)} pages")
        return all_chunks

    def ingest_documents(self, data_folder: str = "./data"):
        """Ingest all PDFs from data_folder into ChromaDB.

        Each chunk is embedded with document-context enrichment:
        the document title is prepended to the chunk text before embedding,
        so the embedding captures document identity. The raw chunk text
        (without the title prefix) is stored in ChromaDB for LLM context.
        """
        pdf_files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]

        if not pdf_files:
            print(f"No PDF files found in {data_folder}")
            return

        print(f"Found {len(pdf_files)} PDF files")

        all_chunks = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(data_folder, pdf_file)
            chunks = self.process_pdf(pdf_path)
            all_chunks.extend(chunks)

        print(f"\nTotal chunks to ingest: {len(all_chunks)}")

        texts = [chunk['text'] for chunk in all_chunks]
        metadatas = [
            {
                'filename': chunk['filename'],
                'page_number': str(chunk['page_number']),
                'chunk_index': str(chunk['chunk_index'])
            }
            for chunk in all_chunks
        ]
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]

        # Document-context enriched embeddings:
        # Prepend document title so the embedding captures source identity.
        # This dramatically helps source-specific queries like
        # "According to the HPMP poster..." or "What does ISO 5149 say..."
        print(f"Generating document-context enriched embeddings with {self.embedding_model_id}...")
        embeddings = []
        for i, chunk in enumerate(all_chunks):
            doc_title = self._build_doc_title(chunk['filename'])
            enriched_text = f"Document: {doc_title}\n\n{chunk['text']}"
            emb = self.embedding_model.encode(enriched_text).tolist()
            embeddings.append(emb)
            if (i + 1) % 50 == 0 or (i + 1) == len(all_chunks):
                print(f"  Embedded {i + 1}/{len(all_chunks)} chunks...")

        batch_size = 100
        for i in range(0, len(texts), batch_size):
            end_idx = min(i + batch_size, len(texts))
            print(f"Ingesting batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

            self.collection.add(
                documents=texts[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )

        print(f"\nSuccessfully ingested {len(texts)} chunks into ChromaDB!")
        print(f"Collection: {self.collection_name}")

    def get_collection_stats(self):
        """Print collection stats."""
        count = self.collection.count()
        print(f"\nCollection Statistics:")
        print(f"  Name: {self.collection_name}")
        print(f"  Total chunks: {count}")

        if count > 0:
            results = self.collection.get(limit=count)
            filenames = set(meta['filename'] for meta in results['metadatas'])
            print(f"  Unique documents: {len(filenames)}")
            print(f"  Documents: {', '.join(sorted(filenames))}")


def main():
    print("=" * 60)
    print("HVAC RAG System - PDF Ingestion")
    print("=" * 60)

    ingestion = PDFIngestion()

    current_count = ingestion.collection.count()
    if current_count > 0:
        print(f"\nWarning: Collection already contains {current_count} chunks")
        response = input("Do you want to continue and add more? (yes/no): ").strip().lower()
        if response != 'yes':
            print("Ingestion cancelled.")
            ingestion.get_collection_stats()
            return

    ingestion.ingest_documents("./Eval Dataset")

    ingestion.get_collection_stats()


if __name__ == "__main__":
    main()

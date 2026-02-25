import os
import fitz  # PyMuPDF
import chromadb
import tiktoken
from dotenv import load_dotenv
from typing import List, Dict
import uuid
import numpy as np

load_dotenv()


class PDFIngestion:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Embedding: BAAI/bge-m3 (1024-dim) via HF Inference API when HF_TOKEN is
        # available; local SentenceTransformer fallback otherwise.
        # MUST match the embedding model used in retrieve.py.
        hf_token = os.environ.get("HF_TOKEN")
        self.embedding_model_id = "BAAI/bge-m3"
        if hf_token:
            from huggingface_hub import InferenceClient
            self._hf_client = InferenceClient(token=hf_token)
            self._use_api_embedding = True
            print(f"Embedding model: {self.embedding_model_id} via HF Inference API")
        else:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(self.embedding_model_id)
            self._use_api_embedding = False
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

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into token-counted chunks with overlap."""
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

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text from each page of a PDF."""
        doc = fitz.open(pdf_path)
        pages_data = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()

            if text.strip():
                pages_data.append({
                    'text': text,
                    'page_number': page_num + 1,
                    'filename': os.path.basename(pdf_path)
                })

        doc.close()
        return pages_data

    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text from a PDF, chunk it, return chunks with metadata."""
        print(f"Processing: {pdf_path}")
        pages_data = self.extract_text_from_pdf(pdf_path)

        all_chunks = []
        for page_data in pages_data:
            chunks = self.chunk_text(page_data['text'])

            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append({
                    'text': chunk,
                    'filename': page_data['filename'],
                    'page_number': page_data['page_number'],
                    'chunk_index': chunk_idx
                })

        print(f"  Created {len(all_chunks)} chunks from {len(pages_data)} pages")
        return all_chunks

    def ingest_documents(self, data_folder: str = "./data"):
        """Ingest all PDFs from data_folder into ChromaDB."""
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

        print(f"Generating embeddings with {self.embedding_model_id}...")
        embeddings = []
        for i, text in enumerate(texts):
            if self._use_api_embedding:
                emb = np.array(self._hf_client.feature_extraction(text, model=self.embedding_model_id)).tolist()
            else:
                emb = self.embedding_model.encode(text).tolist()
            embeddings.append(emb)
            if (i + 1) % 50 == 0 or (i + 1) == len(texts):
                print(f"  Embedded {i + 1}/{len(texts)} chunks...")


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

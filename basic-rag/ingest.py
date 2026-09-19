"""
ingest.py — Basic RAG: Step 1
Reads a PDF, splits it into overlapping chunks, embeds each chunk,
and stores everything in a local Chroma vector database.

Run this once (or whenever the source PDF changes):
    python ingest.py --pdf data/python-handbook.pdf
"""

import argparse
import os

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---- Config ----
CHUNK_SIZE = 800       # characters per chunk (roughly ~150-200 words)
CHUNK_OVERLAP = 150    # characters shared between consecutive chunks
EMBED_MODEL = "all-MiniLM-L6-v2"

# Anchor paths to this file's own folder, not the current working directory.
# Streamlit Cloud runs from the repo root, so a plain "chroma_db" string
# would end up in the wrong place otherwise.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
DEFAULT_PDF_PATH = os.path.join(BASE_DIR, "data", "python-handbook.pdf")
COLLECTION_NAME = "python_handbook"


def extract_pages(pdf_path: str) -> list[dict]:
    """Returns a list of {page_number, text} for every non-empty page."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({"page_number": i, "text": text})
    return pages


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Simple sliding-window chunker over raw characters.
    Walks the text in windows of `chunk_size`, stepping forward by
    (chunk_size - overlap) each time, so consecutive chunks share
    `overlap` characters of context. This is intentionally simple
    (no external splitter library) so you understand exactly how
    your chunks are formed.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += step
    return chunks


def build_chunks(pages: list[dict]) -> list[dict]:
    """Chunk every page's text, keeping track of which page each chunk came from."""
    all_chunks = []
    for page in pages:
        page_chunks = chunk_text(page["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for j, chunk in enumerate(page_chunks):
            all_chunks.append({
                "id": f"page{page['page_number']}_chunk{j}",
                "text": chunk,
                "page_number": page["page_number"],
            })
    return all_chunks


def main(pdf_path: str):
    print(f"Reading PDF: {pdf_path}")
    pages = extract_pages(pdf_path)
    print(f"Extracted text from {len(pages)} non-empty pages")

    chunks = build_chunks(pages)
    print(f"Split into {len(chunks)} chunks "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    print(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("Embedding chunks (this runs locally, no API calls)...")
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    print(f"Storing in Chroma at ./{DB_PATH}")
    client = chromadb.PersistentClient(path=DB_PATH)
    # Fresh collection each run so re-ingesting doesn't duplicate data
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"page_number": c["page_number"]} for c in chunks],
    )

    print(f"Done. {collection.count()} chunks stored in collection "
          f"'{COLLECTION_NAME}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf", default=DEFAULT_PDF_PATH, help="Path to the source PDF"
    )
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        raise FileNotFoundError(f"PDF not found: {args.pdf}")

    main(args.pdf)

"""
query.py — Basic RAG: Step 2
Given a user question:
  1. Embed the question with the same local model used in ingest.py
  2. Retrieve the top-k most similar chunks from Chroma
  3. Stuff those chunks into a prompt
  4. Call an LLM (via Groq) to generate a grounded answer

This is deliberately the *simplest possible* RAG pipeline — one
retrieval pass, no re-ranking, no query rewriting. It's the baseline
the Advanced RAG project will be compared against.
"""

import os

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBED_MODEL = "all-MiniLM-L6-v2"
DB_PATH = "chroma_db"
COLLECTION_NAME = "python_handbook"
LLM_MODEL = "llama-3.3-70b-versatile"
TOP_K = 4

_embedder = None
_collection = None
_groq_client = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=DB_PATH)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Create a .env file with "
                "GROQ_API_KEY=your_key (get a free key at console.groq.com)"
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """Embed the question and pull the top_k most similar chunks."""
    embedder = _get_embedder()
    collection = _get_collection()

    query_embedding = embedder.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    chunks = []
    for text, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": text,
            "page_number": meta["page_number"],
            "distance": distance,
        })
    return chunks


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Page {c['page_number']}]\n{c['text']}" for c in chunks
    )
    return f"""You are a helpful assistant answering questions about "The Python Handbook".
Answer ONLY using the context below. If the context doesn't contain the
answer, say you don't know rather than guessing.

Context:
{context}

Question: {question}

Answer:"""


def answer(question: str, top_k: int = TOP_K) -> dict:
    """Full pipeline: retrieve, build prompt, generate. Returns answer + sources."""
    chunks = retrieve(question, top_k)
    prompt = build_prompt(question, chunks)

    client = _get_groq_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": [{"page_number": c["page_number"]} for c in chunks],
    }


if __name__ == "__main__":
    # Quick CLI test
    q = input("Ask a question about the Python Handbook: ")
    result = answer(q)
    print("\n--- Answer ---")
    print(result["answer"])
    print("\n--- Sources ---")
    for s in result["sources"]:
        print(f"Page {s['page_number']}")
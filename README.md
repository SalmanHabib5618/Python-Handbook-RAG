# Basic RAG — Ask the Python Handbook

A minimal Retrieval-Augmented Generation pipeline over *The Python
Handbook* by Flavio Copes. This is the baseline project in a two-part
series comparing a naive RAG pipeline against an advanced/agentic one
(see `../advanced-rag`).

## How it works

```
PDF → extract text per page → chunk (800 chars, 150 overlap)
    → embed chunks locally (all-MiniLM-L6-v2)
    → store in Chroma (local vector DB)

User question → embed question → similarity search (top-4 chunks)
    → stuff chunks into prompt → Llama 3.3 (via Groq) generates answer
```

One retrieval pass, no re-ranking, no query rewriting — deliberately
simple. This is the baseline the Advanced RAG project improves on.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Get a free Groq API key at [console.groq.com](https://console.groq.com),
   then create a `.env` file:
   ```bash
   cp .env.example .env
   # edit .env and paste your key
   ```

3. Put the source PDF in `data/` (e.g. `data/python-handbook.pdf`).

4. Run ingestion (builds the vector database — do this once):
   ```bash
   python ingest.py --pdf data/python-handbook.pdf
   ```

5. Launch the app:
   ```bash
   streamlit run app.py
   ```

## Tech stack

- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`) — local, free
- **Vector store:** Chroma (persisted locally)
- **LLM:** Llama 3.3 70B via Groq API — free tier, fast inference
- **UI:** Streamlit

## Known limitations (by design)

These are exactly what the Advanced RAG project fixes:
- Pure semantic search misses exact keyword/syntax matches (e.g. `len()`, `__init__`)
- No re-ranking — retrieved chunks aren't scored for actual relevance
- No query rewriting — vague questions retrieve poorly
- No self-check — if retrieval fails, the model still tries to answer
- Single retrieval pass — can't handle multi-hop questions well

## Deployment

Deploy free on [Streamlit Community Cloud](https://streamlit.io/cloud)
or [Hugging Face Spaces](https://huggingface.co/spaces). Remember to
add `GROQ_API_KEY` as a secret in the deployment settings, and either
commit the `chroma_db/` folder or run `ingest.py` as a startup step.

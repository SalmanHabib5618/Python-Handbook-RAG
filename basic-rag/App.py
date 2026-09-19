"""
Basic RAG (Retrieval-Augmented Generation) App
------------------------------------------------
Users upload documents from different sources (PDF, DOCX, TXT, CSV, or a web
URL). The app builds a FAISS vector index over the content, and the user can
then ask natural-language questions answered using only that content, with
sources shown for each answer.

LLM: Groq (openai/gpt-oss-20b by default) via GROQ_API_KEY
Embeddings: local HuggingFace sentence-transformers (free, no extra API key)

Stack: Streamlit + LangChain (0.3.x) + FAISS + Groq
"""

import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA


# ----------------------------- Page setup -----------------------------
st.set_page_config(page_title="Basic RAG App", page_icon="📄", layout="wide")
st.title("📄 Basic RAG — Chat With Your Documents")
st.caption(
    "Upload documents from different sources, then ask questions. "
    "Answers are generated only from the content you provide, powered by Groq."
)

# ----------------------------- Session state -----------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (question, answer, sources)

# ----------------------------- API key resolution -----------------------------
# Prefer a secret set via Streamlit secrets / environment variable (recommended
# for deployment) and only fall back to the sidebar input for local testing.
# GROQ_API_KEY should never be hardcoded or committed to source control.
try:
    default_groq_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
except Exception:
    default_groq_key = os.environ.get("GROQ_API_KEY", "")

GROQ_MODELS = [
    "openai/gpt-oss-20b",   # fastest / cheapest, good default
    "openai/gpt-oss-120b",  # stronger reasoning, slower
    "groq/compound",        # agentic system model
]

# ----------------------------- Sidebar: setup -----------------------------
with st.sidebar:
    st.header("⚙️ Setup")

    groq_api_key = st.text_input(
        "Groq API Key",
        value=default_groq_key,
        type="password",
        help="Stored only for this session. Prefer setting GROQ_API_KEY as a "
             "secret/environment variable instead of pasting it here.",
    )
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key

    model_name = st.selectbox("Groq model", GROQ_MODELS, index=0)

    st.divider()
    st.header("📥 Add Sources")

    uploaded_files = st.file_uploader(
        "Upload documents (PDF, DOCX, TXT, CSV)",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True,
    )

    web_url = st.text_input("Or add a web page URL (optional)")

    chunk_size = st.slider("Chunk size", 300, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, step=50)

    build_clicked = st.button("🔨 Build Knowledge Base", use_container_width=True)


# ----------------------------- Helper: load a single file -----------------------------
def load_document(uploaded_file):
    """Save an uploaded file to a temp path and load it with the right loader."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix == ".docx":
            loader = Docx2txtLoader(tmp_path)
        elif suffix == ".csv":
            loader = CSVLoader(tmp_path)
        else:  # .txt and anything else we treat as plain text
            loader = TextLoader(tmp_path, encoding="utf-8")

        docs = loader.load()
        for d in docs:
            d.metadata["source"] = uploaded_file.name
        return docs
    finally:
        os.unlink(tmp_path)


@st.cache_resource(show_spinner=False)
def get_embeddings():
    # Local, free embedding model — no API key required for this step.
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ----------------------------- Build knowledge base -----------------------------
if build_clicked:
    if not groq_api_key:
        st.sidebar.error("Please enter your Groq API key first.")
    elif not uploaded_files and not web_url:
        st.sidebar.error("Upload at least one file or provide a URL.")
    else:
        with st.spinner("Reading sources and building the vector index..."):
            all_docs = []

            for f in uploaded_files or []:
                try:
                    all_docs.extend(load_document(f))
                except Exception as e:
                    st.sidebar.error(f"Failed to load {f.name}: {e}")

            if web_url:
                try:
                    web_docs = WebBaseLoader(web_url).load()
                    for d in web_docs:
                        d.metadata["source"] = web_url
                    all_docs.extend(web_docs)
                except Exception as e:
                    st.sidebar.error(f"Failed to load URL: {e}")

            if all_docs:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                chunks = splitter.split_documents(all_docs)

                embeddings = get_embeddings()
                vectorstore = FAISS.from_documents(chunks, embeddings)
                st.session_state.vectorstore = vectorstore

                llm = ChatGroq(model=model_name, temperature=0)
                st.session_state.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
                    return_source_documents=True,
                )

                st.sidebar.success(
                    f"Knowledge base built from {len(all_docs)} document(s), "
                    f"{len(chunks)} chunks."
                )
            else:
                st.sidebar.error("No content could be loaded from the given sources.")


# ----------------------------- Main: chat interface -----------------------------
st.subheader("💬 Ask a Question")

if st.session_state.qa_chain is None:
    st.info("Upload documents and click **Build Knowledge Base** in the sidebar to get started.")
else:
    question = st.text_input("Your question", placeholder="e.g. What is the refund policy?")
    ask_clicked = st.button("Ask")

    if ask_clicked and question:
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.qa_chain.invoke({"query": question})
                answer = result["result"]
                sources = sorted({
                    doc.metadata.get("source", "unknown")
                    for doc in result.get("source_documents", [])
                })
                st.session_state.chat_history.insert(0, (question, answer, sources))
            except Exception as e:
                st.error(f"Something went wrong while answering: {e}")

    for q, a, srcs in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)
            if srcs:
                st.caption("Sources: " + ", ".join(srcs))

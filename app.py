"""
app.py — Basic RAG: Step 3
Streamlit chat UI on top of query.py. Run with:
    streamlit run app.py
"""

import streamlit as st

from query import answer

st.set_page_config(page_title="Python Handbook — Basic RAG", page_icon="🐍")
st.title("🐍 Ask the Python Handbook")
st.caption(
    "Basic RAG demo — single retrieval pass over 'The Python Handbook' "
    "by Flavio Copes. Ask any question about Python fundamentals."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("e.g. What's the difference between a list and a tuple?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant sections and generating an answer..."):
            result = answer(question)
            pages = sorted({s["page_number"] for s in result["sources"]})
            sources_line = f"\n\n*Sources: page(s) {', '.join(map(str, pages))}*"
            full_response = result["answer"] + sources_line
            st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

with st.sidebar:
    st.header("About this project")
    st.markdown(
        """
This is the **Basic RAG** baseline in a two-project comparison:

1. **Basic RAG** *(this app)* — single-pass retrieve-then-generate
2. **Advanced/Agentic RAG** — hybrid search, re-ranking, query
   rewriting, and self-checking retrieval

Built with local embeddings (`sentence-transformers`), Chroma
for vector storage, and Groq for fast, free LLM inference.
"""
    )
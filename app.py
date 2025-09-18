import os
import io
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st
from tqdm import tqdm

from utils import (
    init_env,
    ensure_dir,
    save_json,
    load_json,
    FAISS_INDEX_PATH,
    METADATA_JSON_PATH,
)
from parser import parse_pdf
from chunker import chunk_text
from embedder import Embedder
from indexer import build_index, save_index, load_index
from retriever import Retriever
from llm_client import GeminiClient


st.set_page_config(page_title="DocMind v1", layout="wide")
init_env()

# Initialize chat history
if "chat" not in st.session_state:
    st.session_state["chat"] = []  # list of {question, answer, contexts}


def ingest_document(uploaded_file) -> None:
    # Save uploaded file to a temporary path
    temp_dir = "tmp"
    ensure_dir(temp_dir)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("Parsing PDF...")
    pages = parse_pdf(temp_path)

    st.info("Chunking text (~300 tokens per chunk)...")
    chunks = chunk_text(pages, chunk_size_tokens=600, doc_id=uploaded_file.name)

    st.info("Embedding chunks in batches...")
    embedder = Embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_batch(texts, batch_size=32, normalize=True)

    # Build metadata aligned with embeddings
    metadata: List[Dict] = []
    for c in chunks:
        metadata.append({
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "page": c["page"],
            "text": c["text"],
        })

    st.info("Building FAISS index and saving to disk...")
    index = build_index(embeddings, metadata)
    save_index(index, metadata, FAISS_INDEX_PATH, METADATA_JSON_PATH)

    st.success(f"Ingested {len(chunks)} chunks across {len(pages)} pages.")


def clear_index() -> None:
    if os.path.exists(FAISS_INDEX_PATH):
        os.remove(FAISS_INDEX_PATH)
    if os.path.exists(METADATA_JSON_PATH):
        os.remove(METADATA_JSON_PATH)


def index_status() -> str:
    has_index = os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_JSON_PATH)
    if not has_index:
        return "No index found."
    metadata = load_json(METADATA_JSON_PATH) or []
    return f"Index ready with {len(metadata)} chunks."


st.title("DocMind v1 — PDF Q&A (RAG)")

with st.sidebar:
    st.header("Ingestion")
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Ingest Document", use_container_width=True, disabled=uploaded_pdf is None):
            with st.spinner("Ingesting document..."):
                ingest_document(uploaded_pdf)
    with col_b:
        if st.button("Clear Index", use_container_width=True):
            clear_index()
            st.info("Index cleared.")

    st.write("Status:", index_status())

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state["chat"] = []
        st.success("Chat cleared.")

st.header("Ask a Question")
question = st.text_input("Your question")

# Allow user to set Top-k before asking
top_k = st.slider("Top-k chunks", min_value=1, max_value=10, value=8)

if st.button("Ask", disabled=not (os.path.exists(FAISS_INDEX_PATH) and question.strip())):
    with st.spinner("Retrieving..."):
        embedder = Embedder()
        retriever = Retriever(embedder)
        chunks = retriever.retrieve(question, top_k=top_k)

    st.subheader("Top Contexts")
    for i, rc in enumerate(chunks, start=1):
        with st.expander(f"[{i}] Page {rc.page} | Score {rc.score:.3f}"):
            st.write(rc.text)

    with st.spinner("Querying LLM (Gemini 2.0 Flash)..."):
        client = GeminiClient()
        contexts = [
            {"page": c.page, "text": c.text}
            for c in chunks
        ]
        answer = client.generate_answer(question, contexts)

    # Append to chat history
    st.session_state["chat"].append({
        "question": question,
        "answer": answer,
        "contexts": contexts,
    })

st.subheader("Chat History")
for turn in st.session_state["chat"]:
    st.markdown(f"**You:** {turn['question']}")
    st.markdown(f"**DocMind:** {turn['answer']}")
    with st.expander("Sources"):
        for i, ctx in enumerate(turn["contexts"], start=1):
            st.write(f"[{i}] Page {ctx['page']}")
            st.write(ctx["text"])

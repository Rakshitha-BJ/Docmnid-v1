import io
from typing import List

import fitz  # PyMuPDF

from parser import parse_pdf
from chunker import chunk_text
from embedder import Embedder
from indexer import build_index, save_index, load_index
from retriever import Retriever


def create_sample_pdf_text(lines: List[str]) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    text = "\n".join(lines)
    page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def write_pdf_to_file(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)


def main():
    sample_lines = [
        "This document describes the capital of France.",
        "The capital city is Paris and it is known for the Eiffel Tower.",
        "End of sample document.",
    ]
    pdf_bytes = create_sample_pdf_text(sample_lines)
    tmp_path = "tests_sample.pdf"
    write_pdf_to_file(tmp_path, pdf_bytes)

    pages = parse_pdf(tmp_path)
    chunks = chunk_text(pages, chunk_size_tokens=120, doc_id="testdoc")

    embedder = Embedder()
    texts = [c["text"] for c in chunks]
    embs = embedder.embed_batch(texts, batch_size=32, normalize=True)

    metadata = [
        {"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "page": c["page"], "text": c["text"]}
        for c in chunks
    ]
    index = build_index(embs, metadata)
    save_index(index, metadata)

    retriever = Retriever(embedder)
    results = retriever.retrieve("What is the capital of France?", top_k=3)
    assert any("Paris" in r.text for r in results)
    print("Test OK - retrieved chunks contain 'Paris'")


if __name__ == "__main__":
    main()

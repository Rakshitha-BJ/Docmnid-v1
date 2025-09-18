# DocMind v1 — A Lightweight RAG for PDF Question Answering

DocMind v1 is a minimal, production-quality Retrieval-Augmented Generation (RAG) app for Question Answering over a single PDF. It uses sentence-transformers for embedding (default: all-MiniLM-L6-v2), FAISS for vector search (persisted locally), and Gemini 2.0 Flash as the LLM via HTTP API. It provides an end-to-end Streamlit UI to ingest a PDF, ask questions, and view grounded answers with page citations and snippet previews.

## Pre-requisites
- Python 3.10+
- pip

## Setup and Run

1. Clone the repo
```bash
git clone <repo>
cd docmind-v1
```

2. Create a virtual environment and activate it
- macOS/Linux:
```bash
python -m venv .venv && source .venv/bin/activate
```
- Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Configure environment variables
- Export your Gemini API key (or place in a `.env` file at repo root):
```bash
export GEMINI_API_KEY="your_key_here"
```
- Windows PowerShell:
```powershell
$Env:GEMINI_API_KEY = "your_key_here"
```

5. Start the Streamlit app
```bash
python -m nltk.downloader punkt punkt_tab
streamlit run app.py
```

Alternatively on macOS/Linux:
```bash
bash run_demo.sh
```

## Usage
1. In the Streamlit UI:
   - Upload a PDF (or place a file at `data/example.pdf`).
   - Click "Ingest Document". This will parse the PDF, chunk text (~300 tokens per chunk), embed chunks (batched), build a FAISS flat index, and persist both index and metadata.
   - Type a question and click "Ask". The app will search relevant chunks and call Gemini 2.0 Flash with a grounded prompt to produce a concise answer with **[Doc: page X]** citations.
   - Click a source snippet to preview the full page text below.

2. Re-indexing:
   - Use the "Clear Index" button to remove `faiss_index.bin` and `metadata.json`, then re-ingest.

## File Structure
```text
docmind-v1/
├─ app.py                     # Streamlit app (entrypoint)
├─ parser.py                  # PDF parsing & text extraction (PyMuPDF)
├─ chunker.py                 # chunk_text(...) implementation (300-token approx)
├─ embedder.py                # embeddings via sentence-transformers
├─ indexer.py                 # FAISS index build/save/load + metadata store
├─ retriever.py               # query embedding + faiss search (top_k)
├─ llm_client.py              # Gemini adapter (generate_answer(...))
├─ prompt_templates.py        # prompt templates (citation-first)
├─ utils.py                   # helper functions (IO, batching, env load)
├─ requirements.txt
├─ README.md
├─ .gitignore
├─ run_demo.sh                # one-line script to run the app locally
└─ data/
   └─ example.pdf             # placeholder path (not provided here)
```

## Environment Variables
- GEMINI_API_KEY: Required to call Gemini 2.0 Flash API.

You may also create a `.env` file at the repo root:
```
GEMINI_API_KEY=your_key_here
```

## How It Works
- Parsing: `parser.py` uses PyMuPDF to extract text per page while optionally removing repeated headers/footers.
- Chunking: `chunker.py` uses NLTK sentence tokenization and approximates tokens as words to build ~300-token chunks, minimum 50 words, no overlap (v1).
- Embedding: `embedder.py` loads `sentence-transformers/all-MiniLM-L6-v2` and embeds in batches.
- Indexing: `indexer.py` builds a FAISS Flat IP index (cosine similarity via normalization), persists `faiss_index.bin` and `metadata.json`.
- Retrieval: `retriever.py` embeds the query, searches FAISS, returns top-k chunks with scores.
- LLM: `llm_client.py` calls Gemini 2.0 Flash via HTTP (with retries, timeout). The prompt enforces source grounding: "Use only provided excerpts; otherwise respond 'I don't know'." Answers must include citations like `[Doc: page 23]`.
- UI: `app.py` provides upload, ingest, ask, and snippet preview.

## Example Test Case
- Upload any short PDF containing a sentence like: "The Apollo program landed on the Moon in 1969."
- Ask: `When did the Apollo program land on the Moon?`
- Expected answer style:
```
1969. [Doc: page 1]
```
And the UI will list a snippet from the relevant page with a clickable preview.

## Where Artifacts Are Saved
- FAISS index: `faiss_index.bin`
- Metadata: `metadata.json`
These are saved in the project root alongside the app (override paths in `utils.py` if desired).

## Troubleshooting
- Missing NLTK data: The app will download `punkt` on first run. If you see tokenization errors, ensure your environment has internet access or manually run:
```python
import nltk
nltk.download('punkt')
```
- FAISS import error: Ensure `faiss-cpu` is installed and matches your Python version (`pip install faiss-cpu`).
- PDF parsing issues: Some scanned PDFs may require OCR. This app expects extractable text.
- Gemini errors: Confirm `GEMINI_API_KEY` is set. Network errors are retried with exponential backoff.

## Running Tests
A minimal pipeline test is provided that programmatically creates a small PDF, ingests it end-to-end, and verifies retrieval:
```bash
python tests/test_pipeline.py
```

## Sample Streamlit Log
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
Network URL: http://192.168.1.10:8501
```
Sample question and response printed in the UI:
```
Q: What is the capital of France?
A: Paris. [Doc: page 1]
```

## CHANGELOG / Extension Points
- v1: Initial implementation with sentence-transformers, FAISS, Streamlit UI, Gemini adapter.
- Swap LLM: Replace logic in `llm_client.py` (single class `GeminiClient`). Keep the same `generate_answer(question, contexts)` method signature.
- Re-ranking: Add a second-stage ranker in `retriever.py` later if needed.
- Multi-document: Extend metadata schema in `indexer.py` to include `doc_id` per file and manage multiple indices or a combined one.

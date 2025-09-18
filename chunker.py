from typing import Dict, List
import uuid
import nltk

# Ensure required NLTK tokenizers are available; download if missing.
for resource in ['punkt', 'punkt_tab']:
    try:
        # punkt is at tokenizers/punkt, punkt_tab at tokenizers/punkt_tab
        path = f"tokenizers/{resource}"
        nltk.data.find(path)
    except LookupError:
        try:
            nltk.download(resource)
        except Exception:
            # As a last resort, continue; downstream will attempt a naive split
            pass


def _word_count(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


def _split_sentences(text: str) -> List[str]:
    """Try NLTK sentence tokenizer; fallback to naive split on periods."""
    try:
        return nltk.sent_tokenize(text)
    except Exception:
        return [s.strip() for s in text.split('.') if s.strip()]


def chunk_text(pages: List[Dict], chunk_size_tokens: int = 300, doc_id: str = "doc") -> List[Dict]:
    """Chunk pages into approx 300-token (word) chunks using sentence boundaries.

    Args:
        pages: List of {"page": int, "text": str}
        chunk_size_tokens: Approx target tokens per chunk (word approximation).
        doc_id: Identifier for the document.

    Returns:
        List of Chunk dicts: {"chunk_id", "doc_id", "page", "text"}
    """
    chunks: List[Dict] = []

    for page in pages:
        page_num = page["page"]
        text = page.get("text", "")
        if not text:
            continue
        sentences = _split_sentences(text)
        current: List[str] = []
        current_wc = 0

        for sent in sentences:
            wc = _word_count(sent)
            if current_wc + wc >= chunk_size_tokens and current_wc >= 50:
                # finalize current chunk
                chunk_text_str = " ".join(current).strip()
                if chunk_text_str:
                    chunks.append({
                        "chunk_id": str(uuid.uuid4()),
                        "doc_id": doc_id,
                        "page": page_num,
                        "text": chunk_text_str,
                    })
                current = [sent]
                current_wc = wc
            else:
                current.append(sent)
                current_wc += wc

        # flush remainder if sufficiently long or page end
        if current:
            chunk_text_str = " ".join(current).strip()
            if _word_count(chunk_text_str) >= 50:
                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "doc_id": doc_id,
                    "page": page_num,
                    "text": chunk_text_str,
                })
    return chunks

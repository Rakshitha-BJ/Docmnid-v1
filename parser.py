from typing import Dict, List
import re
import fitz  # PyMuPDF


def _clean_repeated_lines(lines: List[str]) -> List[str]:
    """Heuristic removal of repeated header/footer lines across a page.

    Removes lines that are too short or common boilerplate like page numbers.
    """
    cleaned: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip pure page numbers or very short lines likely to be headers/footers
        if re.fullmatch(r"\d+", stripped):
            continue
        if len(stripped) < 3:
            continue
        cleaned.append(stripped)
    return cleaned


def parse_pdf(path: str) -> List[Dict]:
    """Parse a PDF into a list of pages with text.

    Args:
        path: Path to PDF file.

    Returns:
        List of dicts: {"page": int, "text": str}
    """
    pages: List[Dict] = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            # Basic line-level cleanup
            lines = text.splitlines()
            lines = _clean_repeated_lines(lines)
            normalized = "\n".join(lines)
            pages.append({"page": i, "text": normalized})
    return pages

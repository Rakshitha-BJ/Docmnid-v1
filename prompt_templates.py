from typing import List, Dict

EXPLANATION = (
    "You are DocMind. Use ONLY the following numbered excerpts from the user's document to answer the question. "
    "Do NOT invent facts. If evidence is insufficient, reply \"I don't know based on the provided document.\""
)

PROMPT_TEMPLATE = (
    "System: " + EXPLANATION + "\n\n"
    "EXCERPTS:\n{excerpts}\n\n"
    "Question: {user_question}\n\n"
    "Answer as 1–3 concise bullet points. Directly address the question using information from EXCERPTS. "
    "Quote directly from EXCERPTS when possible and keep wording close to source. "
    "Ensure the answer is relevant to the question and supported by the excerpts. "
    "For EVERY factual claim, append a page citation in the exact form [Doc: page X]. Do NOT use numeric references like [1] or [2]. "
    "Use only pages present in EXCERPTS. If no relevant information, say 'I don't know based on the provided document.'"
)


def build_excerpts(contexts: List[Dict]) -> str:
    """Format contexts into a numbered excerpt block for the prompt."""
    lines = []
    for idx, ctx in enumerate(contexts, start=1):
        page = ctx.get("page", "?")
        text = ctx.get("text", "")
        snippet = text.strip().replace("\n", " ")
        if len(snippet) > 500:
            snippet = snippet[:500] + "..."
        lines.append(f"[{idx}] (Page {page}): \"{snippet}\"")
    return "\n".join(lines)


def build_prompt(question: str, contexts: List[Dict]) -> str:
    """Build the final prompt string given question and contexts."""
    excerpts = build_excerpts(contexts)
    return PROMPT_TEMPLATE.format(excerpts=excerpts, user_question=question)

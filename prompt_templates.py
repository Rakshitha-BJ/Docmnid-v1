from typing import List, Dict

EXPLANATION = (
    "You are DocMind. Use ONLY the following numbered excerpts from the user's document to answer the question. "
    "Do NOT invent facts. If evidence is insufficient, reply \"I don't know based on the provided document.\""
)

PROMPT_TEMPLATE = (
    "System: " + EXPLANATION + "\n\n"
    "EXCERPTS:\n{excerpts}\n\n"
    "Question: {user_question}\n\n"
    "Answer concisely and include bracketed citations for factual claims like [Doc: page 23]."
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

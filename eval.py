import json
import re
from typing import List, Dict, Tuple

from embedder import Embedder
from retriever import Retriever
from llm_client import GeminiClient


def lcs_length(a: List[str], b: List[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l(pred: str, ref: str) -> float:
    a = pred.split()
    b = ref.split()
    if not a or not b:
        return 0.0
    lcs = lcs_length(a, b)
    prec = lcs / len(a)
    rec = lcs / len(b)
    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)


def extract_cited_pages(answer: str) -> List[int]:
    pages = re.findall(r"\[Doc: page\s*(\d+)\]", answer, flags=re.IGNORECASE)
    return [int(p) for p in pages]


def load_eval_data(path: str) -> List[Dict]:
    items: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def main():
    data = load_eval_data("tests/eval_data.jsonl")

    embedder = Embedder()
    retriever = Retriever(embedder)
    llm = GeminiClient()

    k = 5
    total = len(data)
    recall_hits = 0
    rouge_scores: List[float] = []
    citation_hits = 0

    for ex in data:
        q = ex["question"]
        gold_answer = ex["gold_answer"]
        gold_pages = set(ex["gold_pages"]) if ex.get("gold_pages") else set()

        results = retriever.retrieve(q, top_k=k)
        retrieved_pages = {r.page for r in results}
        has_hit = bool(gold_pages & retrieved_pages) if gold_pages else False
        recall_hits += 1 if has_hit else 0

        contexts = [{"page": r.page, "text": r.text} for r in results]
        answer = llm.generate_answer(q, contexts)

        rouge_scores.append(rouge_l(answer, gold_answer))

        cited = set(extract_cited_pages(answer))
        citation_ok = cited.issubset(gold_pages) if gold_pages else False
        citation_hits += 1 if citation_ok else 0

        print(f"Q: {q}")
        print(f"Retrieved pages: {sorted(list(retrieved_pages))} | Gold pages: {sorted(list(gold_pages))} | Hit: {has_hit}")
        print(f"Answer: {answer}")
        print(f"ROUGE-L: {rouge_scores[-1]:.3f} | Citation OK: {citation_ok}")
        print("-" * 60)

    recall_at_k = recall_hits / total if total else 0.0
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    citation_acc = citation_hits / total if total else 0.0

    print("Summary:")
    print(f"Recall@{k}: {recall_at_k:.3f}")
    print(f"Avg ROUGE-L: {avg_rouge:.3f}")
    print(f"Citation correctness: {citation_acc:.3f}")


if __name__ == "__main__":
    main()

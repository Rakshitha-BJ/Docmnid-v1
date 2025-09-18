import json
import re
import time
import psutil
import os
from typing import List, Dict, Tuple

from embedder import Embedder
from retriever import Retriever
from llm_client import GeminiClient
import numpy as np
from utils import load_json, METADATA_JSON_PATH


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


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    if u.ndim > 1:
        u = u[0]
    if v.ndim > 1:
        v = v[0]
    denom = (np.linalg.norm(u) * np.linalg.norm(v))
    if denom == 0:
        return 0.0
    return float(np.dot(u, v) / denom)


def ndcg_at_k(retrieved_pages: List[int], gold_pages: List[int], k: int) -> float:
    gold_set = set(gold_pages)
    gains = []
    for rank, page in enumerate(retrieved_pages[:k], start=1):
        rel = 1.0 if page in gold_set else 0.0
        gains.append(rel / np.log2(rank + 1))
    dcg = sum(gains)
    # Ideal DCG with all relevant first
    ideal_rels = [1.0] * min(len(gold_set), k)
    idcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels)])
    return float(dcg / idcg) if idcg > 0 else 0.0


def faithfulness(answer: str, contexts: List[Dict]) -> float:
    cited_pages = extract_cited_pages(answer)
    if not cited_pages:
        return 0.0
    page_to_text = {}
    for c in contexts:
        page_to_text.setdefault(c["page"], c["text"])  # keep first occurrence
    supported = 0
    tokens = [t.lower() for t in answer.split() if len(t) > 3]
    for p in cited_pages:
        txt = page_to_text.get(p, "").lower()
        if txt and any(tok in txt for tok in tokens[:10]):
            supported += 1
    return supported / len(cited_pages)


def relevance(answer: str, question: str, embedder: Embedder) -> float:
    q = embedder.embed_batch([question], batch_size=1, normalize=True)
    a = embedder.embed_batch([answer], batch_size=1, normalize=True)
    return cosine_similarity(q, a)


def completeness(answer: str, gold_answer: str, embedder: Embedder) -> float:
    a = embedder.embed_batch([answer], batch_size=1, normalize=True)
    g = embedder.embed_batch([gold_answer], batch_size=1, normalize=True)
    return cosine_similarity(a, g)


def derive_gold_pages(question: str, gold_answer: str, metadata: List[Dict]) -> List[int]:
    # Build per-page concatenated text
    page_to_text: Dict[int, List[str]] = {}
    for m in metadata:
        page = int(m.get("page", 0))
        page_to_text.setdefault(page, []).append(m.get("text", ""))
    page_texts: Dict[int, str] = {p: " \n".join(txts).lower() for p, txts in page_to_text.items()}

    # Extract keywords from gold answer and question
    def keywords(s: str) -> List[str]:
        return [w.lower() for w in re.findall(r"[A-Za-z0-9]+", s) if len(w) > 3]

    keys = keywords(gold_answer) + keywords(question)
    keys = list(dict.fromkeys(keys))  # dedupe, preserve order

    scores: Dict[int, int] = {}
    for p, txt in page_texts.items():
        hit = sum(1 for k in keys if k in txt)
        scores[p] = hit
    # Keep pages with at least 2 keyword hits (tunable)
    selected = [p for p, s in scores.items() if s >= 2]
    # If none, relax to >=1
    if not selected:
        selected = [p for p, s in scores.items() if s >= 1]
    return sorted(selected)


def main():
    data = load_eval_data("tests/eval_data.jsonl")
    metadata = load_json(METADATA_JSON_PATH) or []

    embedder = Embedder()
    retriever = Retriever(embedder)
    llm = GeminiClient()

    k = 10
    total = len(data)
    recall_hits = 0
    rouge_scores: List[float] = []
    citation_hits = 0
    sem_sims: List[float] = []
    ndcgs: List[float] = []
    faithful_scores: List[float] = []
    relevance_scores: List[float] = []
    completeness_scores: List[float] = []

    t_retrieve = []
    t_llm = []

    process = psutil.Process(os.getpid())

    for ex in data:
        q = ex["question"]
        gold_answer = ex["gold_answer"]
        static_gold_pages = set(ex["gold_pages"]) if ex.get("gold_pages") else set()

        # Derive gold pages dynamically from current index if possible
        dynamic_pages = set(derive_gold_pages(q, gold_answer, metadata)) if metadata else set()
        gold_pages = dynamic_pages if dynamic_pages else static_gold_pages

        t0 = time.time()
        results = retriever.retrieve(q, top_k=k)
        t_retrieve.append(time.time() - t0)

        retrieved_pages_list = [r.page for r in results]
        retrieved_pages = set(retrieved_pages_list)
        has_hit = bool(gold_pages & retrieved_pages) if gold_pages else False
        recall_hits += 1 if has_hit else 0
        ndcgs.append(ndcg_at_k(retrieved_pages_list, list(gold_pages), k))

        contexts = [{"page": r.page, "text": r.text} for r in results]

        t1 = time.time()
        answer = llm.generate_answer(q, contexts)
        t_llm.append(time.time() - t1)

        rouge_scores.append(rouge_l(answer, gold_answer))

        cited = set(extract_cited_pages(answer))
        citation_ok = bool(cited & gold_pages) if gold_pages else False
        citation_hits += 1 if citation_ok else 0

        faithful_scores.append(faithfulness(answer, contexts))
        relevance_scores.append(relevance(answer, q, embedder))
        completeness_scores.append(completeness(answer, gold_answer, embedder))

        mem_mb = process.memory_info().rss / (1024 * 1024)

        print(f"Q: {q}")
        print(f"Retrieved pages: {sorted(list(retrieved_pages))} | Gold pages (auto if found): {sorted(list(gold_pages))} | Hit: {has_hit}")
        print(f"nDCG@{k}: {ndcgs[-1]:.3f}")
        print(f"Answer: {answer}\n")
        print(f"ROUGE-L: {rouge_scores[-1]:.3f} | SemanticSim: {completeness_scores[-1]:.3f} | Citation OK: {citation_ok}")
        print(f"Faithfulness: {faithful_scores[-1]:.3f} | Relevance: {relevance_scores[-1]:.3f}")
        print(f"Latency - retrieve: {t_retrieve[-1]:.2f}s, llm: {t_llm[-1]:.2f}s | RSS: {mem_mb:.0f} MB")
        print("-" * 60)

    recall_at_k = recall_hits / total if total else 0.0
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    citation_acc = citation_hits / total if total else 0.0
    avg_sem = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0
    avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
    avg_faith = sum(faithful_scores) / len(faithful_scores) if faithful_scores else 0.0
    avg_rel = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
    avg_t_ret = sum(t_retrieve) / len(t_retrieve) if t_retrieve else 0.0
    avg_t_llm = sum(t_llm) / len(t_llm) if t_llm else 0.0

    print("Summary:")
    print(f"Recall@{k}: {recall_at_k:.3f}")
    print(f"nDCG@{k}: {avg_ndcg:.3f}")
    print(f"Avg ROUGE-L: {avg_rouge:.3f}")
    print(f"Avg SemanticSim (answer vs gold): {avg_sem:.3f}")
    print(f"Avg Faithfulness: {avg_faith:.3f}")
    print(f"Avg Relevance (answer vs question): {avg_rel:.3f}")
    print(f"Citation correctness: {citation_acc:.3f}")
    print(f"Avg Latency - retrieve: {avg_t_ret:.2f}s, llm: {avg_t_llm:.2f}s")


if __name__ == "__main__":
    main()

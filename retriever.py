from typing import Dict, List
import numpy as np
from embedder import Embedder
from indexer import load_index, search
from utils import RetrievedChunk


class Retriever:
    """Retriever that embeds query and searches FAISS index."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.index, self.metadata = load_index()

    def retrieve(self, question: str, top_k: int = 15) -> List[RetrievedChunk]:
        """Retrieve top-k chunks for the given question."""
        q_emb = self.embedder.embed_batch([question], batch_size=1, normalize=True)
        scores, indices = search(self.index, q_emb[0], top_k=top_k)
        result: List[RetrievedChunk] = []
        top_scores = scores[0]
        top_indices = indices[0]
        for score, idx in zip(top_scores, top_indices):
            if idx < 0 or idx >= len(self.metadata):
                continue
            m = self.metadata[idx]
            result.append(RetrievedChunk(
                chunk_id=m.get("chunk_id"),
                doc_id=m.get("doc_id"),
                page=m.get("page"),
                text=m.get("text"),
                score=float(score),
            ))
        return result

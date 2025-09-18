from typing import List, Optional
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("docmind.embedder")


class Embedder:
    """Sentence-transformers embedder with batching and caching."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L12-v2", device: Optional[str] = None):
        self.model_name = model_name
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name, device=device)
        self.cache = {}  # Simple cache for embeddings

    def embed_batch(self, texts: List[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        """Embed a list of texts in batches with caching.

        Args:
            texts: Input strings.
            batch_size: Batch size.
            normalize: Whether to L2-normalize outputs (recommended for cosine similarity).

        Returns:
            Numpy array of shape (N, D)
        """
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)

        # Check cache
        cached_embeddings = []
        to_embed = []
        indices = []
        for i, text in enumerate(texts):
            if text in self.cache:
                cached_embeddings.append((i, self.cache[text]))
            else:
                to_embed.append(text)
                indices.append(i)

        # Embed uncached texts
        if to_embed:
            new_embeddings = self.model.encode(
                to_embed,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
            ).astype(np.float32)
            # Cache them
            for text, emb in zip(to_embed, new_embeddings):
                self.cache[text] = emb

        # Combine results
        result = np.zeros((len(texts), self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        for i, emb in cached_embeddings:
            result[i] = emb
        if to_embed:
            for idx, emb in zip(indices, new_embeddings):
                result[idx] = emb
        return result

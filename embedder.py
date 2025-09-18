from typing import List, Optional
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("docmind.embedder")


class Embedder:
    """Sentence-transformers embedder with batching."""

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", device: Optional[str] = None):
        self.model_name = model_name
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name, device=device)

    def embed_batch(self, texts: List[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        """Embed a list of texts in batches.

        Args:
            texts: Input strings.
            batch_size: Batch size.
            normalize: Whether to L2-normalize outputs (recommended for cosine similarity).

        Returns:
            Numpy array of shape (N, D)
        """
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embeddings.astype(np.float32)

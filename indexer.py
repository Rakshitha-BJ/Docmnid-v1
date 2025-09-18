from typing import Dict, List, Tuple
import os
import faiss
import numpy as np
from utils import save_json, load_json, FAISS_INDEX_PATH, METADATA_JSON_PATH


def build_index(embeddings: np.ndarray, metadata: List[Dict]) -> faiss.Index:
    """Build a FAISS FlatIP index (cosine via normalized vectors).

    Args:
        embeddings: Float32 array (N, D), assumed L2-normalized for cosine.
        metadata: List of metadata dicts aligned with embeddings (len N).

    Returns:
        FAISS index with all vectors added.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D array")
    num, dim = embeddings.shape
    if num != len(metadata):
        raise ValueError("embeddings and metadata length mismatch")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, metadata: List[Dict], index_path: str = FAISS_INDEX_PATH, metadata_path: str = METADATA_JSON_PATH) -> None:
    """Persist FAISS index and metadata JSON."""
    faiss.write_index(index, index_path)
    save_json(metadata_path, metadata)


def load_index(index_path: str = FAISS_INDEX_PATH, metadata_path: str = METADATA_JSON_PATH) -> Tuple[faiss.Index, List[Dict]]:
    """Load FAISS index and metadata mapping.

    Returns:
        (index, metadata)
    """
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError("Index or metadata not found. Ingest the document first.")
    index = faiss.read_index(index_path)
    metadata = load_json(metadata_path) or []
    return index, metadata


def search(index: faiss.Index, query_vector: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Search top_k results for a single query vector.

    Args:
        index: FAISS index.
        query_vector: (D,) or (1, D) numpy array, normalized.
        top_k: number of results.

    Returns:
        (scores, indices) arrays of shape (1, top_k)
    """
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    scores, indices = index.search(query_vector.astype(np.float32), top_k)
    return scores, indices

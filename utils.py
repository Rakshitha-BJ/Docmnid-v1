import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv


# Initialize logging early
LOGGER_NAME = "docmind"
logger = logging.getLogger(LOGGER_NAME)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def init_env() -> None:
    """Load environment variables from .env if present."""
    load_dotenv(override=False)


def get_env(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """Get environment variable by name.

    Args:
        name: Variable name.
        default: Default value if not set.
        required: If True, raise error when missing.

    Returns:
        The value or default.
    """
    value = os.environ.get(name, default)
    if required and (value is None or value == ""):
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_json(path: str, data: Any) -> None:
    """Save JSON to file with UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    """Load JSON from file if exists, else return None."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def batch_iterable(items: Iterable[Any], batch_size: int) -> Iterable[List[Any]]:
    """Yield lists of size up to batch_size from iterable."""
    batch: List[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# Default artifact paths in project root
FAISS_INDEX_PATH = "faiss_index.bin"
METADATA_JSON_PATH = "metadata.json"


@dataclass
class RetrievedChunk:
    """Container for retrieved chunk and score."""
    chunk_id: str
    doc_id: str
    page: int
    text: str
    score: float


def exponential_backoff_retries(
    func,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """Execute function with exponential backoff retries.

    Args:
        func: Callable with no args that executes the work.
        max_retries: Maximum attempts.
        initial_delay: Initial delay in seconds.
        backoff_factor: Multiplier for each retry.
        exceptions: Exception types to catch and retry.

    Returns:
        The function result.
    """
    attempt = 0
    delay = initial_delay
    while True:
        try:
            return func()
        except exceptions as e:
            attempt += 1
            if attempt > max_retries:
                logger.error("Max retries exceeded: %s", e)
                raise
            logger.warning("Retry %d/%d after error: %s", attempt, max_retries, e)
            time.sleep(delay)
            delay *= backoff_factor

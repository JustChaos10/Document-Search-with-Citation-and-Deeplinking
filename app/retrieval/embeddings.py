from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - optional dependency
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
    HAS_LC_HF = True
except ImportError:  # pragma: no cover - handled at runtime
    HuggingFaceEmbeddings = None  # type: ignore
    HAS_LC_HF = False

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def resolve_embeddings(
    embedding_model_name: str,
    cache_dir: Optional[Path] = None,
    model_path: Optional[Path] = None,
) -> object:
    """
    Returns a LangChain-compatible HuggingFace embedding model.
    Uses BAAI/bge-m3 by default for state-of-the-art multilingual embeddings.
    """
    if not (HAS_LC_HF and HuggingFaceEmbeddings is not None):
        raise ImportError(
            "HuggingFace embeddings backend is unavailable. "
            "Please install: pip install langchain-huggingface sentence-transformers"
        )

    # Handle local model path
    if model_path:
        resolved_path = Path(model_path).expanduser().resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Embedding model path does not exist: {resolved_path}")
        logger.info(
            "embeddings.huggingface.local_path",
            extra={"path": str(resolved_path)},
        )
        kwargs = {"model_name": str(resolved_path)}
        if cache_dir:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            kwargs["cache_folder"] = str(cache_dir)
        return HuggingFaceEmbeddings(**kwargs)

    # Setup cache folder
    cache_folder = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_folder = str(cache_dir)

    logger.info(
        "embeddings.huggingface",
        extra={"model": embedding_model_name, "cache": cache_folder},
    )

    kwargs = {"model_name": embedding_model_name}
    if cache_folder:
        kwargs["cache_folder"] = cache_folder

    return HuggingFaceEmbeddings(**kwargs)

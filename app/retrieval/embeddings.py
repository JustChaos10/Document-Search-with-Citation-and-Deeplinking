from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

try:  # pragma: no cover - optional dependency
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
    HAS_LC_HF = True
except ImportError:  # pragma: no cover - handled at runtime
    HuggingFaceEmbeddings = None  # type: ignore
    HAS_LC_HF = False

from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def resolve_embeddings(
    embedding_model_name: str,
    gemini_api_key: str,
    cache_dir: Optional[Path] = None,
    model_path: Optional[Path] = None,
) -> object:
    """
    Returns a LangChain-compatible embedding model. Defaults to a HuggingFace model for
    locality, but can be configured to use Google embeddings when required.
    """
    if model_path:
        if not (HAS_LC_HF and HuggingFaceEmbeddings is not None):
            raise ImportError(
                "HuggingFace embeddings backend is unavailable; cannot load a local model."
            )
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

    if embedding_model_name.startswith("models/"):
        logger.info("embeddings.google", extra={"model": embedding_model_name})
        return GoogleGenerativeAIEmbeddings(
            model=embedding_model_name,
            google_api_key=gemini_api_key,
        )

    cache_folder = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_folder = str(cache_dir)

    if not (HAS_LC_HF and HuggingFaceEmbeddings is not None):
        logger.warning(
            "embeddings.huggingface_unavailable",
            extra={"model": embedding_model_name},
        )
        return _google_embeddings_fallback(gemini_api_key, embedding_model_name)

    logger.info(
        "embeddings.huggingface",
        extra={"model": embedding_model_name, "cache": cache_folder},
    )
    kwargs = {"model_name": embedding_model_name}
    if cache_folder:
        kwargs["cache_folder"] = cache_folder

    return _FailoverEmbeddings(
        primary_factory=lambda: HuggingFaceEmbeddings(**kwargs),
        fallback_factory=lambda: _google_embeddings_fallback(gemini_api_key, embedding_model_name),
    )


def _google_embeddings_fallback(gemini_api_key: str, requested_model: str) -> object:
    fallback_model = "models/text-embedding-004"
    logger.info(
        "embeddings.google_fallback",
        extra={"requested_model": requested_model, "fallback_model": fallback_model},
    )
    return GoogleGenerativeAIEmbeddings(model=fallback_model, google_api_key=gemini_api_key)


class _FailoverEmbeddings:
    """Embeddings wrapper that fails over from a local model to a remote fallback on demand."""

    def __init__(
        self,
        primary_factory: Callable[[], object],
        fallback_factory: Callable[[], object],
    ) -> None:
        self._primary_factory = primary_factory
        self._fallback_factory = fallback_factory
        self._primary = None
        self._fallback = None
        self._use_fallback = False

    def embed_documents(self, texts):
        return self._call("embed_documents", texts)

    def embed_query(self, text):
        return self._call("embed_query", text)

    def _call(self, method: str, *args, **kwargs):
        if not self._use_fallback:
            primary = self._ensure_primary()
            if primary:
                try:
                    return getattr(primary, method)(*args, **kwargs)
                except Exception as exc:  # pragma: no cover - runtime failure
                    logger.warning(
                        "embeddings.primary_failed",
                        extra={"method": method, "error": str(exc)},
                    )
                    self._use_fallback = True
                    self._primary = None
        fallback = self._ensure_fallback()
        return getattr(fallback, method)(*args, **kwargs)

    def _ensure_primary(self):
        if self._use_fallback:
            return None
        if self._primary is None:
            try:
                self._primary = self._primary_factory()
            except Exception as exc:  # pragma: no cover - instantiation failure
                logger.warning(
                    "embeddings.primary_init_failed",
                    extra={"error": str(exc)},
                )
                self._use_fallback = True
                self._primary = None
        return self._primary

    def _ensure_fallback(self):
        if self._fallback is None:
            self._fallback = self._fallback_factory()
        return self._fallback

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from langchain_core.documents import Document

# Disable Chroma telemetry by default for local-first deployments.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")

try:
    from langchain_community.vectorstores import FAISS

    HAS_FAISS = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_FAISS = False

try:  # pragma: no cover - optional dependency
    from langchain_chroma import Chroma as ModernChroma
    from chromadb.config import Settings as ChromaSettings

    HAS_MODERN_CHROMA = True
except ImportError:
    ModernChroma = None  # type: ignore
    ChromaSettings = None  # type: ignore
    HAS_MODERN_CHROMA = False

try:
    from langchain_community.vectorstores import Chroma as LegacyChroma
except ImportError:  # pragma: no cover - optional dependency
    LegacyChroma = None  # type: ignore

logger = logging.getLogger(__name__)


class VectorStoreManager:
    def __init__(
        self,
        embeddings: object,
        storage_path: Path,
        backend_preference: str = "auto",
    ) -> None:
        self.embeddings = embeddings
        self.storage_path = storage_path
        self.backend_preference = backend_preference
        self._store = None
        self._backend = None

    @property
    def backend(self) -> str:
        if not self._backend:
            raise RuntimeError("Vector store backend is not initialized.")
        return self._backend

    def load(self) -> None:
        backend = self._resolve_backend()
        if backend == "faiss":
            self._load_faiss()
        else:
            self._load_chroma()

    def _resolve_backend(self) -> str:
        preference = self.backend_preference
        if preference == "faiss" and not self._faiss_available():
            logger.warning("FAISS requested but not available. Falling back to Chroma.")
            preference = "chroma"
        if preference == "auto":
            if self._faiss_available():
                preference = "faiss"
            else:
                preference = "chroma"
        return preference

    def _faiss_available(self) -> bool:
        if not HAS_FAISS:
            return False
        return importlib.util.find_spec("faiss") is not None

    def _load_faiss(self) -> None:
        index_dir = self.storage_path
        index_dir.mkdir(parents=True, exist_ok=True)
        existing = any(index_dir.iterdir())
        try:
            if existing:
                logger.info("vectorstore.load.faiss", extra={"path": str(index_dir)})
                self._store = FAISS.load_local(
                    str(index_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            else:
                logger.info("vectorstore.init.faiss", extra={"path": str(index_dir)})
                self._store = None
            self._backend = "faiss"
        except ImportError as exc:
            logger.warning(
                "vectorstore.faiss_unavailable_at_load",
                extra={"path": str(index_dir), "error": str(exc)},
            )
            self._load_chroma()

    def _load_chroma(self) -> None:
        persist_dir = self.storage_path
        persist_dir.mkdir(parents=True, exist_ok=True)
        logger.info("vectorstore.load.chroma", extra={"path": str(persist_dir)})

        if HAS_MODERN_CHROMA and ModernChroma is not None and ChromaSettings is not None:
            client_settings = ChromaSettings(
                anonymized_telemetry=False,
                is_persistent=True,
                allow_reset=True,
                persist_directory=str(persist_dir),
            )
            self._store = ModernChroma(
                embedding_function=self.embeddings,
                persist_directory=str(persist_dir),
                collection_name="documents",
                client_settings=client_settings,
            )
        elif LegacyChroma is not None:
            self._store = LegacyChroma(
                embedding_function=self.embeddings,
                persist_directory=str(persist_dir),
                collection_name="documents",
            )
        else:
            raise ImportError(
                "Chroma vector store is unavailable. Install `langchain-chroma` or "
                "`langchain-community` with Chroma extras."
            )
        self._backend = "chroma"

    def add_documents(self, documents: Iterable[Document]) -> None:
        backend = self.backend
        docs_list = list(documents)
        if not docs_list:
            logger.info("vectorstore.add.skip", extra={"reason": "empty_documents"})
            return

        logger.info("vectorstore.add", extra={"count": len(docs_list), "backend": backend})

        if backend == "faiss":
            try:
                if self._store is None:
                    self._store = FAISS.from_documents(docs_list, self.embeddings)
                else:
                    self._store.add_documents(docs_list)
                self._store.save_local(str(self.storage_path))
                return
            except ImportError as exc:
                logger.warning(
                    "vectorstore.faiss_add_failed",
                    extra={"error": str(exc), "fallback": "chroma"},
                )
                self._switch_to_chroma()
                backend = self.backend  # update backend for chroma flow
                logger.info("vectorstore.fallback", extra={"backend": backend})

        if self._store is None:
            raise RuntimeError("Vector store not loaded.")

        self._store.add_documents(docs_list)
        if hasattr(self._store, "persist"):
            self._store.persist()

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        if self._store is None:
            raise RuntimeError("Vector store not loaded.")
        return self._store.similarity_search(query, k=k)

    def similarity_search_with_scores(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, Optional[float]]]:
        if self._store is None:
            raise RuntimeError("Vector store not loaded.")
        if hasattr(self._store, "similarity_search_with_score"):
            results = self._store.similarity_search_with_score(query, k=k)
            normalized: List[Tuple[Document, Optional[float]]] = []
            for doc, score in results:
                try:
                    normalized_score = float(score)
                except (TypeError, ValueError):
                    normalized_score = None
                normalized.append((doc, normalized_score))
            return normalized
        docs = self._store.similarity_search(query, k=k)
        return [(doc, None) for doc in docs]

    def _switch_to_chroma(self) -> None:
        self._store = None
        self._backend = None
        self.backend_preference = "chroma"
        self._load_chroma()

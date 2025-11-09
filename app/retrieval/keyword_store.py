from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    from rank_bm25 import BM25Okapi

    HAS_BM25 = True
except ImportError:  # pragma: no cover - runtime availability
    BM25Okapi = None  # type: ignore
    HAS_BM25 = False

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"[\w']+")


class KeywordIndex:
    """
    Lightweight BM25 keyword index that stores chunk text locally and supports persistence.
    """

    def __init__(self, storage_path: Path) -> None:
        self.storage_path = Path(storage_path)
        self._documents: List[Document] = []
        self._chunk_to_index: dict[str, int] = {}
        self._bm25: Optional[BM25Okapi] = None

        if not HAS_BM25:
            logger.warning("bm25.unavailable", extra={"path": str(self.storage_path)})

    def load(self) -> None:
        if not self.storage_path.exists():
            self._documents = []
            self._chunk_to_index = {}
            self._bm25 = None
            return

        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.error(
                "bm25.load_failed",
                extra={"path": str(self.storage_path), "error": str(exc)},
            )
            self._documents = []
            self._chunk_to_index = {}
            self._bm25 = None
            return

        documents: List[Document] = []
        chunk_to_index: dict[str, int] = {}
        for entry in payload:
            chunk_id = entry.get("chunk_id")
            text = entry.get("text", "")
            metadata = entry.get("metadata") or {}
            if not chunk_id or not text:
                continue
            doc = Document(page_content=text, metadata=metadata)
            chunk_to_index[chunk_id] = len(documents)
            documents.append(doc)

        self._documents = documents
        self._chunk_to_index = chunk_to_index
        self._rebuild()

    def add_documents(self, documents: Iterable[Document]) -> None:
        if not HAS_BM25:
            return

        updated = False
        for doc in documents:
            chunk_id = doc.metadata.get("chunk_id")
            if not chunk_id:
                continue
            cloned_metadata = dict(doc.metadata)
            cloned_metadata.setdefault("retrieval", "bm25")
            cloned_doc = Document(page_content=doc.page_content, metadata=cloned_metadata)

            if chunk_id in self._chunk_to_index:
                index = self._chunk_to_index[chunk_id]
                self._documents[index] = cloned_doc
                updated = True
            else:
                self._chunk_to_index[chunk_id] = len(self._documents)
                self._documents.append(cloned_doc)
                updated = True

        if updated:
            self._rebuild()
            self._persist()

    def search(self, query: str, k: int = 4) -> List[Document]:
        if not HAS_BM25 or not self._bm25:
            return []
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        indexed_scores = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )
        results: List[Document] = []
        for index, score in indexed_scores:
            if len(results) >= k:
                break
            doc = self._documents[index]
            metadata = dict(doc.metadata)
            metadata["bm25_score"] = float(score)
            results.append(Document(page_content=doc.page_content, metadata=metadata))
        return results

    def _rebuild(self) -> None:
        if not HAS_BM25:
            self._bm25 = None
            return
        if not self._documents:
            self._bm25 = None
            return
        tokenized_corpus = [self._tokenize(doc.page_content) for doc in self._documents]
        if not tokenized_corpus:
            self._bm25 = None
            return
        self._bm25 = BM25Okapi(tokenized_corpus)

    def _persist(self) -> None:
        payload = []
        for chunk_id, index in self._chunk_to_index.items():
            doc = self._documents[index]
            payload.append(
                {
                    "chunk_id": chunk_id,
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                }
            )
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        if not text:
            return []
        tokens = TOKEN_PATTERN.findall(text.lower())
        return tokens

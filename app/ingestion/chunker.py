from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


@dataclass
class ChunkerConfig:
    chunk_size: int = 800
    chunk_overlap: int = 200
    chunk_size_ar: int = 1200
    chunk_overlap_ar: int = 300
    separators: List[str] = None

    def __post_init__(self) -> None:
        if self.separators is None:
            self.separators = ["\n\n", "\n", ". ", " ", ""]


class DocumentChunker:
    def __init__(self, config: ChunkerConfig) -> None:
        self.config = config

        # English/default splitter
        self._splitter_en = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
        )

        # Arabic splitter with larger chunk size
        self._splitter_ar = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size_ar,
            chunk_overlap=config.chunk_overlap_ar,
            separators=config.separators,
        )

    def chunk(self, documents: Iterable[Document]) -> List[Document]:
        chunks: List[Document] = []
        for doc in documents:
            # Select splitter based on document language
            doc_language = doc.metadata.get("language", "en")
            splitter = self._splitter_ar if doc_language == "ar" else self._splitter_en

            splits = splitter.split_text(doc.page_content)
            for index, chunk_text in enumerate(splits):
                metadata = dict(doc.metadata)
                metadata["chunk"] = index
                chunks.append(Document(page_content=chunk_text, metadata=metadata))
        return chunks

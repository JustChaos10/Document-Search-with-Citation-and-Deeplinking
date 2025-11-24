from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.ingestion.semantic_chunker import SemanticChunker


@dataclass
class ChunkerConfig:
    chunk_size: int = 800
    chunk_overlap: int = 200
    chunk_size_ar: int = 1200
    chunk_overlap_ar: int = 300
    separators: List[str] = None
    use_semantic_chunking: bool = True

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

        # Semantic chunkers
        self._semantic_chunker_en = SemanticChunker(
            target_chunk_size=config.chunk_size,
            min_chunk_size=config.chunk_size // 4,
            max_chunk_size=config.chunk_size * 2
        )
        self._semantic_chunker_ar = SemanticChunker(
            target_chunk_size=config.chunk_size_ar,
            min_chunk_size=config.chunk_size_ar // 4,
            max_chunk_size=config.chunk_size_ar * 2
        )

    def chunk(self, documents: Iterable[Document]) -> List[Document]:
        chunks: List[Document] = []
        doc_list = list(documents)

        if self.config.use_semantic_chunking:
            # Use semantic chunking
            for doc in doc_list:
                doc_language = doc.metadata.get("language", "en")
                semantic_chunker = (
                    self._semantic_chunker_ar if doc_language == "ar"
                    else self._semantic_chunker_en
                )
                doc_chunks = semantic_chunker.chunk([doc])
                chunks.extend(doc_chunks)
        else:
            # Use traditional recursive text splitter
            for doc in doc_list:
                doc_language = doc.metadata.get("language", "en")
                splitter = self._splitter_ar if doc_language == "ar" else self._splitter_en

                splits = splitter.split_text(doc.page_content)
                for index, chunk_text in enumerate(splits):
                    metadata = dict(doc.metadata)
                    metadata["chunk"] = index
                    chunks.append(Document(page_content=chunk_text, metadata=metadata))

        return chunks

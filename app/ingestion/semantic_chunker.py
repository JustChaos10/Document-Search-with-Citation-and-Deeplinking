from __future__ import annotations

import logging
from typing import List
import re

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class SemanticChunker:
    """
    Chunks documents based on semantic boundaries (sentences, paragraphs)
    rather than fixed character counts.
    """

    def __init__(
        self,
        target_chunk_size: int = 800,
        min_chunk_size: int = 200,
        max_chunk_size: int = 1500
    ):
        """
        Initialize semantic chunker.

        Args:
            target_chunk_size: Target size for chunks (soft limit)
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size (hard limit)
        """
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents based on semantic boundaries.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents
        """
        chunks = []

        for doc in documents:
            doc_chunks = self._chunk_document(doc)
            chunks.extend(doc_chunks)

        logger.info(
            "semantic_chunking.complete",
            extra={
                "input_docs": len(documents),
                "output_chunks": len(chunks),
                "avg_chunk_size": sum(len(c.page_content) for c in chunks) // len(chunks) if chunks else 0
            }
        )

        return chunks

    def _chunk_document(self, doc: Document) -> List[Document]:
        """
        Chunk a single document based on semantic boundaries.

        Args:
            doc: Document to chunk

        Returns:
            List of chunk documents
        """
        text = doc.page_content

        # First, split into paragraphs
        paragraphs = self._split_paragraphs(text)

        # Then create semantic chunks
        chunks = []
        current_chunk = []
        current_size = 0

        for paragraph in paragraphs:
            paragraph_size = len(paragraph)

            # If adding this paragraph exceeds max size, finalize current chunk
            if current_size + paragraph_size > self.max_chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(chunk_text)
                current_chunk = []
                current_size = 0

            # If paragraph itself is larger than max, split it by sentences
            if paragraph_size > self.max_chunk_size:
                sentences = self._split_sentences(paragraph)
                for sentence in sentences:
                    sentence_size = len(sentence)

                    if current_size + sentence_size > self.target_chunk_size and current_chunk:
                        chunk_text = "\n\n".join(current_chunk)
                        chunks.append(chunk_text)
                        current_chunk = []
                        current_size = 0

                    current_chunk.append(sentence)
                    current_size += sentence_size
            else:
                # Add paragraph to current chunk
                current_chunk.append(paragraph)
                current_size += paragraph_size

                # If we've reached target size, finalize chunk
                if current_size >= self.target_chunk_size:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append(chunk_text)
                    current_chunk = []
                    current_size = 0

        # Add remaining content
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)
            elif chunks:
                # Append to last chunk if too small
                chunks[-1] += "\n\n" + chunk_text
            else:
                # Keep it if it's the only content
                chunks.append(chunk_text)

        # Create Document objects for each chunk
        chunk_documents = []
        for index, chunk_text in enumerate(chunks):
            metadata = dict(doc.metadata)
            metadata["chunk"] = index
            metadata["chunk_method"] = "semantic"
            chunk_documents.append(
                Document(page_content=chunk_text, metadata=metadata)
            )

        return chunk_documents

    def _split_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.

        Args:
            text: Text to split

        Returns:
            List of paragraphs
        """
        # Split on double newlines or more
        paragraphs = re.split(r'\n\s*\n+', text)

        # Clean and filter
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        return paragraphs

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Simple sentence splitting (handles most cases)
        # Split on period, exclamation, or question mark followed by space and capital letter
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)

        # Clean sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

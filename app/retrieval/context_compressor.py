from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ContextualCompressor:
    """
    Compresses retrieved context by removing redundant information
    and keeping only the most relevant portions for the query.
    """

    def __init__(self, max_chars_per_doc: int = 600):
        """
        Initialize the contextual compressor.

        Args:
            max_chars_per_doc: Maximum characters to keep per document
        """
        self.max_chars_per_doc = max_chars_per_doc

    def compress_documents(
        self,
        query: str,
        documents: List[Document]
    ) -> List[Document]:
        """
        Compress documents by extracting the most relevant portions.

        Args:
            query: The user's query
            documents: List of retrieved documents

        Returns:
            List of compressed documents with relevant excerpts
        """
        compressed_docs = []
        query_terms = set(query.lower().split())

        for doc in documents:
            # Extract most relevant sentences based on query overlap
            content = doc.page_content
            compressed_content = self._extract_relevant_excerpts(
                content,
                query_terms
            )

            # Create new document with compressed content
            compressed_doc = Document(
                page_content=compressed_content,
                metadata={
                    **doc.metadata,
                    "original_length": len(content),
                    "compressed_length": len(compressed_content),
                    "compression_ratio": round(len(compressed_content) / len(content), 2)
                }
            )
            compressed_docs.append(compressed_doc)

        logger.info(
            "context.compression_complete",
            extra={
                "original_docs": len(documents),
                "compressed_docs": len(compressed_docs),
                "avg_compression_ratio": sum(
                    d.metadata.get("compression_ratio", 1.0) for d in compressed_docs
                ) / len(compressed_docs) if compressed_docs else 0
            }
        )

        return compressed_docs

    def _extract_relevant_excerpts(
        self,
        text: str,
        query_terms: set
    ) -> str:
        """
        Extract the most relevant portions of text based on query terms.

        Args:
            text: Original document text
            query_terms: Set of query terms to match

        Returns:
            Compressed text with most relevant excerpts
        """
        # Split into sentences (simple split by periods)
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        # Score each sentence by query term overlap
        scored_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for term in query_terms if term in sentence_lower)
            scored_sentences.append((score, sentence))

        # Sort by score (descending) and take top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        # Build compressed content
        selected = []
        current_length = 0

        for score, sentence in scored_sentences:
            if current_length + len(sentence) > self.max_chars_per_doc:
                break
            if score > 0:  # Only include sentences with query term matches
                selected.append(sentence)
                current_length += len(sentence)

        # If no sentences matched, take the first portion
        if not selected:
            return text[:self.max_chars_per_doc]

        # Join selected sentences
        return '. '.join(selected) + '.'

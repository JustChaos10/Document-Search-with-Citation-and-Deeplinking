from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Optional dependencies for enhanced chunking
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    tiktoken = None  # type: ignore

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class ChunkerConfig:
    chunk_size: int = 800
    chunk_overlap: int = 200
    chunk_size_ar: int = 1200
    chunk_overlap_ar: int = 300
    separators: List[str] = None
    # Token-based chunking (more accurate for LLM context)
    use_token_based: bool = True
    # Tokens per chunk (for GPT-4/Gemini: ~500 tokens ≈ good chunk)
    chunk_tokens: int = 500
    chunk_tokens_overlap: int = 100
    chunk_tokens_ar: int = 600  # Arabic needs slightly more
    chunk_tokens_overlap_ar: int = 120
    # Semantic chunking - split at natural boundaries with embedding similarity
    use_semantic_chunking: bool = True
    semantic_threshold: float = 0.65  # Similarity threshold for merging sentences
    # Parent document retrieval - store larger context
    create_parent_chunks: bool = True
    parent_chunk_tokens: int = 2000  # ~1500 words of context

    def __post_init__(self) -> None:
        if self.separators is None:
            self.separators = ["\n\n", "\n", ". ", " ", ""]


class DocumentChunker:
    def __init__(self, config: ChunkerConfig, embeddings: Optional[object] = None) -> None:
        self.config = config
        self.embeddings = embeddings  # For semantic chunking

        # Token encoder (for token-based chunking)
        self._encoder = None
        if HAS_TIKTOKEN and config.use_token_based:
            try:
                # cl100k_base is used by GPT-4 and works well for multilingual
                self._encoder = tiktoken.get_encoding("cl100k_base")
                logger.info("chunker.token_based_enabled", extra={"encoding": "cl100k_base"})
            except Exception as exc:
                logger.warning("chunker.tiktoken_init_failed", extra={"error": str(exc)})
                self._encoder = None

        # Fallback: Character-based splitters
        self._splitter_en = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
        )

        self._splitter_ar = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size_ar,
            chunk_overlap=config.chunk_overlap_ar,
            separators=config.separators,
        )

    def chunk(self, documents: Iterable[Document]) -> List[Document]:
        chunks: List[Document] = []
        for doc in documents:
            doc_language = doc.metadata.get("language", "en")

            # Strategy selection based on configuration
            if self.config.use_token_based and self._encoder:
                doc_chunks = self._token_based_chunk(doc, doc_language)
            elif self.config.use_semantic_chunking and self.embeddings:
                doc_chunks = self._semantic_chunk(doc, doc_language)
            else:
                # Fallback to character-based chunking
                doc_chunks = self._character_based_chunk(doc, doc_language)

            chunks.extend(doc_chunks)

        return chunks

    def _token_based_chunk(self, doc: Document, language: str) -> List[Document]:
        """Split document based on token count for accurate LLM context limits."""
        if not self._encoder:
            return self._character_based_chunk(doc, language)

        # Get token limits based on language
        if language == "ar":
            chunk_tokens = self.config.chunk_tokens_ar
            overlap_tokens = self.config.chunk_tokens_overlap_ar
        else:
            chunk_tokens = self.config.chunk_tokens
            overlap_tokens = self.config.chunk_tokens_overlap

        text = doc.page_content
        parent_text = text  # Store full document for parent retrieval

        # Split on sentences first for natural boundaries
        sentences = self._split_into_sentences(text, language)

        chunks: List[Document] = []
        current_chunk_sentences: List[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(self._encoder.encode(sentence))

            # If adding this sentence exceeds chunk size, finalize current chunk
            if current_tokens + sentence_tokens > chunk_tokens and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(self._create_chunk_document(
                    chunk_text, doc, len(chunks), parent_text, language
                ))

                # Keep overlap sentences
                overlap_count = 0
                overlap_sentences = []
                for s in reversed(current_chunk_sentences):
                    s_tokens = len(self._encoder.encode(s))
                    if overlap_count + s_tokens <= overlap_tokens:
                        overlap_sentences.insert(0, s)
                        overlap_count += s_tokens
                    else:
                        break

                current_chunk_sentences = overlap_sentences
                current_tokens = overlap_count

            current_chunk_sentences.append(sentence)
            current_tokens += sentence_tokens

        # Add final chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(self._create_chunk_document(
                chunk_text, doc, len(chunks), parent_text, language
            ))

        return chunks

    def _semantic_chunk(self, doc: Document, language: str) -> List[Document]:
        """Split document using semantic similarity to find natural boundaries."""
        if not self.embeddings or not HAS_NUMPY:
            return self._token_based_chunk(doc, language)

        text = doc.page_content
        parent_text = text
        sentences = self._split_into_sentences(text, language)

        if len(sentences) <= 1:
            return [self._create_chunk_document(text, doc, 0, parent_text, language)]

        try:
            # Compute embeddings for all sentences
            sentence_embeddings = self.embeddings.embed_documents(sentences)
            sentence_embeddings = np.array(sentence_embeddings)

            # Compute cosine similarities between consecutive sentences
            similarities = []
            for i in range(len(sentence_embeddings) - 1):
                similarity = self._cosine_similarity(
                    sentence_embeddings[i],
                    sentence_embeddings[i + 1]
                )
                similarities.append(similarity)

            # Find split points where similarity drops below threshold
            split_indices = [0]
            for i, sim in enumerate(similarities):
                if sim < self.config.semantic_threshold:
                    split_indices.append(i + 1)
            split_indices.append(len(sentences))

            # Create chunks from split points
            chunks: List[Document] = []
            for i in range(len(split_indices) - 1):
                start_idx = split_indices[i]
                end_idx = split_indices[i + 1]
                chunk_sentences = sentences[start_idx:end_idx]
                chunk_text = " ".join(chunk_sentences)

                # Respect token limits even with semantic chunking
                if self._encoder:
                    max_tokens = self.config.chunk_tokens_ar if language == "ar" else self.config.chunk_tokens
                    chunk_tokens = len(self._encoder.encode(chunk_text))

                    # If chunk too large, fall back to token-based splitting
                    if chunk_tokens > max_tokens * 1.5:
                        # Create a temporary document for this chunk
                        temp_doc = Document(page_content=chunk_text, metadata=doc.metadata)
                        sub_chunks = self._token_based_chunk(temp_doc, language)
                        for sub_chunk in sub_chunks:
                            chunks.append(self._create_chunk_document(
                                sub_chunk.page_content, doc, len(chunks), parent_text, language
                            ))
                        continue

                chunks.append(self._create_chunk_document(
                    chunk_text, doc, len(chunks), parent_text, language
                ))

            return chunks

        except Exception as exc:
            logger.warning("chunker.semantic_chunking_failed", extra={"error": str(exc)})
            return self._token_based_chunk(doc, language)

    def _character_based_chunk(self, doc: Document, language: str) -> List[Document]:
        """Fallback to character-based chunking."""
        splitter = self._splitter_ar if language == "ar" else self._splitter_en
        splits = splitter.split_text(doc.page_content)

        chunks: List[Document] = []
        parent_text = doc.page_content

        for index, chunk_text in enumerate(splits):
            chunks.append(self._create_chunk_document(
                chunk_text, doc, index, parent_text, language
            ))

        return chunks

    def _create_chunk_document(
        self,
        chunk_text: str,
        original_doc: Document,
        chunk_index: int,
        parent_text: str,
        language: str
    ) -> Document:
        """Create a chunk document with parent context metadata."""
        metadata = dict(original_doc.metadata)
        metadata["chunk"] = chunk_index
        metadata["language"] = language

        # Store parent document context for retrieval
        if self.config.create_parent_chunks:
            metadata["parent_text"] = parent_text

            # Calculate parent chunk window (broader context around this chunk)
            parent_window = self._get_parent_window(
                chunk_text, parent_text, language
            )
            if parent_window and parent_window != chunk_text:
                metadata["parent_window"] = parent_window

        # Store token count if available
        if self._encoder:
            metadata["token_count"] = len(self._encoder.encode(chunk_text))

        return Document(page_content=chunk_text, metadata=metadata)

    def _get_parent_window(self, chunk_text: str, full_text: str, language: str) -> Optional[str]:
        """Extract a larger context window around the chunk from the parent document."""
        if not self._encoder or not self.config.create_parent_chunks:
            return None

        parent_tokens = self.config.parent_chunk_tokens

        # Find chunk position in full text
        chunk_start = full_text.find(chunk_text)
        if chunk_start == -1:
            return None

        chunk_end = chunk_start + len(chunk_text)

        # Expand context before and after
        sentences = self._split_into_sentences(full_text, language)

        # Find sentences that overlap with or surround the chunk
        window_sentences: List[str] = []
        window_tokens = 0

        for sentence in sentences:
            sentence_start = full_text.find(sentence)
            sentence_end = sentence_start + len(sentence)

            # Check if sentence overlaps with or is near the chunk
            if (sentence_start <= chunk_end and sentence_end >= chunk_start):
                sentence_tokens = len(self._encoder.encode(sentence))
                if window_tokens + sentence_tokens <= parent_tokens:
                    window_sentences.append(sentence)
                    window_tokens += sentence_tokens

        if window_sentences:
            return " ".join(window_sentences)

        return None

    def _split_into_sentences(self, text: str, language: str) -> List[str]:
        """Split text into sentences, handling both English and Arabic."""
        if language == "ar":
            # Arabic sentence boundaries: period, question mark, exclamation, Arabic question mark
            pattern = r'[.!?؟]+'
        else:
            # English sentence boundaries
            pattern = r'[.!?]+'

        # Split on sentence boundaries but keep delimiter
        parts = re.split(f'({pattern})', text)

        sentences: List[str] = []
        current = ""

        for part in parts:
            current += part
            if re.match(pattern, part):
                sentences.append(current.strip())
                current = ""

        # Add any remaining text
        if current.strip():
            sentences.append(current.strip())

        # Filter out empty sentences and very short ones
        sentences = [s for s in sentences if len(s.strip()) > 10]

        return sentences if sentences else [text]

    @staticmethod
    def _cosine_similarity(vec1, vec2) -> float:
        """Compute cosine similarity between two vectors."""
        if not HAS_NUMPY:
            return 0.5  # Default neutral value

        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

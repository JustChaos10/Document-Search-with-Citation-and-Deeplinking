from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document

from app.config import AppConfig
from app.audio import SpeechToTextService
from app.ingestion.chunker import ChunkerConfig, DocumentChunker
from app.ingestion.extractors import DocumentExtractor
from app.nlp import detect_language
from app.retrieval.embeddings import resolve_embeddings
from app.retrieval.vector_store import VectorStoreManager
from app.retrieval.keyword_store import KeywordIndex

logger = logging.getLogger(__name__)


@dataclass
class IngestionSummary:
    files_processed: int
    sections_extracted: int
    chunks_written: int


class IngestionPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        try:
            self.stt_service = SpeechToTextService.from_config(config)
        except ImportError as exc:  # pragma: no cover - runtime configuration
            logger.warning("whisper.unavailable", extra={"error": str(exc)})
            self.stt_service = None
        self.extractor = DocumentExtractor(stt_service=self.stt_service)

        # Initialize embeddings first so chunker can use them for semantic chunking
        embeddings = resolve_embeddings(
            config.embedding_model_name,
            config.gemini_api_key,
            cache_dir=config.embedding_cache_dir,
            model_path=config.embedding_model_path,
        )

        # Create chunker with embeddings for semantic chunking
        self.chunker = DocumentChunker(
            ChunkerConfig(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                chunk_size_ar=config.chunk_size_ar,
                chunk_overlap_ar=config.chunk_overlap_ar,
                use_token_based=True,
                use_semantic_chunking=True,
                create_parent_chunks=True,
            ),
            embeddings=embeddings
        )
        self.vector_store = VectorStoreManager(
            embeddings=embeddings,
            storage_path=config.vector_store_path,
            backend_preference=config.vector_store_backend,
        )
        self.vector_store.load()
        self.keyword_index = None
        if config.bm25_index_path:
            self.keyword_index = KeywordIndex(config.bm25_index_path)
            self.keyword_index.load()

    def ingest(self, target_files: Iterable[Path] | None = None) -> IngestionSummary:
        uploads_dir = self.config.uploads_dir
        files = (
            sorted(target_files, key=lambda p: p.name.lower())
            if target_files
            else sorted(self._iter_supported_files(uploads_dir), key=lambda p: p.name.lower())
        )

        if not files:
            logger.warning("ingestion.no_files", extra={"uploads_dir": str(uploads_dir)})
            return IngestionSummary(files_processed=0, sections_extracted=0, chunks_written=0)

        files_processed = 0
        sections_extracted = 0
        chunks_written = 0
        for file_path in files:
            files_processed += 1
            result = self.extractor.extract(file_path)
            sections_extracted += len(result.documents)

            chunked_docs = self.chunker.chunk(result.documents)
            chunked_docs = self._enrich_chunks(chunked_docs)
            chunks_written += len(chunked_docs)
            self.vector_store.add_documents(chunked_docs)
            if self.keyword_index:
                self.keyword_index.add_documents(chunked_docs)

        logger.info(
            "ingestion.complete",
            extra={
                "files_processed": files_processed,
                "sections": sections_extracted,
                "chunks": chunks_written,
                "backend": self.vector_store.backend,
            },
        )
        return IngestionSummary(
            files_processed=files_processed,
            sections_extracted=sections_extracted,
            chunks_written=chunks_written,
        )

    def _iter_supported_files(self, uploads_dir: Path) -> Iterable[Path]:
        supported_suffixes = DocumentExtractor.SUPPORTED_EXTENSIONS
        for path in uploads_dir.iterdir():
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in supported_suffixes:
                continue
            if suffix in DocumentExtractor.AUDIO_EXTENSIONS and not self.stt_service:
                logger.warning(
                    "ingestion.audio_skipped_no_stt",
                    extra={"file": str(path)},
                )
                continue
            if suffix == "":
                continue
            if suffix in supported_suffixes:
                yield path

    def _enrich_chunks(self, chunks: List[Document]) -> List[Document]:
        enriched = []
        for chunk in chunks:
            metadata = dict(chunk.metadata)
            metadata["char_count"] = len(chunk.page_content)
            metadata["word_count"] = len(chunk.page_content.split())
            metadata["chunk_id"] = f"{metadata.get('source','unknown')}:{metadata.get('chunk',0)}"
            # Use page-level language detection, fallback to chunk-level if missing
            detected_language = metadata.get("language") or detect_language(chunk.page_content)
            if detected_language:
                metadata["language"] = detected_language
            path = metadata.get("path")
            if path:
                try:
                    relative_path = Path(path).resolve().relative_to(self.config.base_dir.resolve())
                    metadata["relative_path"] = str(relative_path).replace("\\", "/")
                except ValueError:
                    metadata["relative_path"] = metadata.get("source")
            relative_path = metadata.get("relative_path", metadata.get("source", "document"))
            sanitized_relative = str(relative_path).replace("\\", "/")
            metadata["relative_path"] = sanitized_relative
            viewer_path = metadata.get("viewer_path")
            if viewer_path:
                try:
                    viewer_relative = Path(viewer_path).resolve().relative_to(self.config.base_dir.resolve())
                    metadata["viewer_relative_path"] = str(viewer_relative).replace("\\", "/")
                except ValueError:
                    metadata["viewer_relative_path"] = metadata.get("relative_path")
            if metadata.get("viewer_page") and not metadata.get("page"):
                metadata["page"] = metadata["viewer_page"]
            section_hint = metadata.get("section_label") or metadata.get("section_heading")
            metadata["page_label"] = (
                metadata.get("page_label")
                or section_hint
                or self._derive_page_label(metadata)
            )
            metadata["page_window"] = (
                metadata.get("page_window")
                or section_hint
                or metadata.get("page_label")
            )
            if "section_heading" not in metadata or not metadata.get("section_heading"):
                heading = self._extract_heading(chunk.page_content)
                if heading:
                    metadata["section_heading"] = heading
            metadata["preview"] = metadata.get("preview") or self._make_preview(chunk.page_content)
            enriched.append(Document(page_content=chunk.page_content, metadata=metadata))
        return enriched

    @staticmethod
    def _derive_page_label(metadata: dict) -> str | None:
        if "page" in metadata:
            return f"Page {metadata['page']}"
        if "slide" in metadata:
            return f"Slide {metadata['slide']}"
        if "sheet" in metadata:
            return f"Sheet {metadata['sheet']}"
        doc_type = metadata.get("doc_type")
        if doc_type:
            return str(doc_type).replace("_", " ").title()
        return None

    @staticmethod
    def _extract_heading(text: str) -> str | None:
        if not text:
            return None
        for line in text.splitlines():
            candidate = line.strip()
            if not candidate:
                continue
            if len(candidate) > 120:
                candidate = candidate[:117].rstrip() + "..."
            return candidate
        return None

    @staticmethod
    def _make_preview(text: str, width: int = 220) -> str | None:
        if not text:
            return None
        normalized = " ".join(text.split())
        if not normalized:
            return None
        if len(normalized) <= width:
            return normalized
        return f"{normalized[: width - 1].rstrip()}…"

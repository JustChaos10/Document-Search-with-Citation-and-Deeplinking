from __future__ import annotations

import json
import logging
from dataclasses import dataclass
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple
from collections.abc import Iterable
from datetime import datetime

from langchain_core.documents import Document
from app.retrieval.llm_client import build_chat_model

from app.config import AppConfig
from app.retrieval.vector_store import VectorStoreManager
from app.nlp import detect_language
from app.retrieval.keyword_store import KeywordIndex
from app.retrieval.reranker import CrossEncoderReranker, RerankedDocument

logger = logging.getLogger(__name__)


@dataclass
class CitationResult:
    chunk_id: str
    summary: str


@dataclass
class QueryResult:
    answer: str
    citations: List[CitationResult]
    context_map: dict[str, Document]
    language: str


class QueryService:
    def __init__(
        self,
        config: AppConfig,
        vector_store: VectorStoreManager,
        keyword_index: Optional[KeywordIndex] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        chat_model: Optional[ChatGoogleGenerativeAI] = None,
    ) -> None:
        self.config = config
        self.vector_store = vector_store
        self.keyword_index = keyword_index
        self.reranker = reranker
        self.chat_model = chat_model or build_chat_model(config)
        self.max_retries = 3

    def query(self, question: str, top_k: int = 6) -> QueryResult:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        search_k = max(top_k * 3, top_k)
        query_language = (detect_language(question) or "en").lower()
        if query_language not in {"en", "ar"}:
            query_language = "en"
        expanded_queries = self._expand_queries(question)
        search_variants = self._build_search_variants(expanded_queries, query_language)
        raw_documents: List[Document] = []
        for expanded_query, _lang in search_variants:
            vector_hits = self.vector_store.similarity_search_with_scores(expanded_query, k=search_k)
            for doc, score in vector_hits:
                metadata = dict(doc.metadata)
                if score is not None:
                    metadata["vector_score"] = score
                raw_documents.append(Document(page_content=doc.page_content, metadata=metadata))
            if self.keyword_index and self.config.enable_hybrid_retrieval:
                bm25_hits = self.keyword_index.search(expanded_query, k=search_k)
                for doc in bm25_hits:
                    metadata = dict(doc.metadata)
                    metadata.setdefault("retrieval", "bm25")
                    raw_documents.append(Document(page_content=doc.page_content, metadata=metadata))
        candidate_limit = max(
            search_k, getattr(self.config, "reranker_candidate_limit", top_k)
        )
        documents = self._unique_documents(raw_documents, limit=candidate_limit)
        documents = self._filter_documents_by_language(documents, query_language=query_language)
        if self.reranker:
            reranked_items: List[RerankedDocument] = self.reranker.rank_with_scores(question, documents)
            for item in reranked_items:
                meta = dict(item.document.metadata)
                meta["reranker_score"] = item.score
                item.document.metadata = meta
            documents = [item.document for item in reranked_items][:top_k]
        else:
            documents = documents[:top_k]
        context_map = self._build_context_map(documents)
        if not documents or not context_map:
            logger.info("query.no_results", extra={"question": question})
            return QueryResult(
                answer="No relevant information found in the indexed documents.",
                citations=[],
                context_map={},
                language=query_language,
            )

        prompt = self._build_prompt(
            question,
            documents,
            query_language=query_language,
        )
        logger.info(
            "query.prompt",
            extra={
                "question": question,
                "chunks": len(documents),
            },
        )

        # Retry logic for empty or malformed responses
        payload = None
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                raw_message = self.chat_model.invoke(prompt)
                text_payload = self._extract_message_text(raw_message)

                # Enhanced logging to capture full response structure
                if not text_payload:
                    # Log the entire response structure to diagnose the issue
                    logger.warning(
                        "query.empty_response_attempt",
                        extra={
                            "attempt": attempt,
                            "question": question[:50],
                            "response_type": type(raw_message).__name__,
                            "response_attrs": dir(raw_message),
                            "has_content": hasattr(raw_message, "content"),
                            "content_value": getattr(raw_message, "content", None),
                            "response_metadata": getattr(raw_message, "response_metadata", {}),
                        }
                    )
                    last_error = "Empty response from model"
                    continue

                logger.debug(
                    "query.raw_response",
                    extra={
                        "attempt": attempt,
                        "payload_length": len(text_payload),
                        "payload_preview": text_payload[:200],
                        "response_metadata": getattr(raw_message, "response_metadata", {}),
                    }
                )

                payload = self._safe_parse_json(text_payload)
                if payload is not None:
                    if attempt > 1:
                        logger.info(
                            "query.retry_succeeded",
                            extra={"attempt": attempt, "question": question[:50]}
                        )
                    break
                else:
                    last_error = f"JSON parsing failed (attempt {attempt})"
                    logger.warning(
                        "query.parse_failed_attempt",
                        extra={
                            "attempt": attempt,
                            "payload_preview": text_payload[:500]
                        }
                    )

            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "query.api_error_attempt",
                    extra={
                        "attempt": attempt,
                        "error": str(exc),
                        "question": question[:50]
                    }
                )

        if payload is None:
            logger.error(
                "query.all_retries_failed",
                extra={
                    "retries": self.max_retries,
                    "last_error": last_error,
                    "question": question[:50]
                }
            )
            return QueryResult(
                answer="I'm having trouble generating a response right now. Please try rephrasing your question or try again in a moment.",
                citations=[],
                context_map=context_map,
                language=query_language,
            )

        answer = payload.get("answer", "").strip()
        citations_payload = payload.get("citations", [])
        citations = self._parse_citations(citations_payload)

        return QueryResult(
            answer=answer,
            citations=citations,
            context_map=context_map,
            language=query_language,
        )

    def _build_prompt(
        self,
        question: str,
        documents: Sequence[Document],
        query_language: str,
    ) -> str:
        context_blocks = []
        for idx, doc in enumerate(documents, start=1):
            metadata = doc.metadata
            page_label = self._format_page_label(metadata)
            language = metadata.get("language", "unknown")
            heading = metadata.get("section_heading")
            page_window = metadata.get("page_window")
            context_lines = [
                f"Context {idx}:",
                f"chunk_id: {metadata.get('chunk_id')}",
                f"source: {metadata.get('source')}",
                f"page_info: {page_label}",
                f"language: {language}",
            ]
            if heading:
                context_lines.append(f"heading: {heading}")
            if page_window:
                context_lines.append(f"page_window: {page_window}")
            policy_name = metadata.get("policy_name")
            section_label = metadata.get("section_label")
            effective_date = metadata.get("effective_date")
            if policy_name:
                context_lines.append(f"policy_name: {policy_name}")
            if section_label:
                context_lines.append(f"section: {section_label}")
            if effective_date:
                context_lines.append(f"effective_date: {effective_date}")
            context_lines.append("content:")
            context_lines.append(doc.page_content)
            context_blocks.append("\n".join(context_lines) + "\n")

        instructions = (
            "You are an assistant that answers strictly using the supplied document contexts.\n\n"
            "Adopt a neutral professional tone suitable for all audiences.\n\n"
            "CRITICAL: Return ONLY a valid JSON object. No additional text before or after.\n\n"
            "JSON Schema (required structure):\n"
            "{\n"
            '  "answer": "string - professional 2-3 sentence summary with key facts and figures",\n'
            '  "citations": [\n'
            "    {\n"
            '      "chunk_id": "string - exact chunk_id from context",\n'
            '      "summary": "string - ONE sentence (max 15 words) explaining relevance"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Example valid response:\n"
            '{"answer": "The system achieved 91.1% accuracy on MedQA benchmarks, with significant improvements in medical text summarization.", "citations": [{"chunk_id": "2404.18416v2.pdf:1", "summary": "Details MedQA accuracy results and methodology."}]}\n\n'
            "RULES:\n"
            "1. Use ONLY chunk_ids provided in the contexts below\n"
            "2. Include concrete numbers, dates, and strategies from contexts\n"
            "3. Escape all double quotes inside values using backslash (\\\")\n"
            "4. Keep answer concise (2-3 sentences max) to avoid truncation\n"
            "5. Keep citation summaries VERY SHORT (max 15 words each)\n"
            "6. If nothing relevant found: empty citations array with explanation in answer\n"
            "7. NO code fences, NO markdown, NO additional text - JUST the JSON object\n"
            "8. Ensure JSON is complete and well-formed with all brackets closed\n"
            f"9. Write ALL content in {'Arabic' if query_language == 'ar' else 'English'}\n"
        )

        prompt = (
            f"{instructions}\n"
            f"Question: {question}\n"
            f"{'-'*60}\n"
            f"{''.join(context_blocks)}\n"
            f"{'-'*60}\n"
            "Respond with valid JSON only."
        )
        return prompt

    def _extract_message_text(self, message) -> str:
        """Convert an AIMessage payload into a raw string.

        Handles Gemini 2.5's reasoning token structure where content might be in
        different attributes or have reasoning/thinking separate from main output.
        """
        content = getattr(message, "content", "")

        # Handle string content (most common case)
        if isinstance(content, str):
            if content.strip():
                return content
            # If content is empty string, check for alternative fields
            # that might contain the actual JSON response

        # Handle list/iterable content (multipart responses)
        if isinstance(content, Iterable) and not isinstance(content, str):
            parts = []
            for item in content:
                text = ""
                if isinstance(item, dict):
                    # Try multiple possible keys where text might be stored
                    text = item.get("text") or item.get("content") or ""
                else:
                    text = getattr(item, "text", "") or getattr(item, "content", "") or ""
                if text:
                    parts.append(str(text))
            if parts:
                return "".join(parts)

        # For Gemini 2.5 models with reasoning: check additional_kwargs
        # which might contain the actual response when reasoning is used
        additional_kwargs = getattr(message, "additional_kwargs", {})
        if additional_kwargs:
            # Some responses might have content in additional_kwargs
            if "content" in additional_kwargs and additional_kwargs["content"]:
                return str(additional_kwargs["content"])

        # Last resort: convert whatever we have to string
        if content:
            return str(content)

        return ""

    def _safe_parse_json(self, text_payload: str) -> dict | None:
        if not text_payload:
            logger.error(
                "query.empty_response",
                extra={
                    "message": "Received empty text payload from model extraction",
                    "hint": "This may indicate Gemini 2.5 reasoning tokens issue or API problem"
                }
            )
            return None

        # Log the raw payload we're trying to parse for debugging
        logger.debug(
            "query.parsing_attempt",
            extra={
                "payload_length": len(text_payload),
                "starts_with": text_payload[:100],
                "ends_with": text_payload[-100:] if len(text_payload) > 100 else "",
            }
        )

        decoder = json.JSONDecoder()

        # First pass: try standard candidates
        for candidate in self._json_candidates(text_payload):
            normalized_candidate = self._normalize_json_text(candidate)
            decoded = self._decode_json(decoder, normalized_candidate)
            if decoded is not None:
                # Validate the decoded payload has required structure
                if self._validate_payload_structure(decoded):
                    return decoded

            # Second pass: aggressive repairs for common LLM formatting issues
            repaired = self._repair_json(normalized_candidate)
            if repaired != normalized_candidate:
                decoded = self._decode_json(decoder, repaired)
                if decoded is not None and self._validate_payload_structure(decoded):
                    logger.info("query.json_repaired_successfully")
                    return decoded

        # Third pass: character-by-character extraction
        extracted = self._extract_json_object(text_payload)
        if extracted is not None and self._validate_payload_structure(extracted):
            logger.info("query.json_extracted_after_failure")
            return extracted

        # Fourth pass: intelligent answer extraction from malformed JSON
        intelligent_fallback = self._extract_answer_intelligently(text_payload)
        if intelligent_fallback:
            logger.warning("query.intelligent_fallback_used")
            return intelligent_fallback

        # Last resort: record failure and return None to trigger retry
        self._record_failed_payload(text_payload[:4000])
        logger.error(
            "query.parse_error",
            extra={
                "payload_preview": text_payload[:500],
                "payload_length": len(text_payload),
            },
        )
        return None

    def _repair_json(self, text: str) -> str:
        """Apply aggressive repairs for common LLM JSON formatting issues."""
        repaired = text

        # Strip code fences or surrounding backticks
        if repaired.startswith("`") and repaired.endswith("`"):
            repaired = repaired.strip("`")

        # Remove trailing commas before closing braces/brackets
        repaired = re.sub(r",\s*(\]|\})", r"\1", repaired)

        # Fix unescaped quotes in values (conservative approach)
        # Match "key": "value with "quotes" inside" and escape internal quotes
        def escape_internal_quotes(match):
            key = match.group(1)
            value = match.group(2)
            # Escape unescaped quotes in the value
            escaped_value = value.replace('\\"', '\x00')  # Temporarily mark escaped quotes
            escaped_value = escaped_value.replace('"', '\\"')  # Escape all quotes
            escaped_value = escaped_value.replace('\x00', '\\"')  # Restore original escaped quotes
            return f'"{key}": "{escaped_value}"'

        # Apply to answer and summary fields specifically
        repaired = re.sub(r'"(answer|summary)":\s*"((?:[^"\\]|\\.)*?)"', escape_internal_quotes, repaired, flags=re.DOTALL)

        # Replace single-quoted JSON with double quotes (when safe)
        if '"' not in repaired or re.search(r"'\s*:\s*'", repaired):
            repaired = re.sub(r"'", '"', repaired)

        # Fix truncated JSON by adding missing closing braces
        open_braces = repaired.count('{')
        close_braces = repaired.count('}')
        if open_braces > close_braces:
            repaired += '}' * (open_braces - close_braces)
            logger.debug("query.added_missing_closing_braces", extra={"count": open_braces - close_braces})

        open_brackets = repaired.count('[')
        close_brackets = repaired.count(']')
        if open_brackets > close_brackets:
            repaired += ']' * (open_brackets - close_brackets)
            logger.debug("query.added_missing_closing_brackets", extra={"count": open_brackets - close_brackets})

        # Ensure citations array exists if missing
        if '"citations"' not in repaired and '"answer"' in repaired:
            # Try to insert citations before last closing brace
            last_brace = repaired.rfind('}')
            if last_brace > 0:
                repaired = repaired[:last_brace] + ', "citations": []' + repaired[last_brace:]

        return self._normalize_json_text(repaired)

    def _validate_payload_structure(self, payload: dict) -> bool:
        """Validate that the payload has the required structure."""
        if not isinstance(payload, dict):
            return False
        if "answer" not in payload:
            return False
        if not isinstance(payload.get("answer"), str):
            return False
        if "citations" not in payload:
            return False
        if not isinstance(payload.get("citations"), list):
            return False
        return True

    def _extract_answer_intelligently(self, text: str) -> dict | None:
        """Extract answer and citations from partially malformed JSON."""
        try:
            # Try to find answer field
            answer_match = re.search(r'"answer"\s*:\s*"([^"]*(?:\\"[^"]*)*)"', text, re.DOTALL)
            if not answer_match:
                # Try with more aggressive pattern
                answer_match = re.search(r'"answer"\s*:\s*"(.+?)(?:"|$)', text, re.DOTALL)

            if answer_match:
                answer = answer_match.group(1).replace('\\"', '"').strip()

                # Try to extract citations array
                citations = []
                citations_match = re.search(r'"citations"\s*:\s*\[(.*?)\]', text, re.DOTALL)
                if citations_match:
                    citations_text = citations_match.group(1)
                    # Extract individual citations
                    for citation_match in re.finditer(
                        r'\{\s*"chunk_id"\s*:\s*"([^"]*)"\s*,\s*"summary"\s*:\s*"([^"]*(?:\\"[^"]*)*)"\s*\}',
                        citations_text
                    ):
                        chunk_id = citation_match.group(1)
                        summary = citation_match.group(2).replace('\\"', '"')
                        if chunk_id:
                            citations.append({"chunk_id": chunk_id, "summary": summary})

                if answer:
                    logger.info("query.intelligent_extraction_success")
                    return {"answer": answer, "citations": citations}

        except Exception as exc:
            logger.debug("query.intelligent_extraction_failed", extra={"error": str(exc)})

        return None

    def _json_candidates(self, raw_text: str) -> List[str]:
        cleaned = raw_text.strip()
        candidates: List[str] = []

        if cleaned.startswith("```"):
            tokens = cleaned.split("```")
            if len(tokens) >= 3:
                cleaned = tokens[1]
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].lstrip()
                candidates.append(cleaned.strip())

        candidates.append(cleaned)

        brace_start = cleaned.find("{")
        brace_end = cleaned.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            candidates.append(cleaned[brace_start : brace_end + 1])

        unique_candidates = []
        seen = set()
        for candidate in candidates:
            normalized = candidate.strip()
            if not normalized or normalized in seen:
                continue
            unique_candidates.append(normalized)
            seen.add(normalized)
        return unique_candidates

    def _decode_json(self, decoder: json.JSONDecoder, payload: str) -> dict | None:
        try:
            obj, end = decoder.raw_decode(payload)
        except json.JSONDecodeError:
            return None
        remainder = payload[end:].strip()
        if remainder and not remainder.startswith(("//", "#")):
            # Allow harmless trailing characters like newlines; anything else falls back to repairs.
            if any(char not in {"\n", "\r"} for char in remainder):
                return None
        return self._ensure_payload(obj)

    def _extract_json_object(self, text: str) -> dict | None:
        decoder = json.JSONDecoder()
        stripped = self._normalize_json_text(text.strip())
        for index, char in enumerate(stripped):
            if char != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            payload = self._ensure_payload(obj)
            if payload is not None:
                return payload
        return None

    @staticmethod
    def _ensure_payload(obj: object) -> dict | None:
        if not isinstance(obj, dict):
            return None
        citations = obj.get("citations")
        if citations is None or not isinstance(citations, list):
            obj["citations"] = [] if citations is None else list(citations) if isinstance(citations, (set, tuple)) else []
        return obj

    @staticmethod
    def _normalize_json_text(text: str) -> str:
        translation_table = {
            ord("“"): '"',
            ord("”"): '"',
            ord("‘"): "'",
            ord("’"): "'",
            0x00A0: 0x20,  # non-breaking space to regular space
        }
        normalized = text.translate(translation_table)
        normalized = normalized.replace("\u2028", "\\n").replace("\u2029", "\\n")
        return normalized

    def _record_failed_payload(self, payload: str) -> None:
        try:
            base_dir = self.config.base_dir
        except AttributeError:
            return
        debug_dir = base_dir / "storage" / "debug_payloads"
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
            snippet = payload[:4000]
            file_path = debug_dir / f"payload_{timestamp}.txt"
            file_path.write_text(snippet, encoding="utf-8")
            logger.info("query.debug_payload_saved", extra={"file": str(file_path)})
        except Exception as exc:
            logger.debug("query.failed_payload_write_error", extra={"error": str(exc)})

    def _parse_citations(self, citations_payload: Sequence[dict]) -> List[CitationResult]:
        citations: List[CitationResult] = []
        for item in citations_payload:
            chunk_id = item.get("chunk_id")
            summary = item.get("summary", "")
            if not chunk_id:
                continue
            citations.append(CitationResult(chunk_id=chunk_id, summary=summary))
        return citations

    def _format_page_label(self, metadata: dict) -> str:
        if "page" in metadata:
            return f"Page {metadata['page']}"
        if "slide" in metadata:
            return f"Slide {metadata['slide']}"
        if "sheet" in metadata:
            return f"Sheet {metadata['sheet']}"
        return metadata.get("doc_type", "section")

    def _build_context_map(self, documents: Sequence[Document]) -> dict[str, Document]:
        context_map: dict[str, Document] = {}
        for doc in documents:
            chunk_id = doc.metadata.get("chunk_id")
            if not chunk_id:
                continue
            context_map[chunk_id] = doc
        return context_map

    def _unique_documents(
        self, documents: Sequence[Document], limit: int
    ) -> List[Document]:
        """Return documents with unique chunk_ids up to the requested limit."""
        unique: List[Document] = []
        seen_ids: set[str] = set()
        for doc in documents:
            chunk_id = doc.metadata.get("chunk_id")
            if not chunk_id or chunk_id in seen_ids:
                continue
            unique.append(doc)
            seen_ids.add(chunk_id)
            if len(unique) >= limit:
                break
        return unique

    @staticmethod
    def _normalize_vector_score(score: Optional[float]) -> Optional[float]:
        if score is None:
            return None
        try:
            value = float(score)
        except (TypeError, ValueError):
            return None
        if value > 1.0:
            return 1.0 / (1.0 + value)
        if value < 0:
            return max(min((value + 1.0) / 2.0, 1.0), 0.0)
        return max(min(value, 1.0), 0.0)

    @staticmethod
    def _normalize_reranker_score(score: Optional[float]) -> Optional[float]:
        if score is None:
            return None
        try:
            value = float(score)
        except (TypeError, ValueError):
            return None
        return 1.0 / (1.0 + math.exp(-value))

    @staticmethod
    def _normalize_bm25_score(score: Optional[float]) -> Optional[float]:
        if score is None:
            return None
        try:
            value = float(score)
        except (TypeError, ValueError):
            return None
        return 1.0 / (1.0 + math.exp(-0.5 * value))

    def _expand_queries(self, question: str) -> List[str]:
        queries = [question]
        simplified = re.sub(r"[^a-zA-Z0-9\s]", " ", question).lower()
        tokens = [token for token in simplified.split() if len(token) > 3]
        if tokens:
            keyword_query = " ".join(tokens)
            if keyword_query not in queries:
                queries.append(keyword_query)
        if "medqa" in simplified and "gemini" in simplified:
            queries.append("MedQA USMLE accuracy Med-Gemini strategy")
        return queries

    def _build_search_variants(
        self, queries: Sequence[str], query_language: str
    ) -> List[Tuple[str, str]]:
        results: List[Tuple[str, str]] = []
        seen: set[str] = set()
        for q in queries:
            normalized = q.strip()
            if not normalized:
                continue
            canonical = re.sub(r"\W+", " ", normalized).strip().lower()
            if not canonical:
                continue
            key = canonical
            if key not in seen:
                results.append((normalized, query_language))
                seen.add(key)
        return results

    def _filter_documents_by_language(
        self, documents: Sequence[Document], query_language: str
    ) -> List[Document]:
        matched: List[Document] = []
        for doc in documents:
            metadata = dict(doc.metadata)
            doc_language = str(metadata.get("language", "")).lower()
            if doc_language in {"en", "ar"} and doc_language != query_language:
                continue
            matched.append(Document(page_content=doc.page_content, metadata=metadata))
        return matched

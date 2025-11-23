from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
import re
import textwrap
import tempfile
from typing import Dict, List
from urllib.parse import urlencode
 
from flask import (
    Blueprint,
    abort,
    current_app,
    jsonify,
    render_template,
    request,
    send_file,
    url_for,
)
 
from app.nlp import detect_language
from app.audio import SpeechToTextService
from app.retrieval.embeddings import resolve_embeddings
from app.retrieval.query_service import QueryService
from app.retrieval.vector_store import VectorStoreManager
from app.retrieval.keyword_store import KeywordIndex
from app.retrieval.reranker import build_reranker
 
logger = logging.getLogger(__name__)
 
web_bp = Blueprint("web", __name__)
 
_query_service: QueryService | None = None
_stt_service: SpeechToTextService | None = None
_SNIPPET_TRIM_WIDTH = 240
_SNIPPET_LEADING_PATTERN = re.compile(r"^(this\s+(?:part|chunk))[:\s\-]*", re.IGNORECASE)
_SUPPORTED_LANGUAGES = {"en", "ar"}
_ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".mp3", ".wav", ".m4a", ".mp4", ".flac"}
_ARABIC_CHAR_PATTERN = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
 
 
def _normalize_language(value: str | None) -> str:
    if not value:
        return "en"
    value = value.lower()
    return value if value in _SUPPORTED_LANGUAGES else "en"
 
 
def _text_direction(language: str) -> str:
    return "rtl" if language == "ar" else "ltr"


def _card_text_direction(language: str) -> str:
    """Return direction for individual cards: LTR for English, RTL for Arabic."""
    return "rtl" if language == "ar" else "ltr"


def _contains_arabic(text: str | None) -> bool:
    if not text:
        return False
    return bool(_ARABIC_CHAR_PATTERN.search(text))


def _ui_strings(language: str) -> Dict[str, str]:
    lang = _normalize_language(language)
    if lang == "ar":
        return {
            "brand": "الذكاء الوثائقي",
            "tagline": "اطرح الأسئلة حول مستنداتك واحصل على إجابات موثوقة مع الاستشهادات.",
            "search_placeholder": "اكتب سؤالك هنا...",
            "search_button": "ابحث",
            "ai_summary": "ملخص الذكاء الاصطناعي",
            "snippet_label": "مقتطف",
            "download_label": "تحميل ملف PDF",
            "download_original": "تحميل النسخة الأصلية",
            "error_generic": "حدث خطأ أثناء معالجة سؤالك.",
            "no_results": "لم يتم العثور على نتائج. حاول تحسين سؤالك.",
            "loading_message": "جاري تحميل المستند...",
            "copy_link": "انسخ الرابط إلى هذه الصفحة",
            "copy_success": "تم نسخ الرابط",
            "copy_failure": "فشل النسخ",
        }
    return {
        "brand": "Document Intelligence",
        "tagline": "Ask questions about your knowledge base and get trusted answers with citations.",
        "search_placeholder": "Ask anything…",
        "search_button": "Search",
        "ai_summary": "AI Summary",
        "snippet_label": "Snippet",
        "download_label": "Download PDF",
        "download_original": "Download original",
        "error_generic": "Something went wrong while processing your question.",
        "no_results": "No results found. Try refining your question.",
        "loading_message": "Loading document.",
        "copy_link": "Copy link to this page",
        "copy_success": "Link copied",
        "copy_failure": "Copy failed",
    }
 
 
def _get_query_service() -> QueryService:
    global _query_service
    if _query_service:
        return _query_service
 
    config = current_app.config["APP_CONFIG"]
    embeddings = resolve_embeddings(
        config.embedding_model_name,
        cache_dir=config.embedding_cache_dir,
        model_path=config.embedding_model_path,
    )
    vector_store = VectorStoreManager(
        embeddings=embeddings,
        storage_path=config.vector_store_path,
        backend_preference=config.vector_store_backend,
    )
    vector_store.load()
    keyword_index = None
    if config.enable_hybrid_retrieval and config.bm25_index_path:
        keyword_index = KeywordIndex(config.bm25_index_path)
        keyword_index.load()
    reranker = build_reranker(config)
    _query_service = QueryService(
        config,
        vector_store,
        keyword_index=keyword_index,
        reranker=reranker,
    )
    return _query_service


def _get_stt_service() -> SpeechToTextService | None:
    global _stt_service
    if _stt_service is not None:
        return _stt_service

    config = current_app.config["APP_CONFIG"]
    try:
        _stt_service = SpeechToTextService.from_config(config)
    except ImportError as exc:  # pragma: no cover - runtime availability
        logger.warning("stt.unavailable", extra={"error": str(exc)})
        _stt_service = None
    except Exception as exc:  # pragma: no cover - unexpected failure
        logger.exception("stt.init_failed", extra={"error": str(exc)})
        _stt_service = None
    return _stt_service
 
 
@web_bp.route("/", methods=["GET"])
def index():
    config = current_app.config["APP_CONFIG"]
    query = request.args.get("q", "").strip()
    answer = ""
    cards: List[Dict] = []
    error_message = ""
    requested_language = request.args.get("lang")
    query_language = _normalize_language(requested_language)
    query_has_arabic = _contains_arabic(query)

    if requested_language is None:
        if query_has_arabic:
            query_language = "ar"
        if query:
            detected = detect_language(query)
            if detected:
                normalized_detected = _normalize_language(detected)
                if normalized_detected == "ar":
                    query_language = "ar"
                elif query_language != "ar":
                    query_language = normalized_detected
    if query:
        try:
            service = _get_query_service()
            result = service.query(query)
            result_language = getattr(result, "language", None)
            if result_language:
                normalized_result_language = _normalize_language(result_language)
                if normalized_result_language == "ar":
                    query_language = "ar"
                elif query_language != "ar":
                    query_language = normalized_result_language
            answer = result.answer
            ui_for_cards = _ui_strings(query_language)
            cards = _build_cards(result.context_map, result.citations, query_language, ui_for_cards)
        except Exception as exc:  # pragma: no cover - user feedback
            logger.exception("query.failed", extra={"query": query})
            error_message = _ui_strings(query_language)["error_generic"]

    ui_strings = _ui_strings(query_language)
    text_direction = _text_direction(query_language)

    return render_template(
        "index.html",
        query=query,
        answer=answer,
        cards=cards,
        error_message=error_message,
        ui=ui_strings,
        text_direction=text_direction,
        no_results_message=ui_strings["no_results"],
        query_language=query_language,
    )


@web_bp.route("/transcribe", methods=["POST"])
def transcribe_audio():
    service = _get_stt_service()
    if service is None:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Speech-to-text is not configured on the server.",
                }
            ),
            503,
        )

    if "audio" not in request.files:
        return jsonify({"success": False, "error": "No audio file provided."}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"success": False, "error": "No audio file selected."}), 400

    suffix = Path(audio_file.filename).suffix or ".webm"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            audio_file.save(tmp)
            tmp_path = Path(tmp.name)

        result = service.transcribe(tmp_path)
        if not result.text:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Unable to detect speech in the recording.",
                    }
                ),
                422,
            )

        payload = {
            "success": True,
            "text": result.text,
            "language": result.language,
            "duration": result.duration,
        }
        return jsonify(payload), 200
    except Exception as exc:  # pragma: no cover - runtime failure path
        logger.exception("stt.transcribe_failed", extra={"error": str(exc)})
        return (
            jsonify({"success": False, "error": "Speech transcription failed."}),
            500,
        )
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
 
 
@web_bp.route("/upload", methods=["POST"])
def upload_document():
    """
    Handle document or audio upload with automatic ingestion.
    Returns JSON: {success: bool, filename: str, chunks: int} or {success: bool, error: str}
    """
    config = current_app.config["APP_CONFIG"]

    # Validate file presence
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    # Validate file type
    extension = Path(file.filename).suffix.lower()
    if extension not in _ALLOWED_UPLOAD_EXTENSIONS:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Invalid file type. Upload PDF or audio files (MP3, WAV, M4A, MP4, FLAC).",
                }
            ),
            400,
        )

    # Sanitize filename (prevent path traversal)
    original_filename = Path(file.filename).name
    safe_filename = re.sub(r'[^\w\s\-\.]', '_', original_filename)

    # Check file size (50MB limit)
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning

    MAX_SIZE = 50 * 1024 * 1024  # 50MB
    if file_size > MAX_SIZE:
        return (
            jsonify(
                {"success": False, "error": "File too large. Maximum size is 50MB."}
            ),
            400,
        )

    try:
        # Save file to uploads directory
        uploads_dir = config.uploads_dir
        uploads_dir.mkdir(parents=True, exist_ok=True)

        file_path = uploads_dir / safe_filename

        # Handle duplicate filenames
        counter = 1
        while file_path.exists():
            stem = Path(safe_filename).stem
            suffix = Path(safe_filename).suffix
            file_path = uploads_dir / f"{stem}_{counter}{suffix}"
            counter += 1

        file.save(str(file_path))

        logger.info(
            "upload.saved",
            extra={"uploaded_file": safe_filename, "size": file_size, "extension": extension},
        )

        # Import here to avoid circular dependencies
        from app.ingestion.pipeline import IngestionPipeline

        # Run ingestion on the uploaded file
        pipeline = IngestionPipeline(config)
        summary = pipeline.ingest(target_files=[file_path])

        logger.info(
            "upload.ingested",
            extra={
                "uploaded_file": safe_filename,
                "chunks": summary.chunks_written,
                "sections": summary.sections_extracted,
            },
        )

        return jsonify({
            "success": True,
            "filename": safe_filename,
            "chunks": summary.chunks_written,
            "sections": summary.sections_extracted,
        }), 200

    except Exception as exc:
        logger.exception("upload.failed", extra={"uploaded_file": safe_filename})
        return jsonify({"success": False, "error": "Upload failed. Please try again."}), 500


@web_bp.route("/documents/<path:doc_path>", methods=["GET"])
def download_document(doc_path: str):
    full_path = _resolve_document_path(doc_path)
    # Force correct content type for PDFs to encourage inline rendering.
    if full_path.suffix.lower() == ".pdf":
        mimetype = "application/pdf"
    else:
        mimetype, _ = mimetypes.guess_type(full_path.name)
    response = send_file(
        full_path,
        mimetype=mimetype or "application/octet-stream",
        as_attachment=False,  # inline rendering
        download_name=full_path.name,
        conditional=True,
    )
    response.headers["Cache-Control"] = "public, max-age=86400"
    return response
 
 
@web_bp.route("/viewer/<path:doc_path>", methods=["GET"])
def view_document(doc_path: str):
    full_path = _resolve_document_path(doc_path)
    extension = full_path.suffix.lower()
    page = request.args.get("page")
    zoom = request.args.get("zoom")
    search_term = request.args.get("search")
    language = _normalize_language(request.args.get("lang"))
    download_url = url_for("web.download_document", doc_path=doc_path, lang=language)
 
    if extension != ".pdf":
        return download_document(doc_path)
 
    viewer_base = url_for("static", filename="lib/pdfjs/web/viewer.html")
    query_params = {"file": download_url}
    if zoom:
        query_params["zoom"] = zoom
    if search_term:
        query_params["search"] = search_term
    viewer_src = f"{viewer_base}?{urlencode(query_params)}"
    if page:
        viewer_src = f"{viewer_src}#page={page}"
    return render_template(
        "document_viewer.html",
        doc_name=full_path.name,
        doc_path=doc_path,
        viewer_src=viewer_src,
        initial_page=page,
        initial_zoom=zoom,
        initial_search=search_term,
        download_url=download_url,
        ui=_ui_strings(language),
        text_direction=_text_direction(language),
    )
 
 
def _resolve_document_path(doc_path: str) -> Path:
    config = current_app.config["APP_CONFIG"]
    safe_path = Path(doc_path)
    uploads_dir = config.uploads_dir.resolve()
    full_path = (config.base_dir / safe_path).resolve()
 
    try:
        full_path.relative_to(uploads_dir)
    except ValueError:
        abort(404)
    if not full_path.exists():
        abort(404)
 
    return full_path
 
 
def _build_cards(context_map, citations, query_language: str, ui_strings: Dict[str, str]):
    cards = []
    seen_chunks = set()
    for citation in citations:
        doc = context_map.get(citation.chunk_id)
        if not doc:
            continue
        if citation.chunk_id in seen_chunks:
            continue
        seen_chunks.add(citation.chunk_id)
        metadata = doc.metadata
        page_label = _format_page_label(metadata)
        relative_path = metadata.get("relative_path", metadata.get("source"))
        download_link = url_for("web.download_document", doc_path=relative_path, lang=query_language)
        action_link = _build_action_link(relative_path, metadata, query_language)
        snippet = _make_card_snippet(doc.page_content)
        summary = citation.summary.strip()
        if summary.lower().startswith("this chunk"):
            summary = "This part" + summary[10:]
        # Truncate overly long summaries to keep UI clean (max 200 chars)
        if len(summary) > 200:
            summary = summary[:197] + "..."
        language = metadata.get("language", "unknown")
        original_type = metadata.get("original_doc_type")
        doc_type = original_type or metadata.get("doc_type", "document")
        policy_name = metadata.get("policy_name")
        section_label = metadata.get("section_label") or metadata.get("section_heading")
        effective_date = metadata.get("effective_date")
        # Determine card direction based on card's language (standard: en=ltr, ar=rtl)
        normalized_card_lang = _normalize_language(language)
        card_direction = _card_text_direction(normalized_card_lang)
        cards.append(
            {
                "filename": metadata.get("source"),
                "title": summary,
                "title_link": action_link,
                "page_label": page_label,
                "doc_type": doc_type.upper() if isinstance(doc_type, str) else doc_type,
                "language": language.upper() if isinstance(language, str) else language,
                "snippet": snippet,
                "snippet_label": ui_strings["snippet_label"],
                "policy_name": policy_name,
                "section_label": section_label,
                "effective_date": effective_date,
                "download_link": download_link,
                "card_direction": card_direction,
            }
        )
    return cards
 
 
def _make_card_snippet(text: str) -> str:
    if not text:
        return ""
    
    # Detect Arabic presence early
    has_arabic = bool(re.search(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]", text))
    
    # Basic cleaning - preserve Arabic structure
    cleaned = text.replace("\u00a0", " ").replace("\n", " ")
    
    # Remove zero-width and problematic Unicode characters
    cleaned = re.sub(r"[\u200B-\u200D\uFEFF\u00AD]", "", cleaned)  # Zero-width, soft hyphen
    
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    
    # Remove leading pattern
    cleaned = _SNIPPET_LEADING_PATTERN.sub("", cleaned).strip()
    
    if not has_arabic:
        # For non-Arabic text, apply aggressive dash removal
        cleaned = re.sub(r"\s+-\s+", " ", cleaned)
        cleaned = re.sub(r"^\s*-\s+", "", cleaned)
        cleaned = re.sub(r"\s+-\s*$", "", cleaned)
        cleaned = re.sub(r"\s*-\s*-\s*", " ", cleaned)
    else:
        # For Arabic text, be more conservative
        # Only remove clearly problematic dashes (ASCII hyphens with spaces)
        # Don't touch Arabic punctuation or Tatweel (U+0640)
        cleaned = re.sub(r"\s+[-\u002D]\s+", " ", cleaned)  # Only ASCII hyphen
        # Remove multiple consecutive dashes
        cleaned = re.sub(r"\s*[-\u002D]\s*[-\u002D]\s*", " ", cleaned)
        # Remove leading/trailing ASCII dashes
        cleaned = re.sub(r"^\s*[-\u002D]\s+", "", cleaned)
        cleaned = re.sub(r"\s+[-\u002D]\s*$", "", cleaned)
    
    # Final whitespace cleanup
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    
    if not cleaned:
        return ""
    
    try:
        snippet = textwrap.shorten(cleaned, width=_SNIPPET_TRIM_WIDTH, placeholder="…")
    except ValueError:
        snippet = cleaned
    
    return snippet
 
 
def _format_page_label(metadata: dict) -> str:
    if "page" in metadata:
        return f"Page {metadata['page']}"
    if "slide" in metadata:
        return f"Slide {metadata['slide']}"
    if "sheet" in metadata:
        return f"Sheet {metadata['sheet']}"
    return metadata.get("doc_type", "Section").title()
 
 
def _build_action_link(relative_path: str, metadata: dict, query_language: str) -> str:
    viewer_path = metadata.get("viewer_relative_path") or relative_path
    viewer_kwargs = {"doc_path": viewer_path, "lang": query_language}
    page = metadata.get("viewer_page") or metadata.get("page")
    if page:
        viewer_kwargs["page"] = page
    return url_for("web.view_document", **viewer_kwargs)
 
 

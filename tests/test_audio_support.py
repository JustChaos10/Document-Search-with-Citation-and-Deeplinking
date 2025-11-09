from app.ingestion.extractors import DocumentExtractor


def test_document_extractor_supports_audio_extensions():
    audio_exts = {".mp3", ".wav", ".m4a", ".mp4", ".flac"}
    assert audio_exts.issubset(DocumentExtractor.SUPPORTED_EXTENSIONS)

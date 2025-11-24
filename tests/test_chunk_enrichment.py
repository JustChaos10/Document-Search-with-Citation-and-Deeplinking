from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from app.ingestion.pipeline import IngestionPipeline


class DummyConfig(SimpleNamespace):
    base_dir: Path


def test_enrich_chunks_adds_metadata():
    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    pipeline.config = DummyConfig(base_dir=Path.cwd())

    chunk = Document(
        page_content=(
            "Emergency Response Overview\n"
            "This section covers the procedures for immediate medical readiness and safety compliance across teams."
        ),
        metadata={
            "source": "plan.pdf",
            "chunk": 0,
            "page": 3,
            "language": "en",
            "path": str(Path("uploads/plan.pdf").resolve()),
        },
    )

    enriched = pipeline._enrich_chunks([chunk])[0]
    meta = enriched.metadata

    assert meta["chunk_id"] == "plan.pdf:0"
    assert meta["section_heading"].startswith("Emergency Response")
    assert meta["page_label"] == "Page 3"
    assert meta["word_count"] > 0
    assert meta["preview"]

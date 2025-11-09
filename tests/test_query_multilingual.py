from langchain_core.documents import Document

from app.retrieval.query_service import QueryService


def test_build_search_variants_returns_unique_queries():
    qs = QueryService.__new__(QueryService)
    queries = ["Clinical note", "clinical note", "CLINICAL! note"]
    variants = qs._build_search_variants(queries, "en")
    assert variants == [("Clinical note", "en")]


def test_filter_documents_by_language_respects_query_language():
    qs = QueryService.__new__(QueryService)
    english_doc = Document(page_content="data", metadata={"language": "en", "chunk_id": "doc:1"})
    arabic_doc = Document(page_content="data", metadata={"language": "ar", "chunk_id": "doc:2"})
    unknown_doc = Document(page_content="data", metadata={"chunk_id": "doc:3"})

    filtered = qs._filter_documents_by_language(
        [english_doc, arabic_doc, unknown_doc],
        query_language="en",
    )

    assert english_doc in filtered
    assert unknown_doc in filtered
    assert arabic_doc not in filtered

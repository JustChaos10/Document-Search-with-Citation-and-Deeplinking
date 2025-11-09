from langchain_core.documents import Document

from app.retrieval import keyword_store


def test_keyword_index_add_and_search(tmp_path):
    index_path = tmp_path / "bm25.json"
    index = keyword_store.KeywordIndex(index_path)
    index.load()

    doc_medical = Document(
        page_content="Safety protocol for the medical laboratory and emergency procedures.",
        metadata={"chunk_id": "lab:1", "language": "en"},
    )
    doc_finance = Document(
        page_content="Quarterly financial report covering revenue expansion and margins.",
        metadata={"chunk_id": "fin:1", "language": "en"},
    )

    index.add_documents([doc_medical, doc_finance])

    results = index.search("medical safety protocol", k=1)

    if not keyword_store.HAS_BM25:
        assert results == []
    else:
        assert results
        top = results[0]
        assert top.metadata["chunk_id"] == "lab:1"
        assert "bm25_score" in top.metadata

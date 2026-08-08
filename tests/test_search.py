from app.search_engine import SemanticSearch


def test_semantic_search_initializes():
    search = SemanticSearch()

    assert search is not None
    assert search.vector_store is not None


def test_semantic_search_returns_results():
    search = SemanticSearch()

    results = search.search(
        query="machine learning model",
        top_k=3
    )

    assert isinstance(results, list)
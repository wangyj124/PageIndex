import asyncio

from pageindex.vector_retrieve import build_vector_chunks, retrieve_vector_candidates


def test_build_vector_chunks_preserves_page_and_section_metadata(monkeypatch):
    monkeypatch.setattr("pageindex.vector_retrieve._safe_token_count", lambda text: len(text))
    documents = [{"page": 2, "section_path": "付款 > 预付款", "content": "abcdef\nuvwxyz"}]

    chunks = build_vector_chunks(documents, chunk_tokens=9, overlap_tokens=2)

    assert len(chunks) == 2
    assert chunks[0]["page"] == 2
    assert chunks[0]["section_path"] == "付款 > 预付款"
    assert chunks[1]["text"].startswith("ef")
    assert all(len(chunk["text"]) <= 9 for chunk in chunks)


def test_vector_retrieval_persists_index_and_aggregates_pages(tmp_path):
    calls = []

    def embedding_fn(texts, model):
        calls.append(list(texts))
        vectors = []
        for text in texts:
            vectors.append([1.0, 0.0] if "目标金额" in text else [0.0, 1.0])
        return vectors

    kwargs = {
        "doc_id": "doc-demo",
        "source_sha256": "sha",
        "page_documents": [
            {"page": 1, "section_path": "", "content": "无关内容"},
            {"page": 3, "section_path": "价格", "content": "目标金额为500万元"},
        ],
        "query_text": "目标金额",
        "processed_pages": set(),
        "workspace": tmp_path,
        "embedding_model": "openai/test-embedding",
        "embedding_fn": embedding_fn,
    }

    first = asyncio.run(retrieve_vector_candidates(**kwargs))
    second = asyncio.run(retrieve_vector_candidates(**kwargs))

    assert first[0]["page"] == 3
    assert second[0]["page"] == 3
    assert len(calls) == 3  # first call indexes + queries; second call queries from cached index
    assert list((tmp_path / "_retrieval" / "vector").glob("*.json"))


def test_vector_retrieval_obeys_page_context_budget(monkeypatch):
    monkeypatch.setattr("pageindex.vector_retrieve._safe_token_count", lambda text: len(text))

    def embedding_fn(texts, model):
        return [[1.0, 0.0] for _ in texts]

    result = asyncio.run(
        retrieve_vector_candidates(
            doc_id="doc-demo",
            source_sha256="sha",
            page_documents=[
                {"page": 1, "section_path": "", "content": "aa"},
                {"page": 2, "section_path": "", "content": "bbbb"},
                {"page": 3, "section_path": "", "content": "bb"},
            ],
            query_text="目标",
            processed_pages=set(),
            workspace=None,
            embedding_model="openai/test-embedding",
            embedding_fn=embedding_fn,
            page_limit=3,
            context_token_budget=4,
        )
    )

    assert [candidate["page"] for candidate in result] == [1, 3]


def test_vector_retrieval_skips_highest_scored_page_larger_than_budget(monkeypatch):
    monkeypatch.setattr("pageindex.vector_retrieve._safe_token_count", lambda text: len(text))

    def embedding_fn(texts, model):
        vectors = []
        for text in texts:
            vectors.append([1.0, 0.0] if "top" in text or "目标" in text else [0.8, 0.2])
        return vectors

    result = asyncio.run(
        retrieve_vector_candidates(
            doc_id="doc-demo",
            source_sha256="sha",
            page_documents=[
                {"page": 1, "section_path": "", "content": "toplong"},
                {"page": 2, "section_path": "", "content": "ok"},
            ],
            query_text="目标",
            processed_pages=set(),
            workspace=None,
            embedding_model="openai/test-embedding",
            embedding_fn=embedding_fn,
            page_limit=1,
            context_token_budget=4,
        )
    )

    assert [candidate["page"] for candidate in result] == [2]

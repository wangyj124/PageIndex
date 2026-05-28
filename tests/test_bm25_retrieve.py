from pageindex.bm25_retrieve import build_page_documents, retrieve_bm25_candidates, tokenize


def test_tokenize_emits_chinese_bigrams_and_ascii_terms():
    tokens = tokenize("预付款 Ratio-30")

    assert "预付" in tokens
    assert "付款" in tokens
    assert "ratio-30" in tokens


def test_bm25_rrf_prioritizes_primary_query_and_attaches_section_path():
    pages = [
        {"page": 1, "content": "首付款在合同签订后支付。"},
        {"page": 2, "content": "预付款比例为合同总额30%。"},
    ]
    structure = [
        {"title": "支付条款", "start_page": 1, "end_page": 2, "nodes": [{"title": "预付款", "start_page": 2, "end_page": 2}]}
    ]
    documents = build_page_documents(pages, structure)

    candidates = retrieve_bm25_candidates(
        documents,
        ["预付款比例"],
        ["首付款"],
        primary_weight=3.0,
        expanded_weight=1.0,
    )

    assert candidates[0]["page"] == 2
    assert "bm25_primary" in candidates[0]["sources"]
    assert candidates[0]["section_path"] == "支付条款 > 预付款"
    assert candidates[1]["page"] == 1
    assert candidates[0]["bm25_score"] > candidates[1]["bm25_score"]

import asyncio
import json
from types import SimpleNamespace

import pytest

from pageindex.contract_extraction import (
    FieldSpec,
    _build_extraction_prompt,
    _build_locator_prompt,
    _build_long_context_extraction_prompt,
    _build_query_generation_prompt,
    _build_structure_digest,
    _expand_oversized_candidates,
    _generate_query_spec,
    _load_query_cache,
    _normalize_field_result,
    _normalize_tree_locations,
    _query_cache_key,
    _select_extraction_batch,
    _select_initial_candidates,
    _split_extraction_content,
    extract_contract_fields,
    normalize_schema,
)


class StubClient:
    retrieve_model = "openai/test-model"

    def __init__(self):
        self.page_requests = []

    def get_document_structure(self, doc_id):
        return json.dumps(
            [
                {
                    "title": "价格条款",
                    "summary": "本章节描述合同总价和付款安排",
                    "start_page": 4,
                    "end_page": 5,
                    "nodes": [],
                }
            ],
            ensure_ascii=False,
        )

    def get_page_content(self, doc_id, pages):
        self.page_requests.append(pages)
        return json.dumps(
            [
                {
                    "page": 4,
                    "content": "合同总价为人民币壹佰万元整。",
                }
            ],
            ensure_ascii=False,
        )


def test_normalize_schema_accepts_fields_wrapper():
    fields = normalize_schema(
        {
            "fields": [
                {
                    "name": "contract_amount",
                    "description": "合同总金额",
                    "type": "string",
                    "required": True,
                }
            ]
        }
    )
    assert fields[0].name == "contract_amount"
    assert fields[0].required is True


def test_normalize_schema_fills_instruction_from_description_config():
    fields = normalize_schema([{"name": "advance_payment", "description": "10%预付款"}])

    assert "预付款的金额、百分比、付款条件" in fields[0].instruction


def test_normalize_schema_keeps_explicit_instruction_over_config():
    fields = normalize_schema(
        [{"name": "advance_payment", "description": "10%预付款", "instruction": "只提取显式约定"}]
    )

    assert fields[0].instruction == "只提取显式约定"


def test_normalize_schema_matches_description_config_after_whitespace_normalization():
    fields = normalize_schema([{"name": "advance_payment", "description": "  出国\n预付款  "}])

    assert fields[0].instruction == "请提取合同中关于出国预付款金额、比例及支付节点等相关条款。"


def test_model_prompts_do_not_expose_field_name_identifier():
    field = FieldSpec(
        name="field_001_advance_payment_mismatch",
        description="合同签订日期",
        type="string",
        required=True,
        instruction="按中文说明提取",
    )
    prompts = [
        _build_locator_prompt(field, "[]"),
        _build_extraction_prompt(field, "[]"),
        _build_long_context_extraction_prompt(field, "[]"),
        _build_query_generation_prompt(field, 3, 8),
    ]

    assert all("field_001_advance_payment_mismatch" not in prompt for prompt in prompts)
    assert all("合同签订日期" in prompt for prompt in prompts)


def test_extraction_prompts_require_full_clause_value_and_short_quote():
    field = FieldSpec(
        name="field_001",
        description="合同金额",
        type="string",
        required=True,
    )
    prompts = [
        _build_extraction_prompt(field, "[]"),
        _build_long_context_extraction_prompt(field, "[]"),
    ]

    assert all("完整合同条款原文" in prompt for prompt in prompts)
    assert all("不要总结、改写或只返回字段值" in prompt for prompt in prompts)
    assert all("省略号" in prompt for prompt in prompts)
    assert all("不要把完整条款全文放入 evidence" in prompt for prompt in prompts)


def test_tree_locator_prompt_uses_locations_with_page_limit():
    field = FieldSpec(name="field_001", description="付款比例", type="string")
    prompt = _build_locator_prompt(field, "[]", location_limit=3, pages_per_location=3)

    assert "最多返回 3 个候选位置" in prompt
    assert "每个候选位置最多返回 3 页" in prompt
    assert '"locations"' in prompt


def test_tree_location_limit_setting_takes_precedence_over_legacy_tree_page_limit(monkeypatch):
    prompts = []

    async def fake_llm_acompletion(model, prompt):
        prompts.append(prompt)
        if "生成页级检索规格" in prompt:
            return '{"primary_queries":["无命中"],"expanded_keywords":[]}'
        if "定位最相关的页面" in prompt:
            return '{"locations":[]}'
        return '{"status":"not_found","value":"","evidence":"","pages":[],"confidence":"Low","reason":"none"}'

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)
    client = EnhancedStubClient(
        pages=[{"page": 1, "content": "无关"}],
        structure=[{"title": "正文", "start_page": 1, "end_page": 1, "nodes": []}],
        tree_location_limit=2,
        tree_page_limit=9,
        vector_enabled=False,
    )

    extract_contract_fields(
        client,
        "doc-demo",
        [{"name": "target", "description": "目标字段"}],
        include_retrieval_metadata=True,
    )

    locator_prompt = next(prompt for prompt in prompts if "定位最相关的页面" in prompt)
    assert "最多返回 2 个候选位置" in locator_prompt


def test_normalize_tree_locations_limits_locations_pages_and_keeps_legacy_pages():
    payload = {
        "locations": [
            {"title": "第一处", "pages": [1, 2, 3, 4], "reason": "直接相关"},
            {"title": "第二处", "pages": [3, 5, 99], "reason": "补充位置"},
            {"title": "第三处", "pages": [6], "reason": "超出位置限制"},
        ]
    }

    locations = _normalize_tree_locations(payload, valid_pages={1, 2, 3, 4, 5, 6}, location_limit=2, pages_per_location=3)
    legacy_locations = _normalize_tree_locations({"pages": [4, 5, 6, 7]}, location_limit=2, pages_per_location=3)

    assert locations == [
        {"location_id": "tree_location_1", "pages": [1, 2, 3], "reason": "直接相关", "title": "第一处"},
        {"location_id": "tree_location_2", "pages": [3, 5], "reason": "补充位置", "title": "第二处"},
    ]
    assert [location["pages"] for location in legacy_locations] == [[4], [5]]


def test_structure_digest_includes_prefix_summary_and_content_page_range():
    digest = _build_structure_digest(
        [
            {
                "title": "付款",
                "start_page": 1,
                "end_page": 10,
                "content_start_page": 1,
                "content_end_page": 2,
                "prefix_summary": "付款前言",
                "nodes": [],
            }
        ]
    )

    assert '"summary": "付款前言"' in digest
    assert '"start_page": 1' in digest
    assert '"end_page": 10' in digest
    assert '"content_start_page": 1' in digest
    assert '"content_end_page": 2' in digest


def test_extract_contract_fields_retries_invalid_confidence(monkeypatch):
    responses = iter(
        [
            '{"pages":[4],"reason":"金额通常在价格条款页"}',
            '{"status":"found","value":"100万元","evidence":"合同总价为人民币壹佰万元整。","pages":[4],"confidence":"Certain","reason":null}',
            '{"status":"found","value":"100万元","evidence":"合同总价为人民币壹佰万元整。","pages":[4],"confidence":"High","reason":null}',
        ]
    )

    async def fake_llm_acompletion(model, prompt):
        return next(responses)

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)

    client = StubClient()
    result = extract_contract_fields(
        client,
        "doc-1",
        [
            {
                "name": "contract_amount",
                "description": "合同总金额",
                "type": "string",
            }
        ],
        max_concurrency=1,
        retries=1,
    )

    assert result["contract_amount"]["status"] == "found"
    assert result["contract_amount"]["confidence"] == "High"
    assert result["contract_amount"]["pages"] == [4]
    assert client.page_requests == ["4"]


def test_extract_contract_fields_not_found_forces_low_confidence(monkeypatch):
    responses = iter(
        [
            '{"pages":[4],"reason":"签署信息可能在尾页"}',
            '{"status":"not_found","value":"2024-01-01","evidence":"not used","pages":[4],"confidence":"High","reason":"未在提供页中找到明确签订日期"}',
        ]
    )

    async def fake_llm_acompletion(model, prompt):
        return next(responses)

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)

    client = StubClient()
    result = extract_contract_fields(
        client,
        "doc-2",
        [
            {
                "name": "signing_date",
                "description": "合同签订日期",
                "type": "date",
            }
        ],
        max_concurrency=1,
    )

    assert result["signing_date"] == {
        "status": "not_found",
        "value": "",
        "evidence": "",
        "pages": [4],
        "confidence": "Low",
        "reason": "未在提供页中找到明确签订日期",
    }


class EnhancedStubClient:
    retrieve_model = "openai/test-model"

    def __init__(self, pages, structure, workspace=None, **settings):
        self.pages = pages
        self.structure = structure
        self.workspace = workspace
        self.retrieval_config = SimpleNamespace(vector_enabled=settings.pop("vector_enabled", False), **settings)

    def get_document_structure(self, doc_id):
        return json.dumps(self.structure, ensure_ascii=False)

    def get_retrieval_payload(self, doc_id):
        return {
            "type": "pdf",
            "source_sha256": "source-sha",
            "pages": self.pages,
            "structure": self.structure,
        }


class LongContextStubClient:
    retrieve_model = "openai/test-model"
    long_context_model = "openai/long-context-model"

    def __init__(self, pages):
        self.pages = pages

    def get_long_context_payload(self, doc_id):
        return {"doc_id": doc_id, "pages": self.pages}

    def get_document_structure(self, doc_id):
        raise AssertionError("long-context extraction must not load document structure")

    def get_retrieval_payload(self, doc_id):
        raise AssertionError("long-context extraction must not enter retrieval")


def test_long_context_extraction_reads_entire_page_artifact_per_field_and_reports_progress(monkeypatch):
    prompts = []
    models = []
    progress = []

    async def fake_llm_acompletion(model, prompt):
        models.append(model)
        prompts.append(prompt)
        if "- description: 甲方" in prompt:
            return '{"status":"found","value":"甲公司","evidence":"甲方：甲公司","pages":[1],"confidence":"High","reason":null}'
        return '{"status":"found","value":"100万元","evidence":"金额：100万元","pages":[2],"confidence":"High","reason":null}'

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)
    client = LongContextStubClient(
        [{"page": 2, "content": "金额：100万元"}, {"page": 1, "content": "甲方：甲公司"}]
    )

    result = extract_contract_fields(
        client,
        "doc-full",
        {
            "fields": [
                {"name": "party_a", "description": "甲方"},
                {"name": "amount", "description": "金额"},
            ]
        },
        long_context_mode=True,
        include_retrieval_metadata=True,
        progress_callback=lambda current, total: progress.append((current, total)),
    )

    assert len(prompts) == 2
    assert models == ["openai/long-context-model", "openai/long-context-model"]
    assert all("party_a" not in prompt and "amount" not in prompt for prompt in prompts)
    assert all("完整文档分页原文" in prompt and "甲方：甲公司" in prompt and "金额：100万元" in prompt for prompt in prompts)
    assert all(prompt.index('"page": 1') < prompt.index('"page": 2') for prompt in prompts)
    assert result["party_a"]["resolution_method"] == "long_context"
    assert result["amount"]["retrieval_sources"] == ["full_document"]
    assert progress == [(1, 2), (2, 2)]


def test_long_context_extraction_requires_dedicated_page_artifact():
    client = LongContextStubClient([])

    with pytest.raises(ValueError, match="重新上传并完成建树"):
        extract_contract_fields(
            client,
            "doc-legacy",
            {"fields": [{"name": "amount", "description": "合同金额"}]},
            long_context_mode=True,
        )


def test_long_context_extraction_retries_result_with_page_outside_full_document(monkeypatch):
    responses = iter(
        [
            '{"status":"found","value":"500万元","evidence":"伪造证据","pages":[99],"confidence":"High","reason":null}',
            '{"status":"found","value":"500万元","evidence":"合同金额为500万元","pages":[2],"confidence":"High","reason":null}',
        ]
    )

    async def fake_llm_acompletion(model, prompt):
        return next(responses)

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)
    result = extract_contract_fields(
        LongContextStubClient([{"page": 2, "content": "合同金额为500万元"}]),
        "doc-full",
        {"fields": [{"name": "amount", "description": "合同金额"}]},
        long_context_mode=True,
        retries=1,
    )

    assert result["amount"]["status"] == "found"
    assert result["amount"]["pages"] == [2]


def test_enhanced_extraction_uses_bm25_when_tree_locator_misses_and_caches_queries(monkeypatch, tmp_path):
    state = {"query_generation_calls": 0}
    events = []

    class CaptureLogger:
        def info(self, payload):
            events.append(payload)

    async def fake_llm_acompletion(model, prompt):
        if "生成页级检索规格" in prompt:
            state["query_generation_calls"] += 1
            return '{"primary_queries":["预付款比例"],"expanded_keywords":["首付款"]}'
        if "定位最相关的页面" in prompt:
            return '{"pages":[99],"reason":"summary selected a stale page"}'
        assert '"page": 2' in prompt
        return '{"status":"found","value":"30%","evidence":"预付款比例为合同总额30%。","pages":[2],"confidence":"High","reason":null}'

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)
    client = EnhancedStubClient(
        pages=[{"page": 1, "content": "目录"}, {"page": 2, "content": "预付款比例为合同总额30%。"}],
        structure=[{"title": "支付条款", "summary": "", "start_page": 1, "end_page": 2, "nodes": []}],
        workspace=tmp_path,
    )
    schema = [{"name": "advance_payment", "description": "预付款比例", "type": "string"}]

    first = extract_contract_fields(
        client,
        "doc-demo",
        schema,
        include_retrieval_metadata=True,
        retrieval_logger=CaptureLogger(),
    )
    second = extract_contract_fields(
        client,
        "doc-demo",
        schema,
        include_retrieval_metadata=True,
        retrieval_logger=CaptureLogger(),
    )

    assert first["advance_payment"]["value"] == "30%"
    assert first["advance_payment"]["resolution_method"] == "bm25"
    assert "bm25_primary" in first["advance_payment"]["retrieval_sources"]
    assert second["advance_payment"]["value"] == "30%"
    assert state["query_generation_calls"] == 1
    assert (tmp_path / "_retrieval" / "query_specs.json").is_file()
    query_events = [event for event in events if event["event"] == "retrieval_query_spec_resolved"]
    assert [event["generation_source"] for event in query_events] == ["llm_generated", "cache_hit"]
    assert query_events[0]["primary_queries"] == ["预付款比例"]
    assert query_events[0]["expanded_keywords"] == ["首付款"]
    candidate_event = next(event for event in events if event["event"] == "retrieval_candidates_selected")
    page_two = next(candidate for candidate in candidate_event["bm25_candidates"] if candidate["page"] == 2)
    assert "bm25_primary" in page_two["sources"]
    assert page_two["bm25_score"] > 0
    assert page_two["bm25_ranks"]["bm25_primary"] == 1


def test_query_cache_disabled_does_not_read_or_reuse_cached_spec(monkeypatch, tmp_path):
    cache_dir = tmp_path / "_retrieval"
    cache_dir.mkdir()
    (cache_dir / "query_specs.json").write_text('{"old": {"primary_queries": ["旧查询"]}}', encoding="utf-8")
    client = EnhancedStubClient(
        pages=[],
        structure=[],
        workspace=tmp_path,
        retrieval_query_cache_enabled=False,
    )
    field = normalize_schema([{"name": "amount", "description": "合同总价"}])[0]
    called = {"count": 0}

    async def fake_llm_acompletion(model, prompt):
        called["count"] += 1
        return '{"primary_queries":["新查询"],"expanded_keywords":[]}'

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)
    stale_memory_cache = {_query_cache_key(client, field): {"primary_queries": ["内存旧查询"]}}

    assert _load_query_cache(client) == {}
    result = asyncio.run(_generate_query_spec(client, field, stale_memory_cache, retries=0, timeout_seconds=1))

    assert result["primary_queries"] == ["新查询"]
    assert result["generation_source"] == "llm_generated"
    assert called["count"] == 1


def test_query_cache_hit_is_normalized_to_current_limits_and_key_includes_limits(monkeypatch):
    field = normalize_schema([{"name": "amount", "description": "合同总价"}])[0]
    client = EnhancedStubClient(
        pages=[],
        structure=[],
        retrieval_primary_query_limit=1,
        retrieval_expanded_keyword_limit=1,
    )

    async def fail_if_called(model, prompt):
        raise AssertionError("valid cache hit should not regenerate queries")

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fail_if_called)
    cache = {
        _query_cache_key(client, field): {
            "primary_queries": ["总价", "含税总价", "合同金额"],
            "expanded_keywords": ["价款", "金额", "费用"],
        }
    }
    result = asyncio.run(_generate_query_spec(client, field, cache, retries=0, timeout_seconds=1))
    less_restricted_client = EnhancedStubClient(
        pages=[],
        structure=[],
        retrieval_primary_query_limit=2,
        retrieval_expanded_keyword_limit=2,
    )

    assert result == {
        "primary_queries": ["总价"],
        "expanded_keywords": ["价款"],
        "generation_source": "cache_hit",
    }
    assert _query_cache_key(client, field) != _query_cache_key(less_restricted_client, field)


def test_enhanced_extraction_preserves_locator_page_priority(monkeypatch):
    loaded_pages = []

    async def fake_llm_acompletion(model, prompt):
        if "生成页级检索规格" in prompt:
            return '{"primary_queries":["无命中"],"expanded_keywords":[]}'
        if "定位最相关的页面" in prompt:
            return '{"pages":[3,1,2],"reason":"ranked pages"}'
        if "目标字段为目标值。" in prompt:
            loaded_pages.append(3)
            return '{"status":"found","value":"目标值","evidence":"目标字段为目标值。","pages":[3],"confidence":"High","reason":null}'
        loaded_pages.append(1 if "第一页" in prompt else 2)
        return '{"status":"not_found","value":"","evidence":"","pages":[],"confidence":"Low","reason":"wrong order"}'

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)
    client = EnhancedStubClient(
        pages=[
            {"page": 1, "content": "第一页"},
            {"page": 2, "content": "第二页"},
            {"page": 3, "content": "目标字段为目标值。"},
        ],
        structure=[{"title": "正文", "start_page": 1, "end_page": 3, "nodes": []}],
        bm25_selected_page_limit=0,
        extract_batch_page_limit=1,
        vector_enabled=False,
    )

    result = extract_contract_fields(
        client,
        "doc-demo",
        [{"name": "target", "description": "目标字段"}],
        include_retrieval_metadata=True,
    )["target"]

    assert result["value"] == "目标值"
    assert loaded_pages == [3, 1, 2]


def test_enhanced_extraction_deferred_tree_found_survives_later_error(monkeypatch):
    loaded_pages = []
    events = []

    class CaptureLogger:
        def info(self, payload):
            events.append(payload)

    async def fake_llm_acompletion(model, prompt):
        if "生成页级检索规格" in prompt:
            raise AssertionError("BM25 query generation should not run after a deferred tree result")
        if "定位最相关的页面" in prompt:
            return (
                '{"locations":['
                '{"title":"第一处","pages":[1],"reason":"direct"},'
                '{"title":"第二处","pages":[2],"reason":"verify"}'
                ']}'
            )
        if "扫描异常" in prompt:
            loaded_pages.append(2)
            return '{"status":"error","value":"","evidence":"","pages":[2],"confidence":"Low","reason":"OCR unreadable"}'
        if "目标字段为目标值" in prompt:
            loaded_pages.append(1)
            return '{"status":"found","value":"目标值","evidence":"目标字段为目标值。","pages":[1],"confidence":"High","reason":null}'
        raise AssertionError(prompt)

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)
    client = EnhancedStubClient(
        pages=[
            {"page": 1, "content": "目标字段为目标值。"},
            {"page": 2, "content": "扫描异常"},
        ],
        structure=[{"title": "正文", "start_page": 1, "end_page": 2, "nodes": []}],
        bm25_selected_page_limit=0,
        extract_batch_page_limit=1,
        vector_enabled=False,
    )

    result = extract_contract_fields(
        client,
        "doc-demo",
        [{"name": "target", "description": "目标字段"}],
        include_retrieval_metadata=True,
        retrieval_logger=CaptureLogger(),
    )["target"]

    assert result["status"] == "found"
    assert result["value"] == "目标值"
    assert result["pages"] == [1]
    assert loaded_pages == [1, 2]
    found_events = [event for event in events if event["event"] == "field_extraction_found_candidate"]
    assert found_events == [
        {
            "event": "field_extraction_found_candidate",
            "field": "target",
            "attempt_index": 1,
            "loaded_pages": [1],
            "content_units": ["1"],
            "evidence_pages": [1],
            "resolution_method": "tree",
            "retrieval_sources": ["tree"],
            "deferred_due_to_pending_tree_work": True,
            "pending_tree_pages": [2],
            "confidence": "High",
        }
    ]
    assert "evidence_preview" not in found_events[0]
    completed_event = next(event for event in events if event["event"] == "field_extraction_completed")
    assert completed_event["status"] == "found"
    assert completed_event["evidence_pages"] == [1]


def test_found_candidate_log_can_include_truncated_evidence_preview(monkeypatch):
    events = []

    class CaptureLogger:
        def info(self, payload):
            events.append(payload)

    long_evidence = "证据" * 60

    async def fake_llm_acompletion(model, prompt):
        if "生成页级检索规格" in prompt:
            raise AssertionError("BM25 query generation should not run for tree hit")
        if "定位最相关的页面" in prompt:
            return '{"locations":[{"title":"正文","pages":[1],"reason":"direct"}]}'
        return json.dumps(
            {
                "status": "found",
                "value": "目标值",
                "evidence": long_evidence,
                "pages": [1],
                "confidence": "High",
                "reason": None,
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)
    client = EnhancedStubClient(
        pages=[{"page": 1, "content": "目标字段为目标值。"}],
        structure=[{"title": "正文", "start_page": 1, "end_page": 1, "nodes": []}],
        bm25_selected_page_limit=0,
        vector_enabled=False,
        retrieval_log_evidence_preview_enabled=True,
    )

    extract_contract_fields(
        client,
        "doc-demo",
        [{"name": "target", "description": "目标字段"}],
        retrieval_logger=CaptureLogger(),
    )

    found_event = next(event for event in events if event["event"] == "field_extraction_found_candidate")
    assert found_event["evidence_preview"] == long_evidence[:80]
    assert len(found_event["evidence_preview"]) == 80
    assert "value" not in found_event
    assert "evidence" not in found_event


def test_enhanced_extraction_uses_tree_locations_and_skips_initial_bm25(monkeypatch):
    extraction_prompts = []
    calls = []

    async def fake_llm_acompletion(model, prompt):
        calls.append(prompt)
        if "生成页级检索规格" in prompt:
            raise AssertionError("BM25 query generation should be skipped when tree has candidates")
        if "定位最相关的页面" in prompt:
            return (
                '{"locations":['
                '{"title":"第一处","pages":[1,2,3,4],"reason":"direct"},'
                '{"title":"第二处","pages":[3,5,6],"reason":"nearby"},'
                '{"title":"第三处","pages":[7],"reason":"extra"}'
                ']}'
            )
        extraction_prompts.append(prompt)
        return '{"status":"found","value":"目标值","evidence":"第6页目标值","pages":[6],"confidence":"High","reason":null}'

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)
    client = EnhancedStubClient(
        pages=[
            {"page": 1, "content": "tree 1"},
            {"page": 2, "content": "tree 2"},
            {"page": 3, "content": "tree 3"},
            {"page": 4, "content": "truncated"},
            {"page": 5, "content": "tree 5"},
            {"page": 6, "content": "第6页目标值"},
            {"page": 7, "content": "extra"},
        ],
        structure=[{"title": "正文", "start_page": 1, "end_page": 7, "nodes": []}],
        tree_page_limit=2,
        tree_location_page_limit=3,
        extract_batch_page_limit=6,
        vector_enabled=False,
    )

    result = extract_contract_fields(
        client,
        "doc-demo",
        [{"name": "target", "description": "目标字段"}],
        include_retrieval_metadata=True,
    )["target"]

    assert result["status"] == "found"
    assert result["resolution_method"] == "tree"
    assert len(extraction_prompts) == 1
    prompt = extraction_prompts[0]
    assert all(f'"page": {page}' in prompt for page in [1, 2, 3, 5, 6])
    assert '"page": 4' not in prompt
    assert '"page": 7' not in prompt
    assert not any("生成页级检索规格" in prompt for prompt in calls)


def test_enhanced_extraction_falls_back_to_bm25_after_tree_is_exhausted_and_dedupes(monkeypatch):
    loaded_pages = []

    async def fake_llm_acompletion(model, prompt):
        if "定位最相关的页面" in prompt:
            return '{"locations":[{"title":"付款","pages":[1],"reason":"tree hit"}]}'
        if "生成页级检索规格" in prompt:
            return '{"primary_queries":["付款比例"],"expanded_keywords":[]}'
        if '"current_loaded_pages": [' in prompt:
            if '"page": 1' in prompt and '"page": 2' not in prompt:
                loaded_pages.append(1)
                return '{"status":"not_found","value":"","evidence":"","pages":[1],"confidence":"Low","reason":"tree page no value"}'
            loaded_pages.append(2)
            return '{"status":"found","value":"30%","evidence":"付款比例为30%。","pages":[2],"confidence":"High","reason":null}'
        raise AssertionError(prompt)

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)
    client = EnhancedStubClient(
        pages=[
            {"page": 1, "content": "付款比例标题但没有具体值。"},
            {"page": 2, "content": "付款比例为30%。"},
        ],
        structure=[{"title": "付款", "start_page": 1, "end_page": 2, "nodes": []}],
        extract_batch_page_limit=1,
        vector_enabled=False,
    )

    result = extract_contract_fields(
        client,
        "doc-demo",
        [{"name": "payment_ratio", "description": "付款比例"}],
        include_retrieval_metadata=True,
    )["payment_ratio"]

    assert result["status"] == "found"
    assert result["pages"] == [2]
    assert loaded_pages == [1, 2]


def test_bm25_fallback_filters_tree_pages_before_selected_limit(monkeypatch):
    loaded_pages = []

    async def fake_llm_acompletion(model, prompt):
        if "定位最相关的页面" in prompt:
            return '{"locations":[{"title":"付款","pages":[1],"reason":"tree hit"}]}'
        if "生成页级检索规格" in prompt:
            return '{"primary_queries":["付款比例"],"expanded_keywords":[]}'
        if '"current_loaded_pages": [' in prompt:
            if '"page": 1' in prompt and '"page": 2' not in prompt:
                loaded_pages.append(1)
                return '{"status":"not_found","value":"","evidence":"","pages":[1],"confidence":"Low","reason":"tree page no value"}'
            loaded_pages.append(2)
            return '{"status":"found","value":"30%","evidence":"付款比例为30%。","pages":[2],"confidence":"High","reason":null}'
        raise AssertionError(prompt)

    def fake_retrieve_bm25_candidates(*args, **kwargs):
        return [
            {"page": 1, "sources": ["bm25_primary"], "bm25_score": 2.0, "read_status": "pending"},
            {"page": 2, "sources": ["bm25_primary"], "bm25_score": 1.0, "read_status": "pending"},
        ]

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)
    monkeypatch.setattr("pageindex.contract_extraction.retrieve_bm25_candidates", fake_retrieve_bm25_candidates)
    client = EnhancedStubClient(
        pages=[
            {"page": 1, "content": "付款比例标题但没有具体值。"},
            {"page": 2, "content": "付款比例为30%。"},
        ],
        structure=[{"title": "付款", "start_page": 1, "end_page": 2, "nodes": []}],
        bm25_selected_page_limit=1,
        extract_batch_page_limit=1,
        vector_enabled=False,
    )

    result = extract_contract_fields(
        client,
        "doc-demo",
        [{"name": "payment_ratio", "description": "付款比例"}],
        include_retrieval_metadata=True,
    )["payment_ratio"]

    assert result["status"] == "found"
    assert result["pages"] == [2]
    assert loaded_pages == [1, 2]


def test_vector_fallback_excludes_existing_candidate_pool_pages(monkeypatch):
    observed_processed_pages = []

    async def fake_llm_acompletion(model, prompt):
        if "定位最相关的页面" in prompt:
            return '{"locations":[]}'
        if "生成页级检索规格" in prompt:
            return '{"primary_queries":["付款比例"],"expanded_keywords":[]}'
        if '"current_loaded_pages": [' in prompt:
            return '{"status":"not_found","value":"","evidence":"","pages":[1],"confidence":"Low","reason":"bm25 no value"}'
        raise AssertionError(prompt)

    def fake_retrieve_bm25_candidates(*args, **kwargs):
        return [{"page": 1, "sources": ["bm25_primary"], "bm25_score": 1.0, "read_status": "pending"}]

    async def fake_retrieve_vector_candidates(**kwargs):
        observed_processed_pages.append(set(kwargs["processed_pages"]))
        return [{"page": 2, "sources": ["vector_fallback"], "read_status": "pending"}]

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)
    monkeypatch.setattr("pageindex.contract_extraction.retrieve_bm25_candidates", fake_retrieve_bm25_candidates)
    monkeypatch.setattr("pageindex.contract_extraction.retrieve_vector_candidates", fake_retrieve_vector_candidates)
    client = EnhancedStubClient(
        pages=[
            {"page": 1, "content": "付款比例标题但没有具体值。"},
            {"page": 2, "content": "付款比例为30%。"},
        ],
        structure=[{"title": "付款", "start_page": 1, "end_page": 2, "nodes": []}],
        bm25_selected_page_limit=1,
        extract_batch_page_limit=1,
        vector_enabled=True,
        vector_fallback_max_runs=1,
    )

    result = extract_contract_fields(
        client,
        "doc-demo",
        [{"name": "payment_ratio", "description": "付款比例"}],
        include_retrieval_metadata=True,
    )["payment_ratio"]

    assert observed_processed_pages == [{1}]
    assert result["status"] == "not_found"


def test_initial_candidate_pool_preserves_order_without_dropping_pages_for_batch_budget():
    client = EnhancedStubClient(
        pages=[],
        structure=[],
        initial_candidate_page_limit=3,
        initial_context_token_budget=4,
    )
    documents = {1: {"content": "aa"}, 2: {"content": "bbbb"}, 3: {"content": "bb"}}
    bm25_candidates = [
        {"page": 2, "sources": ["bm25_primary"], "bm25_score": 1.0, "read_status": "pending"},
        {"page": 3, "sources": ["bm25_primary"], "bm25_score": 0.5, "read_status": "pending"},
    ]

    candidates = _select_initial_candidates(client, [1], bm25_candidates, documents)

    assert [candidate["page"] for candidate in candidates] == [1, 2, 3]


def test_extraction_batch_defers_page_that_does_not_fit_remaining_budget(monkeypatch):
    monkeypatch.setattr("pageindex.contract_extraction._count_context_tokens", lambda client, text: len(text))
    client = EnhancedStubClient(
        pages=[],
        structure=[],
        extract_batch_page_limit=4,
        initial_context_token_budget=4,
    )
    documents = {1: {"content": "aa"}, 2: {"content": "bbbb"}, 3: {"content": "bb"}}
    candidates = [
        {"page": 1, "read_status": "pending"},
        {"page": 2, "read_status": "pending"},
        {"page": 3, "read_status": "pending"},
    ]

    first_batch, _ = _select_extraction_batch(client, candidates, documents, [])
    first_batch[0]["read_status"] = "loaded"
    second_batch, _ = _select_extraction_batch(
        client,
        [candidate for candidate in candidates if candidate["read_status"] == "pending"],
        documents,
        [],
    )

    assert [candidate["page"] for candidate in first_batch] == [1]
    assert [candidate["page"] for candidate in second_batch] == [2]
    assert candidates[2]["read_status"] == "pending"


def test_extraction_batch_processes_oversized_page_as_standalone_chunks(monkeypatch):
    monkeypatch.setattr("pageindex.contract_extraction._count_context_tokens", lambda client, text: len(text))
    client = EnhancedStubClient(
        pages=[],
        structure=[],
        extract_batch_page_limit=4,
        initial_context_token_budget=4,
    )
    documents = {1: {"content": "123456789"}, 2: {"content": "ok"}}
    candidates = [{"page": 1, "read_status": "pending"}, {"page": 2, "read_status": "pending"}]

    candidates, expanded_pages = _expand_oversized_candidates(client, candidates, documents)
    batch, _ = _select_extraction_batch(client, candidates, documents, [])

    assert expanded_pages == [{"page": 1, "content_units": ["1-1", "1-2", "1-3"]}]
    assert [candidate["content_unit"] for candidate in batch] == ["1-1"]
    assert all(len(candidate["content_override"]) <= 4 for candidate in candidates if candidate.get("is_chunk"))
    assert batch[0]["standalone_batch"] is True


def test_extraction_chunks_overlap_text_at_regular_budget_boundaries(monkeypatch):
    monkeypatch.setattr("pageindex.contract_extraction._count_context_tokens", lambda client, text: len(text))
    client = EnhancedStubClient(pages=[], structure=[], initial_context_token_budget=12)

    chunks = _split_extraction_content(client, "abcdefghijVALUE", 12)

    assert chunks == ["abcdefghijVA", "VALUE"]


def test_enhanced_extraction_reads_oversized_candidate_in_chunks(monkeypatch):
    monkeypatch.setattr("pageindex.contract_extraction._count_context_tokens", lambda client, text: len(text))
    events = []

    class CaptureLogger:
        def info(self, payload):
            events.append(payload)

    async def fake_llm_acompletion(model, prompt):
        if "生成页级检索规格" in prompt:
            return '{"primary_queries":["总价"],"expanded_keywords":[]}'
        if "定位最相关的页面" in prompt:
            return '{"pages":[1],"reason":"price page"}'
        if '"content": "金额"' in prompt:
            return '{"status":"found","value":"金额","evidence":"金额","pages":[1],"confidence":"High","reason":null}'
        if '"content": "xxxx"' in prompt:
            return '{"status":"not_found","value":"","evidence":"","pages":[1],"confidence":"Low","reason":"first chunk"}'
        raise AssertionError(prompt)

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)
    client = EnhancedStubClient(
        pages=[{"page": 1, "content": "xxxx金额"}],
        structure=[{"title": "价格", "start_page": 1, "end_page": 1, "nodes": []}],
        initial_context_token_budget=4,
        bm25_selected_page_limit=0,
        vector_enabled=False,
    )

    result = extract_contract_fields(
        client,
        "doc-demo",
        [{"name": "amount", "description": "总价"}],
        include_retrieval_metadata=True,
        retrieval_logger=CaptureLogger(),
    )["amount"]

    assert result["status"] == "found"
    assert result["value"] == "金额"
    assert any(event["event"] == "candidate_pages_chunked_for_extraction" for event in events)


def test_manifest_retains_evaluated_irrelevant_state_for_following_batch(monkeypatch):
    monkeypatch.setattr("pageindex.contract_extraction._count_context_tokens", lambda client, text: len(text))

    async def fake_llm_acompletion(model, prompt):
        if "生成页级检索规格" in prompt:
            return '{"primary_queries":["付款"],"expanded_keywords":[]}'
        if "定位最相关的页面" in prompt:
            return '{"pages":[1,2],"reason":"candidates"}'
        if '"content": "无关"' in prompt:
            return '{"status":"not_found","value":"","evidence":"","pages":[1],"confidence":"Low","reason":"irrelevant"}'
        assert '"read_status": "evaluated_irrelevant"' in prompt
        return '{"status":"found","value":"30%","evidence":"付款30%","pages":[2],"confidence":"High","reason":null}'

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)
    client = EnhancedStubClient(
        pages=[{"page": 1, "content": "无关"}, {"page": 2, "content": "付款30%"}],
        structure=[{"title": "付款", "start_page": 1, "end_page": 2, "nodes": []}],
        extract_batch_page_limit=1,
        bm25_selected_page_limit=0,
        vector_enabled=False,
    )

    result = extract_contract_fields(
        client,
        "doc-demo",
        [{"name": "payment", "description": "付款"}],
        include_retrieval_metadata=True,
    )["payment"]

    assert result["status"] == "found"
    assert result["pages"] == [2]


def test_reference_to_oversized_page_reuses_chunk_candidates_without_requeueing_full_page(monkeypatch):
    monkeypatch.setattr("pageindex.contract_extraction._count_context_tokens", lambda client, text: len(text))
    loaded_units = []
    events = []

    class CaptureLogger:
        def info(self, payload):
            events.append(payload)

    async def fake_llm_acompletion(model, prompt):
        if "生成页级检索规格" in prompt:
            return '{"primary_queries":["付款"],"expanded_keywords":[]}'
        if "定位最相关的页面" in prompt:
            return '{"pages":[1],"reason":"reference"}'
        if "定位可能的目标物理页" in prompt:
            return '{"pages":[2],"reason":"attachment"}'
        if '"content": "30%"' in prompt:
            loaded_units.append("2-2")
            return '{"status":"found","value":"30%","evidence":"30%","pages":[2],"confidence":"High","reason":null}'
        if '"content": "xxxx"' in prompt:
            loaded_units.append("2-1")
            return '{"status":"not_found","value":"","evidence":"","pages":[2],"confidence":"Low","reason":"first chunk"}'
        return (
            '{"status":"reference_found","value":"","evidence":"见附件甲。","pages":[1],'
            '"confidence":"High","reason":"reference","references":['
            '{"reference_type":"attachment","target_text":"附件甲","relation":"detail"}]}'
        )

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)
    client = EnhancedStubClient(
        pages=[{"page": 1, "content": "见附件"}, {"page": 2, "content": "xxxx30%"}],
        structure=[{"title": "正文", "start_page": 1, "end_page": 2, "nodes": []}],
        initial_context_token_budget=4,
        bm25_selected_page_limit=0,
        extract_batch_page_limit=2,
        vector_enabled=False,
    )

    result = extract_contract_fields(
        client,
        "doc-demo",
        [{"name": "payment", "description": "付款"}],
        include_retrieval_metadata=True,
        retrieval_logger=CaptureLogger(),
    )["payment"]

    assert result["status"] == "found"
    assert result["pages"] == [2]
    assert loaded_units == ["2-1", "2-2"]
    chunk_events = [event for event in events if event["event"] == "candidate_pages_chunked_for_extraction"]
    assert chunk_events == [{"event": "candidate_pages_chunked_for_extraction", "field": "payment", "pages": [{"page": 2, "content_units": ["2-1", "2-2"]}]}]


def test_reference_requeues_previously_evaluated_long_page_as_existing_chunks(monkeypatch):
    monkeypatch.setattr("pageindex.contract_extraction._count_context_tokens", lambda client, text: len(text))
    loaded_units = []

    async def fake_llm_acompletion(model, prompt):
        if "生成页级检索规格" in prompt:
            return '{"primary_queries":["无命中"],"expanded_keywords":[]}'
        if "定位最相关的页面" in prompt:
            return '{"pages":[1,2],"reason":"long page then reference"}'
        if "定位可能的目标物理页" in prompt:
            return '{"pages":[1],"reason":"referenced long page"}'
        if '"content": "见甲"' in prompt:
            loaded_units.append("2")
            return (
                '{"status":"reference_found","value":"","evidence":"见附件甲。","pages":[2],'
                '"confidence":"High","reason":"reference","references":['
                '{"reference_type":"attachment","target_text":"附件甲","relation":"detail"}]}'
            )
        if '"content": "valu"' in prompt and '"read_status": "evaluated_reference"' in prompt:
            assert '"prior_evaluations"' in prompt
            assert '"evaluated_irrelevant"' in prompt
            loaded_units.append("1-2-repeat")
            return '{"status":"found","value":"valu","evidence":"valu","pages":[1],"confidence":"High","reason":null}'
        if '"content": "aaaa"' in prompt:
            loaded_units.append("1-1")
        elif '"content": "valu"' in prompt:
            loaded_units.append("1-2")
        elif '"content": "e"' in prompt:
            loaded_units.append("1-3")
        return '{"status":"not_found","value":"","evidence":"","pages":[1],"confidence":"Low","reason":"not yet"}'

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)
    client = EnhancedStubClient(
        pages=[{"page": 1, "content": "aaaavalue"}, {"page": 2, "content": "见甲"}],
        structure=[{"title": "正文", "start_page": 1, "end_page": 2, "nodes": []}],
        initial_context_token_budget=4,
        bm25_selected_page_limit=0,
        extract_batch_page_limit=1,
        vector_enabled=False,
    )

    result = extract_contract_fields(
        client,
        "doc-demo",
        [{"name": "payment", "description": "付款"}],
        include_retrieval_metadata=True,
    )["payment"]

    assert result["status"] == "found"
    assert result["pages"] == [1]
    assert loaded_units == ["1-1", "1-2", "1-3", "2", "1-1", "1-2-repeat", "1-3"]
    assert result["resolution_method"] == "reference_followup"


def test_enhanced_extraction_follows_reference_and_retains_direct_evidence(monkeypatch):
    async def fake_llm_acompletion(model, prompt):
        if "生成页级检索规格" in prompt:
            return '{"primary_queries":["预付款"],"expanded_keywords":[]}'
        if "定位最相关的页面" in prompt:
            return '{"pages":[1],"reason":"summary match"}'
        if "定位可能的目标物理页" in prompt:
            return '{"pages":[2],"reason":"target clause"}'
        if '"page": 2' in prompt:
            return '{"status":"found","value":"30%","evidence":"支付合同总额30%的预付款。","pages":[2],"confidence":"High","reason":null}'
        return (
            '{"status":"reference_found","value":"","evidence":"付款方式详见专用合同条款第12.3条。",'
            '"pages":[1],"confidence":"High","reason":"reference only","references":['
            '{"reference_type":"clause","target_text":"专用合同条款第12.3条","relation":"detail"}]}'
        )

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)
    client = EnhancedStubClient(
        pages=[
            {"page": 1, "content": "付款方式详见专用合同条款第12.3条。"},
            {"page": 2, "content": "支付合同总额30%的预付款。"},
        ],
        structure=[
            {"title": "付款方式", "start_page": 1, "end_page": 1, "nodes": []},
            {"title": "专用合同条款第12.3条", "start_page": 2, "end_page": 2, "nodes": []},
        ],
        bm25_selected_page_limit=0,
        vector_enabled=False,
    )

    result = extract_contract_fields(
        client,
        "doc-demo",
        [{"name": "advance_payment", "description": "预付款"}],
        include_retrieval_metadata=True,
    )["advance_payment"]

    assert result["status"] == "found"
    assert result["pages"] == [2]
    assert result["resolution_method"] == "reference_followup"
    assert [item["role"] for item in result["evidence_chain"]] == ["reference_source", "value_source"]


def test_reference_llm_fallback_ignores_pages_outside_structure(monkeypatch):
    extraction_prompts = []

    async def fake_llm_acompletion(model, prompt):
        if "生成页级检索规格" in prompt:
            return '{"primary_queries":["付款比例"],"expanded_keywords":[]}'
        if "定位最相关的页面" in prompt:
            return '{"pages":[1],"reason":"reference source"}'
        if "定位可能的目标物理页" in prompt:
            return '{"pages":[3,2],"reason":"possible target"}'
        extraction_prompts.append(prompt)
        if '"page": 2' in prompt:
            return '{"status":"found","value":"30%","evidence":"付款比例为30%。","pages":[2],"confidence":"High","reason":null}'
        return (
            '{"status":"reference_found","value":"","evidence":"比例详见附件甲。","pages":[1],'
            '"confidence":"High","reason":"reference only","references":['
            '{"reference_type":"attachment","target_text":"附件甲","relation":"detail"}]}'
        )

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)
    client = EnhancedStubClient(
        pages=[
            {"page": 1, "content": "比例详见附件甲。"},
            {"page": 2, "content": "付款比例为30%。"},
            {"page": 3, "content": "页缓存中存在但未纳入结构树。"},
        ],
        structure=[{"title": "正文", "start_page": 1, "end_page": 2, "nodes": []}],
        bm25_selected_page_limit=0,
        vector_enabled=False,
    )

    result = extract_contract_fields(
        client,
        "doc-demo",
        [{"name": "payment_ratio", "description": "付款比例"}],
        include_retrieval_metadata=True,
    )["payment_ratio"]

    assert result["status"] == "found"
    assert result["pages"] == [2]
    assert not any('"page": 3' in prompt for prompt in extraction_prompts)


def test_reference_target_in_candidate_pool_is_promoted_to_next_batch(monkeypatch):
    loaded_batches = []

    async def fake_llm_acompletion(model, prompt):
        if "生成页级检索规格" in prompt:
            return '{"primary_queries":["预付款"],"expanded_keywords":["附件甲"]}'
        if "定位最相关的页面" in prompt:
            return '{"pages":[1],"reason":"reference source"}'
        if "定位可能的目标物理页" in prompt:
            return '{"pages":[2],"reason":"attachment target"}'
        if '"current_loaded_pages": [' in prompt:
            if "付款比例为30%。" in prompt:
                loaded_batches.append(2)
                return '{"status":"found","value":"30%","evidence":"付款比例为30%。","pages":[2],"confidence":"High","reason":null}'
            if "预付款定义说明。" in prompt:
                loaded_batches.append(3)
                return '{"status":"not_found","value":"","evidence":"","pages":[3],"confidence":"Low","reason":"not target"}'
            loaded_batches.append(1)
            return (
                '{"status":"reference_found","value":"","evidence":"比例详见附件甲。","pages":[1],'
                '"confidence":"High","reason":"reference only","references":['
                '{"reference_type":"attachment","target_text":"附件甲","relation":"detail"}]}'
            )
        raise AssertionError(prompt)

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)
    client = EnhancedStubClient(
        pages=[
            {"page": 1, "content": "比例详见附件甲。"},
            {"page": 2, "content": "付款比例为30%。"},
            {"page": 3, "content": "预付款定义说明。"},
        ],
        structure=[
            {"title": "正文", "start_page": 1, "end_page": 1, "nodes": []},
            {"title": "附件甲", "start_page": 2, "end_page": 2, "nodes": []},
            {"title": "其他预付款说明", "start_page": 3, "end_page": 3, "nodes": []},
        ],
        extract_batch_page_limit=1,
        vector_enabled=False,
    )

    result = extract_contract_fields(
        client,
        "doc-demo",
        [{"name": "payment_ratio", "description": "预付款比例"}],
        include_retrieval_metadata=True,
    )["payment_ratio"]

    assert result["status"] == "found"
    assert loaded_batches == [1, 2]


def test_reference_resolution_uses_llm_to_select_nested_child_clause(monkeypatch):
    async def fake_llm_acompletion(model, prompt):
        if "生成页级检索规格" in prompt:
            return '{"primary_queries":["付款比例"],"expanded_keywords":[]}'
        if "定位最相关的页面" in prompt:
            return '{"pages":[1],"reason":"reference page"}'
        if "定位可能的目标物理页" in prompt:
            assert "专用合同条款第12.3条" in prompt
            return '{"pages":[35],"reason":"child clause is exact target"}'
        if "付款比例为30%。" in prompt:
            return '{"status":"found","value":"30%","evidence":"付款比例为30%。","pages":[35],"confidence":"High","reason":null}'
        return (
            '{"status":"reference_found","value":"","evidence":"付款详见专用合同条款第12.3条。","pages":[1],'
            '"confidence":"High","reason":"reference","references":['
            '{"reference_type":"clause","target_text":"专用合同条款第12.3条","relation":"detail"}]}'
        )

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)
    client = EnhancedStubClient(
        pages=[
            {"page": 1, "content": "付款详见专用合同条款第12.3条。"},
            {"page": 10, "content": "专用合同条款总则。"},
            {"page": 35, "content": "付款比例为30%。"},
        ],
        structure=[
            {"title": "引用", "start_page": 1, "end_page": 1, "nodes": []},
            {
                "title": "专用合同条款",
                "start_page": 10,
                "end_page": 35,
                "nodes": [{"title": "第12.3条 付款方式", "start_page": 35, "end_page": 35, "nodes": []}],
            },
        ],
        bm25_selected_page_limit=0,
        vector_enabled=False,
    )

    result = extract_contract_fields(
        client,
        "doc-demo",
        [{"name": "payment_ratio", "description": "付款比例"}],
        include_retrieval_metadata=True,
    )["payment_ratio"]

    assert result["pages"] == [35]
    assert result["resolution_method"] == "reference_followup"


def test_reference_resolution_enqueues_multiple_targets_from_same_source(monkeypatch):
    loaded_target_pages = []

    async def fake_llm_acompletion(model, prompt):
        if "生成页级检索规格" in prompt:
            return '{"primary_queries":["付款比例"],"expanded_keywords":[]}'
        if "定位最相关的页面" in prompt:
            return '{"pages":[1],"reason":"reference page"}'
        if "定位可能的目标物理页" in prompt:
            return '{"pages":[2]}' if "引用目标：附件甲" in prompt else '{"pages":[3]}'
        if "附件乙规定付款比例为30%。" in prompt:
            loaded_target_pages.append(3)
            return '{"status":"found","value":"30%","evidence":"附件乙规定付款比例为30%。","pages":[3],"confidence":"High","reason":null}'
        if "附件甲未规定付款比例。" in prompt:
            loaded_target_pages.append(2)
            return '{"status":"not_found","value":"","evidence":"","pages":[2],"confidence":"Low","reason":"not in A"}'
        return (
            '{"status":"reference_found","value":"","evidence":"付款分别见附件甲和附件乙。","pages":[1],'
            '"confidence":"High","reason":"references","references":['
            '{"reference_type":"attachment","target_text":"附件甲","relation":"detail"},'
            '{"reference_type":"attachment","target_text":"附件乙","relation":"detail"}]}'
        )

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)
    client = EnhancedStubClient(
        pages=[
            {"page": 1, "content": "付款分别见附件甲和附件乙。"},
            {"page": 2, "content": "附件甲未规定付款比例。"},
            {"page": 3, "content": "附件乙规定付款比例为30%。"},
        ],
        structure=[
            {"title": "正文", "start_page": 1, "end_page": 1, "nodes": []},
            {"title": "附件甲", "start_page": 2, "end_page": 2, "nodes": []},
            {"title": "附件乙", "start_page": 3, "end_page": 3, "nodes": []},
        ],
        bm25_selected_page_limit=0,
        extract_batch_page_limit=1,
        vector_enabled=False,
    )

    result = extract_contract_fields(
        client,
        "doc-demo",
        [{"name": "payment_ratio", "description": "付款比例"}],
        include_retrieval_metadata=True,
    )["payment_ratio"]

    assert result["pages"] == [3]
    assert loaded_target_pages == [2, 3]
    assert result["resolution_method"] == "reference_followup"


def test_failed_reference_branch_does_not_pollute_direct_candidate_result(monkeypatch):
    async def fake_llm_acompletion(model, prompt):
        if "生成页级检索规格" in prompt:
            return '{"primary_queries":["无命中"],"expanded_keywords":[]}'
        if "定位最相关的页面" in prompt:
            return '{"pages":[1,3],"reason":"two candidates"}'
        if "定位可能的目标物理页" in prompt:
            return '{"pages":[2],"reason":"attachment"}'
        if "付款比例为30%。" in prompt:
            return '{"status":"found","value":"30%","evidence":"付款比例为30%。","pages":[3],"confidence":"High","reason":null}'
        if "附件甲无付款比例。" in prompt:
            return '{"status":"not_found","value":"","evidence":"","pages":[2],"confidence":"Low","reason":"not in attachment"}'
        return (
            '{"status":"reference_found","value":"","evidence":"付款见附件甲。","pages":[1],'
            '"confidence":"High","reason":"reference","references":['
            '{"reference_type":"attachment","target_text":"附件甲","relation":"detail"}]}'
        )

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)
    client = EnhancedStubClient(
        pages=[
            {"page": 1, "content": "付款见附件甲。"},
            {"page": 2, "content": "附件甲无付款比例。"},
            {"page": 3, "content": "付款比例为30%。"},
        ],
        structure=[{"title": "正文", "start_page": 1, "end_page": 3, "nodes": []}],
        bm25_selected_page_limit=0,
        extract_batch_page_limit=1,
        vector_enabled=False,
    )

    result = extract_contract_fields(
        client,
        "doc-demo",
        [{"name": "payment_ratio", "description": "付款比例"}],
        include_retrieval_metadata=True,
    )["payment_ratio"]

    assert result["pages"] == [3]
    assert result["resolution_method"] == "tree"
    assert "evidence_chain" not in result


def test_reference_target_page_limit_applies_across_targets(monkeypatch):
    loaded_text = []

    async def fake_llm_acompletion(model, prompt):
        if "生成页级检索规格" in prompt:
            return '{"primary_queries":["付款"],"expanded_keywords":[]}'
        if "定位最相关的页面" in prompt:
            return '{"pages":[1],"reason":"reference"}'
        if "定位可能的目标物理页" in prompt:
            return '{"pages":[2,3]}' if "引用目标：附件甲" in prompt else '{"pages":[4,5]}'
        if "目标页" in prompt:
            loaded_text.append(prompt)
            return '{"status":"not_found","value":"","evidence":"","pages":[],"confidence":"Low","reason":"none"}'
        return (
            '{"status":"reference_found","value":"","evidence":"见附件甲和附件乙。","pages":[1],'
            '"confidence":"High","reason":"refs","references":['
            '{"reference_type":"attachment","target_text":"附件甲","relation":"detail"},'
            '{"reference_type":"attachment","target_text":"附件乙","relation":"detail"}]}'
        )

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)
    client = EnhancedStubClient(
        pages=[{"page": 1, "content": "见附件甲和附件乙。"}]
        + [{"page": page, "content": f"目标页{page}"} for page in range(2, 6)],
        structure=[{"title": "文档", "start_page": 1, "end_page": 5, "nodes": []}],
        bm25_selected_page_limit=0,
        reference_total_page_limit=3,
        vector_enabled=False,
    )

    extract_contract_fields(
        client,
        "doc-demo",
        [{"name": "payment", "description": "付款"}],
        include_retrieval_metadata=True,
    )

    combined_loaded = "\n".join(loaded_text)
    assert "目标页2" in combined_loaded
    assert "目标页3" in combined_loaded
    assert "目标页4" in combined_loaded
    assert "目标页5" not in combined_loaded


@pytest.mark.parametrize(
    "payload",
    [
        {
            "status": "reference_found",
            "evidence": "详见附件甲",
            "pages": [1],
            "confidence": "High",
            "references": [{}],
        },
        {
            "status": "reference_found",
            "evidence": "详见附件甲",
            "pages": [1],
            "confidence": "Invalid",
            "references": [{"reference_type": "attachment", "target_text": "附件甲", "relation": "detail"}],
        },
    ],
)
def test_reference_found_requires_complete_references_and_valid_confidence(payload):
    with pytest.raises(ValueError):
        _normalize_field_result("payment", payload)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "status": "found",
            "value": "30%",
            "evidence": "付款比例为30%。",
            "pages": [35],
            "confidence": "High",
            "reason": None,
        },
        {
            "status": "reference_found",
            "value": "",
            "evidence": "详见附件甲。",
            "pages": [35],
            "confidence": "High",
            "reason": "reference",
            "references": [{"reference_type": "attachment", "target_text": "附件甲", "relation": "detail"}],
        },
    ],
)
def test_evidence_pages_must_be_loaded_in_current_batch(payload):
    with pytest.raises(ValueError, match="not loaded"):
        _normalize_field_result("payment", payload, default_pages=[1], allowed_pages=[1])


def test_enhanced_extraction_retries_model_result_using_unloaded_evidence_page(monkeypatch):
    extraction_calls = {"count": 0}

    async def fake_llm_acompletion(model, prompt):
        if "生成页级检索规格" in prompt:
            return '{"primary_queries":["无命中"],"expanded_keywords":[]}'
        if "定位最相关的页面" in prompt:
            return '{"pages":[1,35],"reason":"ranked"}'
        extraction_calls["count"] += 1
        if extraction_calls["count"] == 1:
            return '{"status":"found","value":"30%","evidence":"伪造证据","pages":[35],"confidence":"High","reason":null}'
        return '{"status":"found","value":"10%","evidence":"已加载页含比例10%。","pages":[1],"confidence":"High","reason":null}'

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)
    client = EnhancedStubClient(
        pages=[{"page": 1, "content": "已加载页含比例10%。"}, {"page": 35, "content": "付款比例为30%。"}],
        structure=[{"title": "付款", "start_page": 1, "end_page": 35, "nodes": []}],
        bm25_selected_page_limit=0,
        extract_batch_page_limit=1,
        vector_enabled=False,
    )

    result = extract_contract_fields(
        client,
        "doc-demo",
        [{"name": "payment", "description": "付款比例"}],
        retries=1,
        include_retrieval_metadata=True,
    )["payment"]

    assert result["pages"] == [1]
    assert result["value"] == "10%"
    assert extraction_calls["count"] == 4


def test_enhanced_extraction_error_does_not_trigger_vector_fallback(monkeypatch):
    vector_calls = []

    async def fake_llm_acompletion(model, prompt):
        if "生成页级检索规格" in prompt:
            return '{"primary_queries":["总价"],"expanded_keywords":[]}'
        if "定位最相关的页面" in prompt:
            return '{"pages":[1],"reason":"candidate"}'
        return '{"status":"error","value":"","evidence":"","pages":[1],"confidence":"Low","reason":"OCR unreadable"}'

    async def fail_vector(**kwargs):
        vector_calls.append(kwargs)
        return []

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)
    monkeypatch.setattr("pageindex.contract_extraction.retrieve_vector_candidates", fail_vector)
    client = EnhancedStubClient(
        pages=[{"page": 1, "content": "扫描异常"}],
        structure=[{"title": "价格", "start_page": 1, "end_page": 1, "nodes": []}],
        bm25_selected_page_limit=0,
        vector_enabled=True,
    )

    result = extract_contract_fields(
        client,
        "doc-demo",
        [{"name": "amount", "description": "合同总价"}],
        include_retrieval_metadata=True,
    )["amount"]

    assert result["status"] == "error"
    assert result["reason"] == "OCR unreadable"
    assert vector_calls == []


def test_vector_unavailable_returns_explicit_error_status(monkeypatch):
    events = []

    class CaptureLogger:
        def info(self, payload):
            events.append(payload)

    async def fake_llm_acompletion(model, prompt):
        if "生成页级检索规格" in prompt:
            return '{"primary_queries":["总价"],"expanded_keywords":[]}'
        if "定位最相关的页面" in prompt:
            return '{"pages":[1],"reason":"candidate"}'
        return '{"status":"not_found","value":"","evidence":"","pages":[1],"confidence":"Low","reason":"not here"}'

    async def unavailable_vector(**kwargs):
        raise ValueError("vector embedding service is not configured")

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)
    monkeypatch.setattr("pageindex.contract_extraction.retrieve_vector_candidates", unavailable_vector)
    client = EnhancedStubClient(
        pages=[{"page": 1, "content": "无结果"}],
        structure=[{"title": "价格", "start_page": 1, "end_page": 1, "nodes": []}],
        bm25_selected_page_limit=0,
        vector_enabled=True,
    )

    result = extract_contract_fields(
        client,
        "doc-demo",
        [{"name": "amount", "description": "总价"}],
        retrieval_logger=CaptureLogger(),
    )["amount"]

    assert result["status"] == "error"
    assert result["fallback_status"] == "unavailable"
    assert result["reason"] == "vector fallback is enabled but unavailable"
    assert any(event["event"] == "vector_fallback_unavailable" for event in events)


def test_enhanced_extraction_uses_vector_only_after_initial_candidates_fail(monkeypatch):
    vector_calls = []

    async def fake_llm_acompletion(model, prompt):
        if "生成页级检索规格" in prompt:
            return '{"primary_queries":["无匹配查询"],"expanded_keywords":[]}'
        if "定位最相关的页面" in prompt:
            return '{"pages":[1],"reason":"try tree page"}'
        if '"page": 3' in prompt:
            return '{"status":"found","value":"500万元","evidence":"含税总价为500万元。","pages":[3],"confidence":"High","reason":null}'
        return '{"status":"not_found","value":"","evidence":"","pages":[1],"confidence":"Low","reason":"not here"}'

    async def fake_vector_candidates(**kwargs):
        vector_calls.append(kwargs["processed_pages"])
        return [
            {
                "page": 3,
                "sources": ["vector_fallback"],
                "section_path": "价格",
                "vector_score": 0.9,
                "reason": "向量命中",
                "read_status": "pending",
            }
        ]

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)
    monkeypatch.setattr("pageindex.contract_extraction.retrieve_vector_candidates", fake_vector_candidates)
    client = EnhancedStubClient(
        pages=[{"page": 1, "content": "无关"}, {"page": 3, "content": "含税总价为500万元。"}],
        structure=[{"title": "价格", "start_page": 1, "end_page": 3, "nodes": []}],
        bm25_selected_page_limit=0,
        vector_enabled=True,
    )

    result = extract_contract_fields(
        client,
        "doc-demo",
        [{"name": "amount", "description": "合同总价"}],
        include_retrieval_metadata=True,
    )["amount"]

    assert result["value"] == "500万元"
    assert result["resolution_method"] == "vector_fallback"
    assert vector_calls == [{1}]

import json
from pathlib import Path

import pytest

import service


def test_build_document_tree_returns_doc_and_tree_ids(monkeypatch, tmp_path):
    pdf_path = tmp_path / "demo_contract.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%stub\n")

    output_dir = tmp_path / "output"
    workspace_dir = tmp_path / "workspace"
    captured = {}

    class DummyLogger:
        def __init__(self, file_path, base_dir="artifacts/logs"):
            captured["logger_file_path"] = file_path
            captured["logger_base_dir"] = base_dir

        def info(self, message, *args, **kwargs):
            captured.setdefault("logger_info", []).append(message)

        def exception(self, message, *args, **kwargs):
            captured.setdefault("logger_exception", []).append(message)

    class DummyClient:
        def __init__(self, workspace):
            captured["workspace"] = workspace

        def index(self, file_path, strategy="standard", hybrid_output_dir=None, progress_logger=None):
            captured["index_file_path"] = file_path
            captured["index_strategy"] = strategy
            captured["hybrid_output_dir"] = hybrid_output_dir
            captured["progress_logger"] = progress_logger
            return "doc-demo"

        def get_tree_id(self, doc_id):
            captured["tree_doc_id"] = doc_id
            return "tree-demo"

    monkeypatch.setattr(service, "JsonLogger", DummyLogger)
    monkeypatch.setattr(service, "PageIndexClient", DummyClient)

    result = service.build_document_tree(
        file_path=str(pdf_path),
        output_dir=str(output_dir),
        workspace_dir=str(workspace_dir),
        strategy="hybrid",
    )

    assert result == {
        "status": "success",
        "doc_id": "doc-demo",
        "tree_id": "tree-demo",
        "source_file": str(pdf_path.resolve()),
    }
    assert captured["workspace"] == str(workspace_dir.resolve())
    assert captured["index_file_path"] == str(pdf_path.resolve())
    assert captured["index_strategy"] == "hybrid"
    assert captured["hybrid_output_dir"] == str(output_dir.resolve())
    assert captured["tree_doc_id"] == "doc-demo"
    assert captured["logger_file_path"] == str(pdf_path.resolve())
    assert captured["logger_base_dir"] == str((output_dir.resolve() / "logs"))
    assert captured["logger_info"][0]["event"] == "build_document_tree_started"
    assert captured["logger_info"][-1]["event"] == "build_document_tree_completed"


def test_build_document_tree_converts_word_before_index(monkeypatch, tmp_path):
    word_path = tmp_path / "demo_contract.docx"
    word_path.write_bytes(b"word-content")

    converted_pdf_path = tmp_path / "output" / "demo_contract.pdf"
    converted_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    converted_pdf_path.write_bytes(b"%PDF-1.4\nconverted\n")

    output_dir = tmp_path / "output"
    workspace_dir = tmp_path / "workspace"
    captured = {}

    class DummyLogger:
        def __init__(self, file_path, base_dir="artifacts/logs"):
            captured["logger_file_path"] = file_path
            captured["logger_base_dir"] = base_dir

        def info(self, message, *args, **kwargs):
            captured.setdefault("logger_info", []).append(message)

        def exception(self, message, *args, **kwargs):
            captured.setdefault("logger_exception", []).append(message)

    class DummyClient:
        def __init__(self, workspace):
            captured["workspace"] = workspace

        def index(self, file_path, strategy="standard", hybrid_output_dir=None, progress_logger=None):
            captured["index_file_path"] = file_path
            captured["index_strategy"] = strategy
            captured["hybrid_output_dir"] = hybrid_output_dir
            captured["progress_logger"] = progress_logger
            return "doc-word-demo"

        def get_tree_id(self, doc_id):
            captured["tree_doc_id"] = doc_id
            return "tree-word-demo"

    def fake_convert_word_to_pdf(word_path_arg, output_dir_arg):
        captured["convert_word_path"] = word_path_arg
        captured["convert_output_dir"] = output_dir_arg
        return str(converted_pdf_path.resolve())

    monkeypatch.setattr(service, "JsonLogger", DummyLogger)
    monkeypatch.setattr(service, "PageIndexClient", DummyClient)
    monkeypatch.setattr(service, "convert_word_to_pdf", fake_convert_word_to_pdf)

    result = service.build_document_tree(
        file_path=str(word_path),
        output_dir=str(output_dir),
        workspace_dir=str(workspace_dir),
        strategy="hybrid",
    )

    assert result == {
        "status": "success",
        "doc_id": "doc-word-demo",
        "tree_id": "tree-word-demo",
        "source_file": str(converted_pdf_path.resolve()),
    }
    assert captured["convert_word_path"] == str(word_path.resolve())
    assert captured["convert_output_dir"] == str(output_dir.resolve())
    assert captured["index_file_path"] == str(converted_pdf_path.resolve())
    assert captured["hybrid_output_dir"] == str(output_dir.resolve())
    assert captured["tree_doc_id"] == "doc-word-demo"
    assert captured["logger_file_path"] == str(word_path.resolve())
    assert [item["event"] for item in captured["logger_info"]] == [
        "build_document_tree_started",
        "word_document_detected",
        "word_document_converted",
        "build_document_tree_completed",
    ]


def test_extract_dynamic_schema_persists_result_and_forwards_progress(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    workspace_dir = tmp_path / "workspace"
    schema = {"fields": [{"name": "contract_amount", "description": "合同金额"}]}
    captured = {"progress_calls": []}

    class DummyClient:
        def __init__(self, workspace):
            captured["workspace"] = workspace

        def get_tree_id(self, doc_id):
            captured["tree_doc_id"] = doc_id
            return "tree-demo"

    def fake_extract(
        client,
        doc_id,
        input_schema,
        max_concurrency=8,
        progress_callback=None,
        retrieval_logger=None,
        long_context_mode=False,
    ):
        captured["extract_doc_id"] = doc_id
        captured["extract_schema"] = input_schema
        captured["max_concurrency"] = max_concurrency
        captured["retrieval_logger"] = retrieval_logger
        captured["long_context_mode"] = long_context_mode
        if progress_callback is not None:
            progress_callback(1, 1)
        return {
            "contract_amount": {
                "status": "found",
                "value": "100万元",
                "evidence": "合同总价为人民币壹佰万元整。",
                "pages": [4],
                "confidence": "High",
                "reason": None,
            }
        }

    monkeypatch.setattr(service, "PageIndexClient", DummyClient)
    monkeypatch.setattr(service, "extract_contract_fields", fake_extract)

    result = service.extract_dynamic_schema(
        doc_id="doc-demo",
        schema=schema,
        output_dir=str(output_dir),
        workspace_dir=str(workspace_dir),
        max_concurrency=4,
        progress_callback=lambda current, total: captured["progress_calls"].append((current, total)),
    )

    result_path = Path(result["output_path"])
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert result["status"] == "success"
    assert result["doc_id"] == "doc-demo"
    assert result_path.name == "doc-demo_extraction.json"
    assert payload["status"] == "success"
    assert payload["doc_id"] == "doc-demo"
    assert payload["tree_id"] == "tree-demo"
    assert payload["require_evidence"] is False
    assert payload["extraction_result"]["contract_amount"]["value"] == "100万元"
    assert captured["workspace"] == str(workspace_dir.resolve())
    assert captured["tree_doc_id"] == "doc-demo"
    assert captured["extract_doc_id"] == "doc-demo"
    assert captured["extract_schema"] == schema
    assert captured["max_concurrency"] == 4
    assert captured["long_context_mode"] is False
    assert captured["progress_calls"] == [(1, 1)]
    assert Path(payload["extraction_log_path"]).is_file()
    log_payload = json.loads(Path(payload["extraction_log_path"]).read_text(encoding="utf-8"))
    assert [entry["event"] for entry in log_payload] == ["dynamic_extraction_started", "dynamic_extraction_completed"]


def test_extract_dynamic_schema_forwards_json_schema_field_instruction(monkeypatch, tmp_path):
    captured = {}
    json_schema = {
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "合同价格 合同总价",
                "instruction": "优先提取含税总价",
            }
        },
        "required": ["param1"],
    }

    class DummyClient:
        def __init__(self, workspace):
            self.workspace = workspace

        def get_tree_id(self, doc_id):
            return "tree-demo"

    def fake_extract(
        client,
        doc_id,
        input_schema,
        max_concurrency=8,
        progress_callback=None,
        retrieval_logger=None,
        long_context_mode=False,
    ):
        captured["extract_schema"] = input_schema
        return {
            "param1": {
                "status": "found",
                "value": "500万元",
                "evidence": "含税合同总价为人民币500万元。",
                "pages": [4],
                "confidence": "High",
                "reason": None,
            }
        }

    monkeypatch.setattr(service, "PageIndexClient", DummyClient)
    monkeypatch.setattr(service, "extract_contract_fields", fake_extract)

    service.extract_dynamic_schema(
        doc_id="doc-demo",
        schema=json_schema,
        output_dir=str(tmp_path / "output"),
        workspace_dir=str(tmp_path / "workspace"),
    )

    field = captured["extract_schema"]["fields"][0]
    assert field["name"] == "param1"
    assert field["description"] == "合同价格 合同总价"
    assert field["required"] is True
    assert field["instruction"] == "优先提取含税总价"


def test_extract_dynamic_schema_injects_evidence_and_reformats_output(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    workspace_dir = tmp_path / "workspace"
    captured = {}
    json_schema = {
        "type": "object",
        "properties": {
            "amount": {
                "type": "string",
                "description": "合同金额",
                "instruction": "优先提取含税总价",
            }
        },
        "required": ["amount"],
    }

    class DummyClient:
        def __init__(self, workspace):
            captured["workspace"] = workspace

        def get_tree_id(self, doc_id):
            return "tree-demo"

        def get_document_structure(self, doc_id):
            return json.dumps(
                [
                    {
                        "title": "价格条款",
                        "start_page": 3,
                        "end_page": 5,
                        "nodes": [],
                    }
                ],
                ensure_ascii=False,
            )

    def fake_extract(
        client,
        doc_id,
        input_schema,
        max_concurrency=8,
        progress_callback=None,
        include_retrieval_metadata=False,
        retrieval_logger=None,
        long_context_mode=False,
    ):
        captured["extract_schema"] = input_schema
        captured["include_retrieval_metadata"] = include_retrieval_metadata
        captured["retrieval_logger"] = retrieval_logger
        captured["long_context_mode"] = long_context_mode
        return {
            "amount": {
                "status": "found",
                "value": "500万",
                "evidence": "合同总价暂定为人民币500万元。",
                "pages": [4],
                "confidence": "High",
                "reason": None,
                "resolution_method": "reference_followup",
                "retrieval_sources": ["reference_tree"],
                "evidence_chain": [
                    {"role": "reference_source", "page_number": [3], "original_quote": "详见付款条款。"},
                    {"role": "value_source", "page_number": [4], "original_quote": "合同总价暂定为人民币500万元。"},
                ],
            }
        }

    monkeypatch.setattr(service, "PageIndexClient", DummyClient)
    monkeypatch.setattr(service, "extract_contract_fields", fake_extract)

    result = service.extract_dynamic_schema(
        doc_id="doc-demo",
        schema=json_schema,
        output_dir=str(output_dir),
        workspace_dir=str(workspace_dir),
        require_evidence=True,
    )

    payload = json.loads(Path(result["output_path"]).read_text(encoding="utf-8"))

    assert captured["extract_schema"]["fields"][0]["name"] == "amount"
    assert captured["include_retrieval_metadata"] is True
    assert captured["retrieval_logger"] is not None
    assert captured["long_context_mode"] is False
    assert "优先提取含税总价" in captured["extract_schema"]["fields"][0]["instruction"]
    assert "value、page_number、section_title、original_quote" in captured["extract_schema"]["fields"][0]["instruction"]
    assert "完整合同条款原文" in captured["extract_schema"]["fields"][0]["instruction"]
    assert "original_quote 只返回核心原文片段" in captured["extract_schema"]["fields"][0]["instruction"]
    assert payload["require_evidence"] is True
    assert payload["extraction_result"]["amount"]["value"] == "500万"
    assert payload["extraction_result"]["amount"]["page_number"] == [4]
    assert payload["extraction_result"]["amount"]["section_title"] == "价格条款"
    assert payload["extraction_result"]["amount"]["original_quote"] == "合同总价暂定为人民币500万元。"
    assert payload["extraction_result"]["amount"]["resolution_method"] == "reference_followup"
    assert payload["extraction_result"]["amount"]["retrieval_sources"] == ["reference_tree"]
    assert payload["extraction_result"]["amount"]["evidence_chain"][0]["role"] == "reference_source"


def test_extract_dynamic_schema_long_context_preserves_evidence_shape_without_reading_tree(monkeypatch, tmp_path):
    captured = {}
    json_schema = {
        "type": "object",
        "properties": {"amount": {"type": "string", "description": "合同金额"}},
        "required": ["amount"],
    }

    class DummyClient:
        def __init__(self, workspace):
            self.workspace = workspace

        def get_tree_id(self, doc_id):
            return "tree-demo"

        def get_long_context_payload(self, doc_id):
            return {"doc_id": doc_id, "pages": [{"page": 4, "content": "合同总价为500万元。"}]}

        def get_document_structure(self, doc_id):
            raise AssertionError("long-context evidence must not load document structure")

    def fake_extract(
        client,
        doc_id,
        input_schema,
        max_concurrency=8,
        progress_callback=None,
        include_retrieval_metadata=False,
        retrieval_logger=None,
        long_context_mode=False,
    ):
        captured["long_context_mode"] = long_context_mode
        captured["include_retrieval_metadata"] = include_retrieval_metadata
        return {
            "amount": {
                "status": "found",
                "value": "500万元",
                "evidence": "合同总价为500万元。",
                "pages": [4],
                "confidence": "High",
                "reason": None,
                "resolution_method": "long_context",
                "retrieval_sources": ["full_document"],
            }
        }

    monkeypatch.setattr(service, "PageIndexClient", DummyClient)
    monkeypatch.setattr(service, "extract_contract_fields", fake_extract)

    result = service.extract_dynamic_schema(
        doc_id="doc-demo",
        schema=json_schema,
        output_dir=str(tmp_path / "output"),
        workspace_dir=str(tmp_path / "workspace"),
        require_evidence=True,
        long_context_mode=True,
    )
    payload = json.loads(Path(result["output_path"]).read_text(encoding="utf-8"))

    assert captured == {"long_context_mode": True, "include_retrieval_metadata": True}
    assert payload["long_context_mode"] is True
    assert payload["extraction_result"]["amount"]["section_title"] == ""
    assert payload["extraction_result"]["amount"]["resolution_method"] == "long_context"
    assert payload["extraction_result"]["amount"]["retrieval_sources"] == ["full_document"]


def test_extract_dynamic_schema_long_context_requires_dedicated_page_payload(monkeypatch, tmp_path):
    class DummyClient:
        def __init__(self, workspace):
            self.workspace = workspace

        def get_tree_id(self, doc_id):
            return "tree-demo"

        def get_long_context_payload(self, doc_id):
            return {}

    monkeypatch.setattr(service, "PageIndexClient", DummyClient)

    with pytest.raises(ValueError, match="重新上传并完成建树"):
        service.extract_dynamic_schema(
            doc_id="doc-legacy",
            schema={"fields": [{"name": "amount", "description": "合同金额"}]},
            output_dir=str(tmp_path / "output"),
            workspace_dir=str(tmp_path / "workspace"),
            long_context_mode=True,
        )


def test_extract_dynamic_schema_raises_when_doc_id_not_found(monkeypatch, tmp_path):
    class DummyClient:
        def __init__(self, workspace):
            self.workspace = workspace

        def get_tree_id(self, doc_id):
            return ""

    monkeypatch.setattr(service, "PageIndexClient", DummyClient)

    with pytest.raises(ValueError, match="未找到 doc_id 对应的文档树"):
        service.extract_dynamic_schema(
            doc_id="doc-missing",
            schema={"fields": [{"name": "contract_total_price", "description": "合同总价"}]},
            output_dir=str(tmp_path / "output"),
            workspace_dir=str(tmp_path / "workspace"),
        )


def test_extract_dynamic_schema_raises_when_result_fields_mismatch(monkeypatch, tmp_path):
    class DummyClient:
        def __init__(self, workspace):
            self.workspace = workspace

        def get_tree_id(self, doc_id):
            return "tree-demo"

    def fake_extract(
        client,
        doc_id,
        input_schema,
        max_concurrency=8,
        progress_callback=None,
        retrieval_logger=None,
        long_context_mode=False,
    ):
        return {
            "contract_total_price": {
                "status": "found",
                "value": "100万元",
                "evidence": "合同总价为100万元。",
                "pages": [1],
                "confidence": "High",
                "reason": None,
            }
        }

    monkeypatch.setattr(service, "PageIndexClient", DummyClient)
    monkeypatch.setattr(service, "extract_contract_fields", fake_extract)

    with pytest.raises(ValueError, match="missing="):
        service.extract_dynamic_schema(
            doc_id="doc-demo",
            schema={
                "fields": [
                    {"name": "contract_total_price", "description": "合同总价"},
                    {"name": "equipment_total_price", "description": "设备总价"},
                ]
            },
            output_dir=str(tmp_path / "output"),
            workspace_dir=str(tmp_path / "workspace"),
        )

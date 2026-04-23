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

    class DummyClient:
        def __init__(self, workspace):
            captured["workspace"] = workspace

        def index(self, file_path, strategy="standard", progress_logger=None):
            captured["index_file_path"] = file_path
            captured["index_strategy"] = strategy
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
    assert captured["tree_doc_id"] == "doc-demo"
    assert captured["logger_file_path"] == str(pdf_path.resolve())
    assert captured["logger_base_dir"] == str((output_dir.resolve() / "logs"))


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

    def fake_extract(client, doc_id, input_schema, max_concurrency=8, progress_callback=None):
        captured["extract_doc_id"] = doc_id
        captured["extract_schema"] = input_schema
        captured["max_concurrency"] = max_concurrency
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
    assert captured["progress_calls"] == [(1, 1)]


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

    def fake_extract(client, doc_id, input_schema, max_concurrency=8, progress_callback=None):
        captured["extract_schema"] = input_schema
        return {
            "amount": {
                "status": "found",
                "value": "500万",
                "evidence": "合同总价暂定为人民币500万元。",
                "pages": [4],
                "confidence": "High",
                "reason": None,
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
    assert "value、page_number、section_title、original_quote" in captured["extract_schema"]["fields"][0]["instruction"]
    assert payload["require_evidence"] is True
    assert payload["extraction_result"]["amount"]["value"] == "500万"
    assert payload["extraction_result"]["amount"]["page_number"] == [4]
    assert payload["extraction_result"]["amount"]["section_title"] == "价格条款"
    assert payload["extraction_result"]["amount"]["original_quote"] == "合同总价暂定为人民币500万元。"


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

    def fake_extract(client, doc_id, input_schema, max_concurrency=8, progress_callback=None):
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

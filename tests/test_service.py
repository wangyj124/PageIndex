import json
from pathlib import Path

import pytest

import service


def test_process_and_extract_contract_persists_result(monkeypatch, tmp_path):
    pdf_path = tmp_path / "demo_contract.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%stub\n")

    output_dir = tmp_path / "output"
    workspace_dir = tmp_path / "workspace"
    schema = {"fields": [{"name": "contract_amount", "description": "合同金额"}]}
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

    def fake_extract(client, doc_id, input_schema, max_concurrency=8):
        captured["extract_client"] = client
        captured["extract_doc_id"] = doc_id
        captured["extract_schema"] = input_schema
        captured["max_concurrency"] = max_concurrency
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

    monkeypatch.setattr(service, "JsonLogger", DummyLogger)
    monkeypatch.setattr(service, "PageIndexClient", DummyClient)
    monkeypatch.setattr(service, "extract_contract_fields", fake_extract)

    result = service.process_and_extract_contract(
        file_path=str(pdf_path),
        schema=schema,
        output_dir=str(output_dir),
        workspace_dir=str(workspace_dir),
        strategy="hybrid",
        max_concurrency=4,
    )

    result_path = Path(result["output_path"])
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert result["status"] == "success"
    assert result["doc_id"] == "doc-demo"
    assert result_path.name == "demo_contract_extraction.json"
    assert payload["status"] == "success"
    assert payload["doc_id"] == "doc-demo"
    assert payload["tree_id"] == "tree-demo"
    assert payload["source_file"] == str(pdf_path.resolve())
    assert payload["extraction_result"]["contract_amount"]["value"] == "100万元"

    assert captured["workspace"] == str(workspace_dir.resolve())
    assert captured["index_file_path"] == str(pdf_path.resolve())
    assert captured["index_strategy"] == "hybrid"
    assert captured["extract_doc_id"] == "doc-demo"
    assert captured["extract_schema"] == schema
    assert captured["max_concurrency"] == 4
    assert captured["logger_file_path"] == str(pdf_path.resolve())
    assert captured["logger_base_dir"] == str((output_dir.resolve() / "logs"))


def test_process_and_extract_contract_raises_when_result_fields_mismatch(monkeypatch, tmp_path):
    pdf_path = tmp_path / "demo_contract.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%stub\n")

    class DummyLogger:
        def __init__(self, file_path, base_dir="artifacts/logs"):
            self.file_path = file_path
            self.base_dir = base_dir

    class DummyClient:
        def __init__(self, workspace):
            self.workspace = workspace

        def index(self, file_path, strategy="standard", progress_logger=None):
            return "doc-demo"

        def get_tree_id(self, doc_id):
            return "tree-demo"

    def fake_extract(client, doc_id, input_schema, max_concurrency=8):
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

    monkeypatch.setattr(service, "JsonLogger", DummyLogger)
    monkeypatch.setattr(service, "PageIndexClient", DummyClient)
    monkeypatch.setattr(service, "extract_contract_fields", fake_extract)

    with pytest.raises(ValueError, match="missing="):
        service.process_and_extract_contract(
            file_path=str(pdf_path),
            schema={
                "fields": [
                    {"name": "contract_total_price", "description": "合同总价"},
                    {"name": "equipment_total_price", "description": "设备总价"},
                ]
            },
            output_dir=str(tmp_path / "output"),
            workspace_dir=str(tmp_path / "workspace"),
        )

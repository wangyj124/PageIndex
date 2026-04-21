import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient

import api


@pytest.fixture(autouse=True)
def isolate_api_state(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "API_WORKSPACE", tmp_path / "api_workspace")
    api.task_store.clear()


def _write_schema_file(path: Path, field_name: str = "party_a") -> dict:
    schema = {
        "fields": [
            {
                "name": field_name,
                "description": "甲方",
                "type": "string",
                "required": False,
                "instruction": "",
            }
        ]
    }
    path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    return schema


def test_upload_and_extract_runs_background_task(monkeypatch):
    captured = {}
    schema_path = api.API_WORKSPACE.parent / "contract_fields_xt_full.json"
    expected_schema = _write_schema_file(schema_path)

    def fake_process_and_extract_contract(
        file_path,
        schema,
        output_dir,
        workspace_dir="artifacts/workspace",
        strategy="hybrid",
        max_concurrency=4,
    ):
        captured["file_path"] = file_path
        captured["schema"] = schema
        captured["output_dir"] = output_dir
        captured["workspace_dir"] = workspace_dir
        captured["strategy"] = strategy
        captured["max_concurrency"] = max_concurrency

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / "demo_extraction.json"
        result_file.write_text("{}", encoding="utf-8")
        return {
            "status": "success",
            "output_path": str(result_file.resolve()),
            "doc_id": "doc-api-demo",
        }

    monkeypatch.setattr(api, "FULL_SCHEMA_PATH", schema_path)
    monkeypatch.setattr(api, "process_and_extract_contract", fake_process_and_extract_contract)

    with TestClient(api.app) as client:
        response = client.post(
            "/api/v1/upload_and_extract",
            files={"file": ("demo.pdf", b"%PDF-1.4\napi-demo\n", "application/pdf")},
        )

        assert response.status_code == 202
        body = response.json()
        task_id = body["task_id"]

        task_info = api.task_store[task_id]
        task_dir = api.API_WORKSPACE / task_id
        saved_pdf = task_dir / "input" / "demo.pdf"

        assert task_info["status"] == "completed"
        assert task_info["doc_id"] == "doc-api-demo"
        assert task_info["schema_path"] == str(schema_path.resolve())
        assert saved_pdf.read_bytes() == b"%PDF-1.4\napi-demo\n"
        assert captured["schema"] == expected_schema
        assert captured["file_path"] == str(saved_pdf)
        assert captured["workspace_dir"] == str(task_dir / "workspace")
        assert captured["output_dir"] == str(task_dir / "output")

        query_response = client.get(f"/api/v1/task/{task_id}")
        assert query_response.status_code == 200
        assert query_response.json()["status"] == "completed"


def test_upload_and_extract_rejects_non_pdf():
    with TestClient(api.app) as client:
        response = client.post(
            "/api/v1/upload_and_extract",
            files={"file": ("demo.txt", b"not-pdf", "text/plain")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "仅支持上传 .pdf 文件"


def test_upload_and_extract_marks_failed_task(monkeypatch):
    schema_path = api.API_WORKSPACE.parent / "contract_fields_xt_full.json"
    _write_schema_file(schema_path, field_name="contract_total_price")

    def fake_process_and_extract_contract(*args, **kwargs):
        raise RuntimeError("mock extraction failure")

    monkeypatch.setattr(api, "FULL_SCHEMA_PATH", schema_path)
    monkeypatch.setattr(api, "process_and_extract_contract", fake_process_and_extract_contract)

    with TestClient(api.app) as client:
        response = client.post(
            "/api/v1/upload_and_extract",
            files={"file": ("demo.pdf", b"%PDF-1.4\napi-demo\n", "application/pdf")},
        )

        assert response.status_code == 202
        task_id = response.json()["task_id"]

        query_response = client.get(f"/api/v1/task/{task_id}")
        assert query_response.status_code == 200
        assert query_response.json()["status"] == "failed"
        assert "mock extraction failure" in query_response.json()["error"]


def test_get_task_status_returns_404_for_unknown_task():
    with TestClient(api.app) as client:
        response = client.get("/api/v1/task/not-exists")

    assert response.status_code == 404
    assert response.json()["detail"] == "任务不存在"

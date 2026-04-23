from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient

import api


@pytest.fixture(autouse=True)
def isolate_api_state(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "API_WORKSPACE", tmp_path / "api_workspace")
    monkeypatch.setattr(api, "API_TASKS_DIR", api.API_WORKSPACE / "tasks")
    monkeypatch.setattr(api, "API_SHARED_WORKSPACE", api.API_WORKSPACE / "workspace")
    api.task_store.clear()


def test_upload_and_build_runs_background_task(monkeypatch):
    captured = {}

    def fake_build_document_tree(
        file_path,
        output_dir,
        workspace_dir="artifacts/workspace",
        strategy="hybrid",
    ):
        captured["file_path"] = file_path
        captured["output_dir"] = output_dir
        captured["workspace_dir"] = workspace_dir
        captured["strategy"] = strategy
        return {
            "status": "success",
            "doc_id": "doc-build-demo",
            "tree_id": "tree-build-demo",
            "source_file": file_path,
        }

    monkeypatch.setattr(api, "build_document_tree", fake_build_document_tree)

    with TestClient(api.app) as client:
        response = client.post(
            "/api/v1/upload_and_build",
            files={"file": ("demo.pdf", b"%PDF-1.4\nbuild-demo\n", "application/pdf")},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["code"] == 200
        assert body["message"] == "文件已接收，建树任务已提交到后台。"

        task_id = body["data"]["task_id"]
        task_info = api.task_store[task_id]
        task_dir = api.API_TASKS_DIR / task_id
        saved_pdf = task_dir / "input" / "demo.pdf"

        assert task_info["task_type"] == "build_tree"
        assert task_info["status"] == "completed"
        assert task_info["doc_id"] == "doc-build-demo"
        assert task_info["tree_id"] == "tree-build-demo"
        assert saved_pdf.read_bytes() == b"%PDF-1.4\nbuild-demo\n"
        assert captured["file_path"] == str(saved_pdf)
        assert captured["workspace_dir"] == str(api.API_SHARED_WORKSPACE)
        assert captured["output_dir"] == str(task_dir / "output")

        query_response = client.get(f"/api/v1/task/{task_id}")
        assert query_response.status_code == 200
        query_body = query_response.json()
        assert query_body["code"] == 200
        assert query_body["data"]["status"] == "completed"


def test_extract_runs_background_task_with_progress_and_evidence_flag(monkeypatch):
    captured = {}

    def fake_extract_dynamic_schema(
        doc_id,
        schema,
        output_dir,
        workspace_dir="artifacts/workspace",
        max_concurrency=4,
        progress_callback=None,
        require_evidence=False,
    ):
        captured["doc_id"] = doc_id
        captured["schema"] = schema
        captured["output_dir"] = output_dir
        captured["workspace_dir"] = workspace_dir
        captured["max_concurrency"] = max_concurrency
        captured["require_evidence"] = require_evidence
        if progress_callback is not None:
            progress_callback(1, 2)
            progress_callback(2, 2)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"{doc_id}_extraction.json"
        result_file.write_text("{}", encoding="utf-8")
        return {
            "status": "success",
            "output_path": str(result_file.resolve()),
            "doc_id": doc_id,
        }

    monkeypatch.setattr(api, "extract_dynamic_schema", fake_extract_dynamic_schema)

    with TestClient(api.app) as client:
        response = client.post(
            "/api/v1/extract",
            json={
                "doc_id": "doc-extract-demo",
                "schema_def": {
                    "type": "object",
                    "properties": {
                        "party_a": {"type": "string", "description": "甲方"},
                        "party_b": {"type": "string", "description": "乙方"},
                    },
                },
                "require_evidence": True,
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["code"] == 200
        assert body["message"] == "动态 Schema 抽取任务已提交到后台。"

        task_id = body["data"]["task_id"]
        task_info = api.task_store[task_id]
        task_dir = api.API_TASKS_DIR / task_id

        assert task_info["task_type"] == "extraction"
        assert task_info["status"] == "completed"
        assert task_info["doc_id"] == "doc-extract-demo"
        assert task_info["extracted_count"] == 2
        assert task_info["total_count"] == 2
        assert captured["doc_id"] == "doc-extract-demo"
        assert captured["schema"]["properties"]["party_a"]["description"] == "甲方"
        assert captured["workspace_dir"] == str(api.API_SHARED_WORKSPACE)
        assert captured["output_dir"] == str(task_dir / "output")
        assert captured["require_evidence"] is True
        assert task_info["output_path"].endswith("doc-extract-demo_extraction.json")

        query_response = client.get(f"/api/v1/task/{task_id}")
        assert query_response.status_code == 200
        query_body = query_response.json()
        assert query_body["code"] == 200
        assert query_body["data"]["status"] == "completed"


def test_task_status_returns_progress_for_processing_task():
    api.task_store["processing-task"] = {
        "task_id": "processing-task",
        "task_type": "extraction",
        "status": "processing",
        "doc_id": "doc-demo",
        "extracted_count": 5,
        "total_count": 20,
    }

    with TestClient(api.app) as client:
        response = client.get("/api/v1/task/processing-task")

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 200
    assert body["data"]["progress"] == {
        "current": 5,
        "total": 20,
        "percent": 25.0,
    }


def test_upload_and_build_rejects_non_pdf():
    with TestClient(api.app) as client:
        response = client.post(
            "/api/v1/upload_and_build",
            files={"file": ("demo.txt", b"not-pdf", "text/plain")},
        )

    assert response.status_code == 400
    body = response.json()
    assert body == {
        "code": 400,
        "message": "仅支持上传 .pdf 文件",
        "data": None,
    }


def test_build_task_marks_failed_task(monkeypatch):
    def fake_build_document_tree(*args, **kwargs):
        raise RuntimeError("mock build failure")

    monkeypatch.setattr(api, "build_document_tree", fake_build_document_tree)

    with TestClient(api.app) as client:
        response = client.post(
            "/api/v1/upload_and_build",
            files={"file": ("demo.pdf", b"%PDF-1.4\nbuild-demo\n", "application/pdf")},
        )

        assert response.status_code == 200
        task_id = response.json()["data"]["task_id"]

        query_response = client.get(f"/api/v1/task/{task_id}")
        assert query_response.status_code == 200
        assert query_response.json()["data"]["status"] == "failed"
        assert "mock build failure" in query_response.json()["data"]["error"]


def test_extraction_task_marks_failed_task(monkeypatch):
    def fake_extract_dynamic_schema(*args, **kwargs):
        raise RuntimeError("mock extraction failure")

    monkeypatch.setattr(api, "extract_dynamic_schema", fake_extract_dynamic_schema)

    with TestClient(api.app) as client:
        response = client.post(
            "/api/v1/extract",
            json={
                "doc_id": "doc-extract-demo",
                "schema_def": {"fields": [{"name": "party_a", "description": "甲方"}]},
            },
        )

        assert response.status_code == 200
        task_id = response.json()["data"]["task_id"]

        query_response = client.get(f"/api/v1/task/{task_id}")
        assert query_response.status_code == 200
        assert query_response.json()["data"]["status"] == "failed"
        assert "mock extraction failure" in query_response.json()["data"]["error"]


def test_get_task_status_returns_404_for_unknown_task():
    with TestClient(api.app) as client:
        response = client.get("/api/v1/task/not-exists")

    assert response.status_code == 404
    assert response.json() == {
        "code": 404,
        "message": "任务不存在",
        "data": None,
    }


def test_upload_and_build_returns_429_when_capacity_is_full():
    api.task_store["busy-task"] = {
        "task_id": "busy-task",
        "status": "processing",
    }

    with TestClient(api.app) as client:
        response = client.post(
            "/api/v1/upload_and_build",
            files={"file": ("demo.pdf", b"%PDF-1.4\nbuild-demo\n", "application/pdf")},
        )

    assert response.status_code == 429
    assert response.json() == {
        "code": 429,
        "message": "系统当前正忙，一次只能处理一个任务，请稍后再试",
        "data": None,
    }


def test_extract_returns_wrapped_validation_error():
    with TestClient(api.app) as client:
        response = client.post(
            "/api/v1/extract",
            json={"schema_def": {"type": "object", "properties": {}}},
        )

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == 422
    assert body["message"] == "请求参数校验失败"
    assert isinstance(body["data"], list)

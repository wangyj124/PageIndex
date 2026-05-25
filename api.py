from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from service import build_document_tree, extract_dynamic_schema


app = FastAPI(title="PageIndex Contract Extraction API")

# 使用进程内字典保存任务状态，适合当前轻量级异步接口场景。
task_store: dict[str, dict[str, Any]] = {}
API_WORKSPACE = Path("artifacts/api_workspace")
API_TASKS_DIR = API_WORKSPACE / "tasks"
API_SHARED_WORKSPACE = API_WORKSPACE / "workspace"
COPY_CHUNK_SIZE = 1024 * 1024
MAX_CONCURRENT_TASKS = 1
SUPPORTED_UPLOAD_SUFFIXES = {".pdf", ".doc", ".docx"}

UPLOAD_AND_BUILD_RESPONSE_EXAMPLE = {
    "code": 200,
    "message": "文件已接收，建树任务已提交到后台。",
    "data": {
        "task_id": "a39e1289415e4c838dff9d7146300da0",
    },
}

EXTRACTION_REQUEST_EXAMPLE = {
    "doc_id": "doc_1079388f5212c5d90f705bac4a6ad9612ff5d6cfa284802084d2ddb7d8544fab",
    "schema_def": {
        "type": "object",
        "properties": {
            "party_a": {
                "type": "string",
                "description": "甲方",
            },
            "party_b": {
                "type": "string",
                "description": "乙方",
            },
        },
    },
    "require_evidence": True,
}

EXTRACTION_RESPONSE_EXAMPLE = {
    "code": 200,
    "message": "动态 Schema 抽取任务已提交到后台。",
    "data": {
        "task_id": "bbf7051b87ff4de394c834526a74df8a",
    },
}

EXTRACTION_TASK_QUERY_EXAMPLE = {
    "code": 200,
    "message": "任务状态查询成功",
    "data": {
        "task_id": "bbf7051b87ff4de394c834526a74df8a",
        "task_type": "extraction",
        "doc_id": "doc_1079388f5212c5d90f705bac4a6ad9612ff5d6cfa284802084d2ddb7d8544fab",
        "status": "completed",
        "output_path": (
            "/home/sgc/PageIndex-MJ/artifacts/api_workspace/tasks/"
            "bbf7051b87ff4de394c834526a74df8a/output/"
            "doc_1079388f5212c5d90f705bac4a6ad9612ff5d6cfa284802084d2ddb7d8544fab_extraction.json"
        ),
        "error": "",
        "created_at": "2026-05-18T04:43:41Z",
        "workspace_dir": "/home/sgc/PageIndex-MJ/artifacts/api_workspace/workspace",
        "schema_field_count": 2,
        "extracted_count": 2,
        "total_count": 2,
        "require_evidence": True,
        "started_at": "2026-05-18T04:43:41Z",
        "output_dir": (
            "/home/sgc/PageIndex-MJ/artifacts/api_workspace/tasks/"
            "bbf7051b87ff4de394c834526a74df8a/output"
        ),
        "completed_at": "2026-05-18T04:44:11Z",
        "result_status": "success",
        "extraction_result": {
            "party_a": {
                "value": "潮州深能甘露热电有限公司",
                "page_number": [5],
                "section_title": "第一章 合同协议书 > 买方",
                "original_quote": "买方 名称：潮州深能甘露热电有限公司",
                "status": "found",
                "confidence": "High",
                "reason": "None",
            },
            "party_b": {
                "value": "上海电气集团股份有限公司",
                "page_number": [5],
                "section_title": "第一章 合同协议书 > 卖方",
                "original_quote": "卖方 名称：上海电气集团股份有限公司",
                "status": "found",
                "confidence": "High",
                "reason": "None",
            },
        },
    },
}

BUILD_TREE_TASK_QUERY_EXAMPLE = {
    "code": 200,
    "message": "任务状态查询成功",
    "data": {
        "task_id": "a39e1289415e4c838dff9d7146300da0",
        "task_type": "build_tree",
        "status": "completed",
        "file_name": "甘露机岛合同-20181211版最终-签字版（无价格版）.docx",
        "file_path": (
            "/home/sgc/PageIndex-MJ/artifacts/api_workspace/tasks/"
            "a39e1289415e4c838dff9d7146300da0/input/"
            "甘露机岛合同-20181211版最终-签字版（无价格版）.docx"
        ),
        "output_path": "",
        "error": "",
        "created_at": "2026-05-18T04:28:39Z",
        "workspace_dir": "/home/sgc/PageIndex-MJ/artifacts/api_workspace/workspace",
        "started_at": "2026-05-18T04:28:39Z",
        "output_dir": (
            "/home/sgc/PageIndex-MJ/artifacts/api_workspace/tasks/"
            "a39e1289415e4c838dff9d7146300da0/output"
        ),
        "completed_at": "2026-05-18T04:38:48Z",
        "result_status": "success",
        "doc_id": "doc_1079388f5212c5d90f705bac4a6ad9612ff5d6cfa284802084d2ddb7d8544fab",
        "tree_id": "tree_7b9762e3f834d7c3416b544462b57e4a75dfffbafaafc9f350b3ef9c06f4ccb0",
        "source_file": (
            "/home/sgc/PageIndex-MJ/artifacts/api_workspace/tasks/"
            "a39e1289415e4c838dff9d7146300da0/output/"
            "甘露机岛合同-20181211版最终-签字版（无价格版）.pdf"
        ),
    },
}


class StandardResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": UPLOAD_AND_BUILD_RESPONSE_EXAMPLE})

    code: int = Field(..., description="业务状态码，成功统一为 200")
    message: str = Field(..., description="响应提示信息")
    data: Optional[Any] = Field(default=None, description="响应数据负载")


class ExtractionRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": EXTRACTION_REQUEST_EXAMPLE})

    doc_id: str = Field(..., description="已完成建树的文档 ID")
    schema_def: dict[str, Any] = Field(..., description="动态抽取 schema 定义")
    require_evidence: bool = Field(default=False, description="是否启用带证据溯源的结果结构")


def _utcnow_iso() -> str:
    """统一生成 ISO 时间，便于前后端排查任务状态。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _success_response(message: str, data: Any = None) -> StandardResponse:
    return StandardResponse(code=200, message=message, data=data)


def _error_response(status_code: int, message: str, data: Any = None) -> JSONResponse:
    payload = StandardResponse(code=status_code, message=message, data=data)
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def check_system_capacity() -> None:
    """检查系统当前活跃任务数，超过容量时主动拒绝新任务。"""
    active_task_count = sum(1 for task in task_store.values() if task.get("status") in {"pending", "processing"})
    if active_task_count >= MAX_CONCURRENT_TASKS:
        raise HTTPException(
            status_code=429,
            detail="系统当前正忙，一次只能处理一个任务，请稍后再试",
        )


def _update_task(task_key: str, **fields: Any) -> None:
    task_store.setdefault(task_key, {})
    task_store[task_key].update(fields)


def _build_task_dir(task_id: str) -> Path:
    task_dir = API_TASKS_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir


def _count_schema_fields(schema: dict[str, Any]) -> int:
    """统计 schema 的顶层字段数，兼容 field-list 与 JSON Schema。"""
    if not isinstance(schema, dict):
        return 0
    if isinstance(schema.get("fields"), list):
        return len(schema["fields"])
    if isinstance(schema.get("properties"), dict):
        return len(schema["properties"])
    return 0


def _load_extraction_result(output_path: str) -> dict[str, Any]:
    """读取抽取结果文件中与动态 schema 对齐的字段结果。"""
    if not output_path:
        return {}

    result_path = Path(output_path)
    if not result_path.is_file():
        return {}

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    extraction_result = payload.get("extraction_result", {})
    return extraction_result if isinstance(extraction_result, dict) else {}


def _process_build_tree_task(task_id: str, file_path: str, task_dir: str) -> None:
    """
    后台建树任务。

    该任务只负责上传文件后的索引构建，并将结果文档持久化到共享 workspace。
    """
    task_root = Path(task_dir)
    output_dir = task_root / "output"

    _update_task(
        task_id,
        status="processing",
        started_at=_utcnow_iso(),
        workspace_dir=str(API_SHARED_WORKSPACE.resolve()),
        output_dir=str(output_dir.resolve()),
    )

    try:
        result = build_document_tree(
            file_path=file_path,
            output_dir=str(output_dir),
            workspace_dir=str(API_SHARED_WORKSPACE),
        )
    except Exception as exc:
        _update_task(
            task_id,
            status="failed",
            error=str(exc),
            completed_at=_utcnow_iso(),
        )
        return

    result_payload = {key: value for key, value in result.items() if key != "status"}
    _update_task(
        task_id,
        status="completed",
        completed_at=_utcnow_iso(),
        result_status=result.get("status", ""),
        **result_payload,
    )


def _process_extraction_task(
    task_id: str,
    doc_id: str,
    schema: dict[str, Any],
    require_evidence: bool,
    task_dir: str,
) -> None:
    """
    后台动态 schema 抽取任务。

    该任务复用共享 workspace 中已存在的 doc_id，不再重复上传文件和建树。
    """
    task_root = Path(task_dir)
    output_dir = task_root / "output"
    total_count = _count_schema_fields(schema)

    _update_task(
        task_id,
        status="processing",
        started_at=_utcnow_iso(),
        workspace_dir=str(API_SHARED_WORKSPACE.resolve()),
        output_dir=str(output_dir.resolve()),
        extracted_count=0,
        total_count=total_count,
    )

    def progress_callback(current: int, total: int) -> None:
        _update_task(task_id, extracted_count=current, total_count=total)

    try:
        result = extract_dynamic_schema(
            doc_id=doc_id,
            schema=schema,
            output_dir=str(output_dir),
            workspace_dir=str(API_SHARED_WORKSPACE),
            progress_callback=progress_callback,
            require_evidence=require_evidence,
        )
    except Exception as exc:
        _update_task(
            task_id,
            status="failed",
            error=str(exc),
            completed_at=_utcnow_iso(),
        )
        return

    result_payload = {key: value for key, value in result.items() if key != "status"}
    result_payload["extraction_result"] = _load_extraction_result(str(result_payload.get("output_path", "")))
    _update_task(
        task_id,
        status="completed",
        completed_at=_utcnow_iso(),
        extracted_count=total_count,
        total_count=total_count,
        result_status=result.get("status", ""),
        **result_payload,
    )


@app.exception_handler(HTTPException)
async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
    return _error_response(exc.status_code, str(exc.detail))


@app.exception_handler(RequestValidationError)
async def handle_validation_exception(request: Request, exc: RequestValidationError) -> JSONResponse:
    return _error_response(422, "请求参数校验失败", exc.errors())


@app.exception_handler(Exception)
async def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(500, f"服务器内部错误: {exc}")


@app.post(
    "/api/v1/upload_and_build",
    response_model=StandardResponse,
    responses={
        200: {
            "description": "文件接收成功，建树任务已进入后台队列。",
            "content": {
                "application/json": {
                    "example": UPLOAD_AND_BUILD_RESPONSE_EXAMPLE,
                }
            },
        }
    },
)
async def upload_and_build(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> StandardResponse:
    """
    接收 PDF / Word 二进制流并异步构建文档树。
    """
    check_system_capacity()

    filename = Path(file.filename or "").name
    file_suffix = Path(filename).suffix.lower()
    if not filename or file_suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        raise HTTPException(status_code=400, detail="仅支持上传 .pdf、.doc、.docx 文件")

    task_id = uuid.uuid4().hex
    task_dir = _build_task_dir(task_id)
    input_dir = task_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    saved_file_path = input_dir / filename
    try:
        with saved_file_path.open("wb") as output_file:
            shutil.copyfileobj(file.file, output_file, length=COPY_CHUNK_SIZE)
    finally:
        await file.close()

    _update_task(
        task_id,
        task_id=task_id,
        task_type="build_tree",
        status="pending",
        file_name=filename,
        file_path=str(saved_file_path.resolve()),
        output_path="",
        error="",
        created_at=_utcnow_iso(),
        workspace_dir=str(API_SHARED_WORKSPACE.resolve()),
    )

    background_tasks.add_task(
        _process_build_tree_task,
        task_id,
        str(saved_file_path),
        str(task_dir),
    )

    return _success_response(
        "文件已接收，建树任务已提交到后台。",
        {"task_id": task_id},
    )


@app.post(
    "/api/v1/extract",
    response_model=StandardResponse,
    responses={
        200: {
            "description": "动态 Schema 抽取任务已进入后台队列。",
            "content": {
                "application/json": {
                    "example": EXTRACTION_RESPONSE_EXAMPLE,
                }
            },
        }
    },
)
async def extract_with_dynamic_schema(
    request: ExtractionRequest,
    background_tasks: BackgroundTasks,
) -> StandardResponse:
    """
    基于已有 doc_id 和动态 schema 定义发起抽取任务。
    """
    check_system_capacity()

    task_id = uuid.uuid4().hex
    task_dir = _build_task_dir(task_id)
    total_count = _count_schema_fields(request.schema_def)

    _update_task(
        task_id,
        task_id=task_id,
        task_type="extraction",
        doc_id=request.doc_id,
        status="pending",
        output_path="",
        error="",
        created_at=_utcnow_iso(),
        workspace_dir=str(API_SHARED_WORKSPACE.resolve()),
        schema_field_count=total_count,
        extracted_count=0,
        total_count=total_count,
        require_evidence=request.require_evidence,
    )

    background_tasks.add_task(
        _process_extraction_task,
        task_id,
        request.doc_id,
        request.schema_def,
        request.require_evidence,
        str(task_dir),
    )

    return _success_response(
        "动态 Schema 抽取任务已提交到后台。",
        {"task_id": task_id},
    )


@app.get(
    "/api/v1/task/{task_id}",
    response_model=StandardResponse,
    responses={
        200: {
            "description": "任务状态查询成功。",
            "content": {
                "application/json": {
                    "examples": {
                        "extraction_task": {
                            "summary": "字段提取任务查询",
                            "value": EXTRACTION_TASK_QUERY_EXAMPLE,
                        },
                        "build_tree_task": {
                            "summary": "建树任务查询",
                            "value": BUILD_TREE_TASK_QUERY_EXAMPLE,
                        },
                    }
                }
            },
        }
    },
)
async def get_task_status(task_id: str) -> StandardResponse:
    """查询指定任务的当前状态与产物路径。"""
    task_info = task_store.get(task_id)
    if task_info is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    response = dict(task_info)
    if response.get("status") == "processing" and response.get("total_count") is not None:
        total = int(response.get("total_count") or 0)
        current = int(response.get("extracted_count") or 0)
        percent = round((current / total) * 100, 2) if total > 0 else 0.0
        response["progress"] = {
            "current": current,
            "total": total,
            "percent": percent,
        }

    return _success_response("任务状态查询成功", response)

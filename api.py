from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from service import build_document_tree, extract_dynamic_schema


app = FastAPI(title="PageIndex Contract Extraction API")

# 使用进程内字典保存任务状态，适合当前轻量级异步接口场景。
task_store: dict[str, dict[str, Any]] = {}
API_WORKSPACE = Path("artifacts/api_workspace")
API_TASKS_DIR = API_WORKSPACE / "tasks"
API_SHARED_WORKSPACE = API_WORKSPACE / "workspace"
COPY_CHUNK_SIZE = 1024 * 1024
MAX_CONCURRENT_TASKS = 1


class StandardResponse(BaseModel):
    code: int = Field(..., description="业务状态码，成功统一为 200")
    message: str = Field(..., description="响应提示信息")
    data: Optional[Any] = Field(default=None, description="响应数据负载")


class ExtractionRequest(BaseModel):
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


@app.post("/api/v1/upload_and_build", response_model=StandardResponse)
async def upload_and_build(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> StandardResponse:
    """
    接收 PDF 二进制流并异步构建文档树。
    """
    check_system_capacity()

    filename = Path(file.filename or "").name
    if not filename or Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="仅支持上传 .pdf 文件")

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


@app.post("/api/v1/extract", response_model=StandardResponse)
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


@app.get("/api/v1/task/{task_id}", response_model=StandardResponse)
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

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile

from service import process_and_extract_contract


app = FastAPI(title="PageIndex Contract Extraction API")

# 使用进程内字典保存任务状态，适合当前轻量级异步接口场景。
task_store: dict[str, dict[str, Any]] = {}
API_WORKSPACE = Path("artifacts/api_workspace")
FULL_SCHEMA_PATH = Path(__file__).resolve().parent / "sample_data" / "schemas" / "contract_fields_xt_full.json"
COPY_CHUNK_SIZE = 1024 * 1024


def _utcnow_iso() -> str:
    """统一生成 ISO 时间，便于前后端排查任务状态。"""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _update_task(task_key: str, **fields: Any) -> None:
    task_store.setdefault(task_key, {})
    task_store[task_key].update(fields)


def _load_full_contract_schema() -> dict[str, Any]:
    """加载内置 XT 全字段 schema。"""
    return json.loads(FULL_SCHEMA_PATH.read_text(encoding="utf-8"))


def _process_task(task_id: str, file_path: str, task_dir: str) -> None:
    """
    后台任务入口。

    这里负责切换任务状态，并调用核心 Facade 业务函数执行真正的建树和抽取。
    """
    task_root = Path(task_dir)
    workspace_dir = task_root / "workspace"
    output_dir = task_root / "output"

    _update_task(
        task_id,
        status="processing",
        started_at=_utcnow_iso(),
        workspace_dir=str(workspace_dir.resolve()),
        output_dir=str(output_dir.resolve()),
    )

    try:
        schema = _load_full_contract_schema()
        result = process_and_extract_contract(
            file_path=file_path,
            schema=schema,
            output_dir=str(output_dir),
            workspace_dir=str(workspace_dir),
        )
    except Exception as exc:
        _update_task(
            task_id,
            status="failed",
            error=str(exc),
            completed_at=_utcnow_iso(),
        )
        return

    _update_task(
        task_id,
        status="completed",
        output_path=result["output_path"],
        doc_id=result["doc_id"],
        completed_at=_utcnow_iso(),
    )


@app.post("/api/v1/upload_and_extract", status_code=202)
async def upload_and_extract(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict[str, str]:
    """
    接收 PDF 二进制流，保存到任务隔离目录，并将抽取任务放入后台执行。
    """
    filename = Path(file.filename or "").name
    if not filename or Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="仅支持上传 .pdf 文件")

    task_id = uuid.uuid4().hex
    task_dir = API_WORKSPACE / task_id
    input_dir = task_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    saved_file_path = input_dir / filename
    try:
        # 按块写入上传流，避免一次性把整个 PDF 读入内存。
        with saved_file_path.open("wb") as output_file:
            shutil.copyfileobj(file.file, output_file, length=COPY_CHUNK_SIZE)
    finally:
        await file.close()

    _update_task(
        task_id,
        task_id=task_id,
        status="pending",
        file_name=filename,
        file_path=str(saved_file_path.resolve()),
        output_path="",
        error="",
        created_at=_utcnow_iso(),
        schema_path=str(FULL_SCHEMA_PATH.resolve()),
    )

    background_tasks.add_task(
        _process_task,
        task_id,
        str(saved_file_path),
        str(task_dir),
    )

    return {
        "task_id": task_id,
        "message": "文件已接收，合同抽取任务已提交到后台，默认使用 XT 全字段 schema。",
    }


@app.get("/api/v1/task/{task_id}")
async def get_task_status(task_id: str) -> dict[str, Any]:
    """查询指定任务的当前状态与产物路径。"""
    task_info = task_store.get(task_id)
    if task_info is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task_info

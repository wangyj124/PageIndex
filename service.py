from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pageindex import PageIndexClient, extract_contract_fields, normalize_schema
from pageindex.logging_utils import JsonLogger


def _validate_extraction_result(schema: dict[str, Any], extraction_result: dict[str, Any]) -> None:
    """确保抽取结果与 schema 中定义的字段完全一致。"""
    expected_field_names = {field.name for field in normalize_schema(schema)}
    actual_field_names = set(extraction_result.keys())

    missing_fields = sorted(expected_field_names - actual_field_names)
    extra_fields = sorted(actual_field_names - expected_field_names)
    if missing_fields or extra_fields:
        raise ValueError(
            "抽取结果字段与 schema 不一致："
            f"missing={missing_fields or []}, extra={extra_fields or []}"
        )


def process_and_extract_contract(
    file_path: str,
    schema: dict[str, Any],
    output_dir: str,
    workspace_dir: str = "artifacts/workspace",
    strategy: str = "hybrid",
    max_concurrency: int = 4,
) -> dict[str, str]:
    """
    统一封装合同解析与字段抽取流程。

    该函数作为 Facade 入口，对外屏蔽底层建树、抽取和结果持久化细节。
    """
    pdf_path = Path(file_path).expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    workspace_path = Path(workspace_dir).expanduser().resolve()
    workspace_path.mkdir(parents=True, exist_ok=True)

    output_path_dir = Path(output_dir).expanduser().resolve()
    output_path_dir.mkdir(parents=True, exist_ok=True)

    # 将索引进度日志落到本次输出目录，便于定位单次任务问题。
    progress_logger = JsonLogger(str(pdf_path), base_dir=str(output_path_dir / "logs"))
    client = PageIndexClient(workspace=str(workspace_path))

    # 先建树，再基于稳定的 doc_id/tree_id 执行字段抽取。
    doc_id = client.index(
        str(pdf_path),
        strategy=strategy,
        progress_logger=progress_logger,
    )
    tree_id = client.get_tree_id(doc_id)
    extraction_result = extract_contract_fields(
        client,
        doc_id,
        schema,
        max_concurrency=max_concurrency,
    )
    _validate_extraction_result(schema, extraction_result)

    payload = {
        "status": "success",
        "doc_id": doc_id,
        "tree_id": tree_id,
        "source_file": str(pdf_path),
        "extraction_result": extraction_result,
    }

    result_path = output_path_dir / f"{pdf_path.stem}_extraction.json"
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "status": "success",
        "output_path": str(result_path),
        "doc_id": doc_id,
    }

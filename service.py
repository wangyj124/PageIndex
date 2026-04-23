from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from pageindex import PageIndexClient, extract_contract_fields, normalize_schema
from pageindex.logging_utils import JsonLogger


def _schema_field_names(schema: dict[str, Any] | list[dict[str, Any]]) -> set[str]:
    """提取 schema 中顶层字段名，兼容 field-list 与 JSON Schema 两种输入。"""
    if isinstance(schema, list):
        return {str(item.get("name", "")).strip() for item in schema if isinstance(item, dict) and item.get("name")}

    if isinstance(schema, dict) and "fields" in schema:
        fields = schema.get("fields", [])
        if not isinstance(fields, list):
            raise TypeError("schema['fields'] must be a list")
        return {str(item.get("name", "")).strip() for item in fields if isinstance(item, dict) and item.get("name")}

    if isinstance(schema, dict) and isinstance(schema.get("properties"), dict):
        return {str(name).strip() for name in schema["properties"].keys() if str(name).strip()}

    raise TypeError("schema must be a field-definition list, a dict with 'fields', or a JSON Schema with 'properties'")


def _normalize_to_extraction_schema(schema: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any] | list[dict[str, Any]]:
    """
    将标准 JSON Schema 顶层 properties 转为当前抽取引擎可理解的 field-list 结构。

    如果输入本身已经是现有的 `fields` 风格，则原样返回，保证向下兼容。
    """
    if isinstance(schema, list):
        return schema

    if isinstance(schema, dict) and "fields" in schema:
        return schema

    if not isinstance(schema, dict) or not isinstance(schema.get("properties"), dict):
        raise TypeError("schema must be a dict with 'fields' or a JSON Schema dict with 'properties'")

    required_fields = set(schema.get("required", []))
    fields = []
    for name, property_schema in schema["properties"].items():
        if not isinstance(property_schema, dict):
            continue

        description = str(property_schema.get("description", "")).strip() or f"提取字段“{name}”的值"
        field_type = property_schema.get("type", "string")
        instruction = ""

        # evidence 注入后的字段会被包装成 object，这里退回到 value 子字段的定义，
        # 以便继续适配当前 field-based 的抽取引擎。
        if field_type == "object" and isinstance(property_schema.get("properties"), dict):
            value_schema = property_schema["properties"].get("value", {})
            if isinstance(value_schema, dict):
                description = str(value_schema.get("description", "")).strip() or description
                field_type = value_schema.get("type", "string")
            instruction = "请基于原文同时给出 value、page_number、section_title、original_quote。"

        fields.append(
            {
                "name": str(name),
                "description": description,
                "type": str(field_type or "string"),
                "required": name in required_fields,
                "instruction": instruction,
            }
        )
    return {"fields": fields}


def _validate_extraction_result(
    schema: dict[str, Any] | list[dict[str, Any]],
    extraction_result: dict[str, Any],
) -> None:
    """确保抽取结果与 schema 中定义的字段完全一致。"""
    expected_field_names = _schema_field_names(schema)
    actual_field_names = set(extraction_result.keys())

    missing_fields = sorted(expected_field_names - actual_field_names)
    extra_fields = sorted(actual_field_names - expected_field_names)
    if missing_fields or extra_fields:
        raise ValueError(
            "抽取结果字段与 schema 不一致："
            f"missing={missing_fields or []}, extra={extra_fields or []}"
        )


def _inject_evidence_to_schema(original_schema: dict[str, Any]) -> dict[str, Any]:
    """
    为标准 JSON Schema 的顶层字段自动注入溯源证据结构。

    当前实现采用“遍历顶层 properties”的方式处理输入 schema：
    - 复制原始 schema，避免原地修改调用方传入的数据。
    - 逐个读取顶层字段定义，将原本直接返回值的字段包装成 object。
    - 包装后的 object 强制包含 `value/page_number/section_title/original_quote` 四个字段。

    由于当前抽取引擎是面向“顶层字段”的，所以这里不递归处理任意深度的嵌套对象；
    如果未来需要支持深层 JSON Schema，可以在这里继续向内递归遍历 `properties`。
    """
    if not isinstance(original_schema, dict) or not isinstance(original_schema.get("properties"), dict):
        raise TypeError("require_evidence=True 时，schema_def 必须是带有 'properties' 的标准 JSON Schema dict")

    schema_with_evidence = deepcopy(original_schema)
    new_properties: dict[str, Any] = {}

    for field_name, field_schema in schema_with_evidence["properties"].items():
        field_schema = field_schema if isinstance(field_schema, dict) else {}
        value_type = field_schema.get("type", "string")
        value_description = str(field_schema.get("description", "")).strip() or f"字段“{field_name}”的值"

        new_properties[field_name] = {
            "type": "object",
            "properties": {
                "value": {
                    "type": value_type,
                    "description": value_description,
                },
                "page_number": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "出处所在的物理页码列表，找不到填 null",
                },
                "section_title": {
                    "type": "string",
                    "description": "出处所在的章节或标题名称",
                },
                "original_quote": {
                    "type": "string",
                    "description": "支撑该提取结果的原文摘录片段（10-50字）",
                },
            },
            "required": ["value", "page_number", "section_title", "original_quote"],
        }

    schema_with_evidence["properties"] = new_properties
    return schema_with_evidence


def _find_section_title_by_page(structure: list[dict[str, Any]], page_number: int) -> str:
    """根据页码在树结构中查找最匹配的章节路径。"""

    def walk(nodes: list[dict[str, Any]], path: list[str]) -> str:
        for node in nodes:
            title = str(node.get("title", "")).strip()
            start_page = node.get("start_page", node.get("start_index"))
            end_page = node.get("end_page", node.get("end_index", start_page))

            if not isinstance(start_page, int):
                continue
            if not isinstance(end_page, int):
                end_page = start_page
            if not (start_page <= page_number <= end_page):
                continue

            current_path = path + ([title] if title else [])
            child_result = walk(node.get("nodes", []), current_path)
            if child_result:
                return child_result
            return " > ".join(current_path)
        return ""

    return walk(structure, [])


def _build_evidence_result(
    extraction_result: dict[str, Any],
    structure: list[dict[str, Any]],
) -> dict[str, Any]:
    """将现有抽取结果重组为带 evidence 溯源信息的输出结构。"""
    evidence_result: dict[str, Any] = {}
    for field_name, payload in extraction_result.items():
        page_numbers = payload.get("pages") or []
        first_page = page_numbers[0] if page_numbers else None
        section_title = _find_section_title_by_page(structure, first_page) if isinstance(first_page, int) else ""

        evidence_result[field_name] = {
            "value": payload.get("value", ""),
            "page_number": page_numbers or None,
            "section_title": section_title,
            "original_quote": payload.get("evidence", ""),
            "status": payload.get("status"),
            "confidence": payload.get("confidence"),
            "reason": payload.get("reason"),
        }
    return evidence_result


def build_document_tree(
    file_path: str,
    output_dir: str,
    workspace_dir: str = "artifacts/workspace",
    strategy: str = "hybrid",
) -> dict[str, str]:
    """
    只负责接收本地 PDF 并构建文档树。

    返回稳定的 doc_id / tree_id，供后续按动态 schema 二次抽取使用。
    """
    pdf_path = Path(file_path).expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    workspace_path = Path(workspace_dir).expanduser().resolve()
    workspace_path.mkdir(parents=True, exist_ok=True)

    output_path_dir = Path(output_dir).expanduser().resolve()
    output_path_dir.mkdir(parents=True, exist_ok=True)

    progress_logger = JsonLogger(str(pdf_path), base_dir=str(output_path_dir / "logs"))
    client = PageIndexClient(workspace=str(workspace_path))

    doc_id = client.index(
        str(pdf_path),
        strategy=strategy,
        progress_logger=progress_logger,
    )
    tree_id = client.get_tree_id(doc_id)

    return {
        "status": "success",
        "doc_id": doc_id,
        "tree_id": tree_id,
        "source_file": str(pdf_path),
    }


def extract_dynamic_schema(
    doc_id: str,
    schema: dict[str, Any],
    output_dir: str,
    workspace_dir: str = "artifacts/workspace",
    max_concurrency: int = 4,
    progress_callback: Callable[[int, int], None] | None = None,
    require_evidence: bool = False,
) -> dict[str, str]:
    """
    基于已有 doc_id 和动态 schema 执行字段抽取。

    抽取结果会落盘到 output_dir/{doc_id}_extraction.json。
    """
    if not doc_id.strip():
        raise ValueError("doc_id 不能为空")

    workspace_path = Path(workspace_dir).expanduser().resolve()
    workspace_path.mkdir(parents=True, exist_ok=True)

    output_path_dir = Path(output_dir).expanduser().resolve()
    output_path_dir.mkdir(parents=True, exist_ok=True)

    client = PageIndexClient(workspace=str(workspace_path))
    tree_id = client.get_tree_id(doc_id)
    if not tree_id:
        raise ValueError(f"未找到 doc_id 对应的文档树: {doc_id}")

    effective_schema = _inject_evidence_to_schema(schema) if require_evidence else schema
    extraction_schema = _normalize_to_extraction_schema(effective_schema)
    extraction_result = extract_contract_fields(
        client,
        doc_id,
        extraction_schema,
        max_concurrency=max_concurrency,
        progress_callback=progress_callback,
    )
    _validate_extraction_result(extraction_schema, extraction_result)

    final_result = extraction_result
    if require_evidence:
        structure = json.loads(client.get_document_structure(doc_id))
        final_result = _build_evidence_result(extraction_result, structure)

    payload = {
        "status": "success",
        "doc_id": doc_id,
        "tree_id": tree_id,
        "require_evidence": require_evidence,
        "extraction_result": final_result,
    }

    result_path = output_path_dir / f"{doc_id}_extraction.json"
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "status": "success",
        "output_path": str(result_path),
        "doc_id": doc_id,
    }

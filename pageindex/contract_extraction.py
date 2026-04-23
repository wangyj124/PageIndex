import asyncio
import concurrent.futures
import json
from dataclasses import dataclass
from enum import Enum

from .llm import extract_json, llm_acompletion


class ConfidenceLevel(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class ExtractionStatus(str, Enum):
    FOUND = "found"
    NOT_FOUND = "not_found"
    ERROR = "error"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    description: str
    type: str = "string"
    required: bool = False
    instruction: str = ""


def normalize_schema(schema):
    if isinstance(schema, dict) and "fields" in schema:
        schema = schema["fields"]
    if not isinstance(schema, list):
        raise TypeError("schema must be a list of field definitions or a dict containing 'fields'")

    fields = []
    for item in schema:
        if not isinstance(item, dict):
            raise TypeError("each field definition must be a dict")
        name = str(item.get("name", "")).strip()
        description = str(item.get("description", "")).strip()
        if not name or not description:
            raise ValueError("each field definition must include non-empty 'name' and 'description'")
        fields.append(
            FieldSpec(
                name=name,
                description=description,
                type=str(item.get("type", "string")).strip() or "string",
                required=bool(item.get("required", False)),
                instruction=str(item.get("instruction", "")).strip(),
            )
        )
    return fields


def _flatten_structure(structure, path=None):
    path = path or []
    rows = []
    for node in structure:
        title = str(node.get("title", "")).strip()
        if not title:
            continue
        start_page = node.get("start_page", node.get("start_index"))
        end_page = node.get("end_page", node.get("end_index", start_page))
        summary = str(node.get("summary", "")).strip()
        current_path = path + [title]
        rows.append(
            {
                "path": " > ".join(current_path),
                "title": title,
                "summary": summary,
                "start_page": start_page,
                "end_page": end_page,
            }
        )
        rows.extend(_flatten_structure(node.get("nodes", []), current_path))
    return rows


def _build_structure_digest(structure):
    rows = _flatten_structure(structure)
    return json.dumps(rows, ensure_ascii=False, indent=2)


def _format_page_selection(page_numbers):
    ordered = sorted({int(page) for page in page_numbers})
    if not ordered:
        return ""

    ranges = []
    start = end = ordered[0]
    for page in ordered[1:]:
        if page == end + 1:
            end = page
            continue
        ranges.append(f"{start}-{end}" if start != end else str(start))
        start = end = page
    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ",".join(ranges)


def _normalize_page_list(value):
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, str) and value.strip().isdigit():
        return [int(value.strip())]
    if not isinstance(value, list):
        return []

    pages = []
    for item in value:
        if isinstance(item, int):
            pages.append(item)
        elif isinstance(item, str) and item.strip().isdigit():
            pages.append(int(item.strip()))
    return sorted({page for page in pages if page > 0})


def _normalize_field_result(field_name, payload, default_pages=None):
    default_pages = default_pages or []
    if not isinstance(payload, dict):
        raise ValueError("field result must be a JSON object")

    status = str(payload.get("status", "")).strip().lower()
    if status not in {item.value for item in ExtractionStatus}:
        raise ValueError(f"invalid status for field {field_name}: {status!r}")

    if status != ExtractionStatus.FOUND.value:
        reason = str(payload.get("reason", "")).strip() or "model did not find a supported answer"
        return {
            "status": status,
            "value": "",
            "evidence": "",
            "pages": _normalize_page_list(payload.get("pages")) or list(default_pages),
            "confidence": ConfidenceLevel.LOW.value,
            "reason": reason,
        }

    confidence = str(payload.get("confidence", "")).strip()
    if confidence not in {item.value for item in ConfidenceLevel}:
        raise ValueError(f"invalid confidence for field {field_name}: {confidence!r}")

    value = str(payload.get("value", "")).strip()
    evidence = str(payload.get("evidence", "")).strip()
    if not value or not evidence:
        raise ValueError(f"field {field_name} must include both value and evidence when status='found'")

    return {
        "status": ExtractionStatus.FOUND.value,
        "value": value,
        "evidence": evidence,
        "pages": _normalize_page_list(payload.get("pages")) or list(default_pages),
        "confidence": confidence,
        "reason": str(payload.get("reason", "")).strip() or None,
    }


def _build_locator_prompt(field, structure_digest):
    return f"""
You are locating the most relevant pages for one contract extraction task.

Task field:
- name: {field.name}
- description: {field.description}
- type: {field.type}
- required: {field.required}
- instruction: {field.instruction or "N/A"}

Document structure digest:
{structure_digest}

Rules:
- Use only the structure digest above.
- Return the smallest relevant page set, at most 3 pages total.
- Prefer pages whose title or summary directly mention the target field.
- If no strong candidate exists, return an empty pages list.

Return JSON only with this schema:
{{
  "pages": [4, 5],
  "reason": "why these pages are likely relevant"
}}
""".strip()


def _build_extraction_prompt(field, page_content_json):
    return f"""
You extract exactly one contract field from the provided page text.

Field spec:
- name: {field.name}
- description: {field.description}
- type: {field.type}
- required: {field.required}
- instruction: {field.instruction or "N/A"}

Page content:
{page_content_json}

Output rules:
- Return JSON only.
- status must be exactly one of: "found", "not_found", "error"
- confidence must be exactly one of: "High", "Medium", "Low"
- If status is "found":
  - provide non-empty value
  - provide non-empty evidence copied from the page text
  - include pages as an array of physical page numbers
  - confidence standard:
    - High: the value is explicitly stated in the text and the evidence directly supports it
    - Medium: the value is derived from nearby context or mild inference
    - Low: the evidence is indirect, ambiguous, or only weakly supportive
- If status is "not_found" or "error":
  - set value to ""
  - set evidence to ""
  - set confidence to "Low"
  - include a non-empty reason

Return JSON with this schema:
{{
  "status": "found",
  "value": "example",
  "evidence": "exact supporting text",
  "pages": [4],
  "confidence": "High",
  "reason": null
}}
""".strip()


async def _run_json_prompt(model, prompt, retries=1, timeout_seconds=45):
    last_error = None
    for _ in range(retries + 1):
        response = await asyncio.wait_for(llm_acompletion(model, prompt), timeout=timeout_seconds)
        payload = extract_json(response)
        if payload:
            return payload
        last_error = response
    raise ValueError(f"model did not return valid JSON: {last_error!r}")


async def _extract_one_field(client, doc_id, field, structure_digest, semaphore, retries=1, timeout_seconds=45):
    async with semaphore:
        try:
            locator_payload = await _run_json_prompt(
                client.retrieve_model,
                _build_locator_prompt(field, structure_digest),
                retries=retries,
                timeout_seconds=timeout_seconds,
            )
            pages = _normalize_page_list(locator_payload.get("pages"))
            if not pages:
                return field.name, {
                    "status": ExtractionStatus.NOT_FOUND.value,
                    "value": "",
                    "evidence": "",
                    "pages": [],
                    "confidence": ConfidenceLevel.LOW.value,
                    "reason": str(locator_payload.get("reason", "")).strip() or "unable to locate relevant pages from summaries",
                }

            page_content_json = client.get_page_content(doc_id, _format_page_selection(pages))
            last_error = None
            for _ in range(retries + 1):
                extraction_payload = await _run_json_prompt(
                    client.retrieve_model,
                    _build_extraction_prompt(field, page_content_json),
                    retries=0,
                    timeout_seconds=timeout_seconds,
                )
                try:
                    return field.name, _normalize_field_result(field.name, extraction_payload, default_pages=pages)
                except ValueError as exc:
                    last_error = exc
            raise last_error or ValueError("field extraction validation failed")
        except Exception as exc:
            return field.name, {
                "status": ExtractionStatus.ERROR.value,
                "value": "",
                "evidence": "",
                "pages": [],
                "confidence": ConfidenceLevel.LOW.value,
                "reason": str(exc),
            }


async def _extract_contract_fields_async(
    client,
    doc_id,
    schema,
    max_concurrency=8,
    timeout_seconds=45,
    retries=1,
    progress_callback=None,
):
    fields = normalize_schema(schema)
    structure = json.loads(client.get_document_structure(doc_id))
    structure_digest = _build_structure_digest(structure)
    semaphore = asyncio.Semaphore(max(1, min(max_concurrency, len(fields))))
    total_fields = len(fields)
    completed_count = 0

    async def run_field(field):
        nonlocal completed_count
        result = await _extract_one_field(
            client,
            doc_id,
            field,
            structure_digest,
            semaphore,
            retries=retries,
            timeout_seconds=timeout_seconds,
        )
        completed_count += 1
        if progress_callback is not None:
            progress_callback(completed_count, total_fields)
        return result

    tasks = [run_field(field) for field in fields]
    results = await asyncio.gather(*tasks)
    return {name: payload for name, payload in results}


def extract_contract_fields(
    client,
    doc_id,
    schema,
    max_concurrency=8,
    timeout_seconds=45,
    retries=1,
    progress_callback=None,
):
    coro = _extract_contract_fields_async(
        client,
        doc_id,
        schema,
        max_concurrency=max_concurrency,
        timeout_seconds=timeout_seconds,
        retries=retries,
        progress_callback=progress_callback,
    )
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)


__all__ = [
    "ConfidenceLevel",
    "ExtractionStatus",
    "FieldSpec",
    "extract_contract_fields",
    "normalize_schema",
]

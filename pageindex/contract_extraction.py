import asyncio
import concurrent.futures
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml

from .bm25_retrieve import build_page_documents, retrieve_bm25_candidates
from .llm import count_tokens, extract_json, llm_acompletion
from .reference_resolution import filter_valid_reference_pages, normalize_reference_text
from .vector_retrieve import retrieve_vector_candidates


logger = logging.getLogger(__name__)
QUERY_SPEC_PROMPT_VERSION = "v2"
FIELD_INSTRUCTIONS_PATH = Path(__file__).parent / "field_instructions.yaml"
_FIELD_INSTRUCTION_CACHE = None


class ConfidenceLevel(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class ExtractionStatus(str, Enum):
    FOUND = "found"
    NOT_FOUND = "not_found"
    ERROR = "error"
    REFERENCE_FOUND = "reference_found"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    description: str
    type: str = "string"
    required: bool = False
    instruction: str = ""


def _normalize_field_instruction_key(value):
    return "".join(str(value or "").split())


def _load_field_instruction_map():
    global _FIELD_INSTRUCTION_CACHE
    if _FIELD_INSTRUCTION_CACHE is not None:
        return _FIELD_INSTRUCTION_CACHE
    try:
        with open(FIELD_INSTRUCTIONS_PATH, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning("Field instruction config not found: %s", FIELD_INSTRUCTIONS_PATH)
        _FIELD_INSTRUCTION_CACHE = {}
        return _FIELD_INSTRUCTION_CACHE
    raw_mapping = payload.get("field_instructions", payload)
    if not isinstance(raw_mapping, dict):
        logger.warning("Field instruction config must be a mapping: %s", FIELD_INSTRUCTIONS_PATH)
        _FIELD_INSTRUCTION_CACHE = {}
        return _FIELD_INSTRUCTION_CACHE
    _FIELD_INSTRUCTION_CACHE = {
        _normalize_field_instruction_key(description): str(instruction).strip()
        for description, instruction in raw_mapping.items()
        if _normalize_field_instruction_key(description) and str(instruction).strip() and str(instruction).strip() != "/"
    }
    return _FIELD_INSTRUCTION_CACHE


def _default_instruction_for_description(description):
    return _load_field_instruction_map().get(_normalize_field_instruction_key(description), "")


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
        instruction = str(item.get("instruction", "")).strip()
        if not instruction:
            instruction = _default_instruction_for_description(description)
        fields.append(
            FieldSpec(
                name=name,
                description=description,
                type=str(item.get("type", "string")).strip() or "string",
                required=bool(item.get("required", False)),
                instruction=instruction,
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
        content_start_page = node.get("content_start_page", node.get("content_start_index", start_page))
        content_end_page = node.get("content_end_page", node.get("content_end_index", end_page))
        summary = str(node.get("summary") or node.get("prefix_summary") or "").strip()
        current_path = path + [title]
        rows.append(
            {
                "path": " > ".join(current_path),
                "title": title,
                "summary": summary,
                "start_page": start_page,
                "end_page": end_page,
                "content_start_page": content_start_page,
                "content_end_page": content_end_page,
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


def _normalize_page_list_input(value):
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
    return pages


def _normalize_page_list(value):
    return sorted({page for page in _normalize_page_list_input(value) if page > 0})


def _normalize_ranked_page_list(value):
    pages = []
    for page in _normalize_page_list_input(value):
        if page > 0 and page not in pages:
            pages.append(page)
    return pages


def _normalize_tree_locations(payload, valid_pages=None, location_limit=3, pages_per_location=3):
    valid_pages = set(valid_pages) if valid_pages is not None else None
    location_limit = max(0, int(location_limit))
    pages_per_location = max(1, int(pages_per_location))
    if not isinstance(payload, dict):
        return []
    if location_limit <= 0:
        return []

    def normalize_pages(value):
        pages = []
        for page in _normalize_ranked_page_list(value):
            if valid_pages is not None and page not in valid_pages:
                continue
            pages.append(page)
            if len(pages) >= pages_per_location:
                break
        return pages

    locations = []
    raw_locations = payload.get("locations")
    if isinstance(raw_locations, list):
        for item in raw_locations:
            if isinstance(item, dict):
                pages = normalize_pages(item.get("pages"))
                reason = str(item.get("reason", "")).strip()
                title = str(item.get("title", item.get("section_title", ""))).strip()
            else:
                pages = normalize_pages(item)
                reason = ""
                title = ""
            if not pages:
                continue
            locations.append(
                {
                    "location_id": f"tree_location_{len(locations) + 1}",
                    "pages": pages,
                    "reason": reason,
                    "title": title,
                }
            )
            if len(locations) >= location_limit:
                break

    if locations:
        return locations

    # Backward compatibility for older locator prompts/tests that return
    # {"pages": [...]} without explicit tree locations.
    for page in normalize_pages(payload.get("pages"))[:location_limit]:
        locations.append(
            {
                "location_id": f"tree_location_{len(locations) + 1}",
                "pages": [page],
                "reason": str(payload.get("reason", "")).strip(),
                "title": "",
            }
        )
    return locations


def _normalize_field_result(field_name, payload, default_pages=None, allowed_pages=None):
    default_pages = default_pages or []
    allowed_pages = set(allowed_pages) if allowed_pages is not None else None
    if not isinstance(payload, dict):
        raise ValueError("field result must be a JSON object")

    status = str(payload.get("status", "")).strip().lower()
    if status not in {item.value for item in ExtractionStatus}:
        raise ValueError(f"invalid status for field {field_name}: {status!r}")

    if status == ExtractionStatus.REFERENCE_FOUND.value:
        references = payload.get("references")
        if not isinstance(references, list) or not references:
            raise ValueError(f"field {field_name} must include references when status='reference_found'")
        normalized_references = []
        for reference in references:
            if not isinstance(reference, dict):
                raise ValueError(f"field {field_name} reference entries must be JSON objects")
            normalized_reference = {
                "reference_type": str(reference.get("reference_type", "")).strip(),
                "target_text": str(reference.get("target_text", "")).strip(),
                "relation": str(reference.get("relation", "")).strip(),
            }
            if not all(normalized_reference.values()):
                raise ValueError(
                    f"field {field_name} references must include reference_type, target_text, and relation"
                )
            normalized_references.append(normalized_reference)
        evidence = str(payload.get("evidence", "")).strip()
        if not evidence:
            raise ValueError(f"field {field_name} must include evidence when status='reference_found'")
        confidence = str(payload.get("confidence", "")).strip()
        if confidence not in {item.value for item in ConfidenceLevel}:
            raise ValueError(f"invalid confidence for field {field_name}: {confidence!r}")
        pages = _normalize_page_list(payload.get("pages")) or list(default_pages)
        if allowed_pages is not None and not set(pages).issubset(allowed_pages):
            raise ValueError(f"field {field_name} referenced evidence pages that were not loaded")
        return {
            "status": status,
            "value": "",
            "evidence": evidence,
            "pages": pages,
            "confidence": confidence,
            "reason": str(payload.get("reason", "")).strip() or "current pages contain a reference only",
            "references": normalized_references,
        }

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
    pages = _normalize_page_list(payload.get("pages")) or list(default_pages)
    if allowed_pages is not None and not set(pages).issubset(allowed_pages):
        raise ValueError(f"field {field_name} referenced evidence pages that were not loaded")

    return {
        "status": ExtractionStatus.FOUND.value,
        "value": value,
        "evidence": evidence,
        "pages": pages,
        "confidence": confidence,
        "reason": str(payload.get("reason", "")).strip() or None,
    }


def _build_locator_prompt(field, structure_digest, location_limit=3, pages_per_location=3):
    return f"""
你正在为一项合同字段抽取任务定位最相关的页面。

任务字段：
- description: {field.description}
- type: {field.type}
- required: {field.required}
- instruction: {field.instruction or "无"}

文档结构摘要：
{structure_digest}

规则：
- 只能使用上面的结构摘要进行判断。
- 最多返回 {location_limit} 个候选位置；一个章节或一个连续页段算一个位置。
- 每个候选位置最多返回 {pages_per_location} 页，只选择该位置内最可能包含答案的最小页集合。
- 如果结构摘要同时包含 start_page/end_page 和 content_start_page/content_end_page，优先使用 content 页码范围定位该节点摘要对应的正文。
- 不要因为章节范围是 start_page 到 end_page 就机械展开全部页面。
- 优先选择标题或摘要中直接提到目标字段的位置。
- 如果没有强相关候选位置，返回空的 locations 列表。

只返回 JSON，格式如下：
{{
  "locations": [
    {{"title": "章节或位置标题", "pages": [4, 5], "reason": "说明这个位置为什么可能相关"}}
  ],
  "reason": "整体定位说明"
}}
""".strip()


def _build_extraction_prompt(field, extraction_context_json):
    return f"""
请从提供的页面文本中准确抽取一个合同字段。

字段定义：
- description: {field.description}
- type: {field.type}
- required: {field.required}
- instruction: {field.instruction or "无"}

候选池与页面内容：
{extraction_context_json}

输出规则：
- 只返回 JSON。
- status 必须严格是以下之一："found"、"reference_found"、"not_found"、"error"
- confidence 必须严格是以下之一："High"、"Medium"、"Low"
- 当前页直接包含字段值和支持证据时，返回 "found"。
- 当前页没有最终值、但明确要求查看其他条款、附件或协议时，返回 "reference_found"，并提供 references。
- 如果当前提供的多页中同时包含引用页和最终值页，应直接返回 "found"。
- 如果 status 为 "found"：
  - value 必须是命中字段所在的完整合同条款原文，不要总结、改写或只返回字段值。
  - 如果命中内容位于某一编号条款、标题条款或条款项下的局部段落，必须扩展返回该条完整条款；如果多个条款共同支持结果，value 中逐条返回这些完整条款，并用换行分隔。
  - evidence 必须是支撑判断的核心原文片段，可使用省略号压缩上下文，例如“合同总价……人民币500万元……”，不要把完整条款全文放入 evidence。
  - pages 必须是物理页码数组
  - confidence 判定标准：
    - High：文本中明确写出了该条款内容，且 evidence 能直接支持结论
    - Medium：该值需要结合附近上下文或轻度推断得到
    - Low：证据间接、含糊，或支持力度较弱
- 如果 status 为 "not_found" 或 "error"：
  - value 设为 ""
  - evidence 设为 ""
  - confidence 设为 "Low"
  - 必须提供非空的 reason
- 如果 status 为 "reference_found"：
  - value 设为 ""
  - evidence 必须复制明确引用的核心原文片段，可使用省略号压缩上下文
  - pages 必须是引用入口的物理页码数组
  - references 必须为数组，每项包含 reference_type、target_text、relation

返回 JSON，格式如下：
{{
  "status": "found",
  "value": "第X条 完整合同条款原文……",
  "evidence": "核心原文……可省略中间内容……",
  "pages": [4],
  "confidence": "High",
  "reason": null
}}
""".strip()


def _build_long_context_extraction_prompt(field, pages_json):
    return f"""
请从完整文档的分页原文中准确抽取一个合同字段。

字段定义：
- description: {field.description}
- type: {field.type}
- required: {field.required}
- instruction: {field.instruction or "无"}

完整文档分页原文：
{pages_json}

输出规则：
- 只返回 JSON。
- status 必须严格是以下之一："found"、"not_found"、"error"。
- confidence 必须严格是以下之一："High"、"Medium"、"Low"。
- 如果 status 为 "found"，value 必须是命中字段所在的完整合同条款原文，不要总结、改写或只返回字段值。
- 如果命中内容位于某一编号条款、标题条款或条款项下的局部段落，必须扩展返回该条完整条款；如果多个条款共同支持结果，value 中逐条返回这些完整条款，并用换行分隔。
- evidence 必须是支撑判断的核心原文片段，可使用省略号压缩上下文，例如“合同总价……人民币500万元……”，不要把完整条款全文放入 evidence。
- pages 必须是支撑条款所在的物理页码数组。
- 如果 status 为 "not_found" 或 "error"，value 和 evidence 设为 ""，confidence 设为 "Low"，并提供非空 reason。
- 不要返回未出现在完整文档分页原文中的页码。

返回 JSON，格式如下：
{{
  "status": "found",
  "value": "第X条 完整合同条款原文……",
  "evidence": "核心原文……可省略中间内容……",
  "pages": [4],
  "confidence": "High",
  "reason": null
}}
""".strip()


def _build_query_generation_prompt(field, primary_limit, keyword_limit):
    return f"""
请为合同字段生成页级检索规格，用于 BM25 召回。

字段：
- description: {field.description}
- type: {field.type}
- instruction: {field.instruction or "无"}

规则：
- primary_queries 只提炼字段核心含义，最多 {primary_limit} 个。
- expanded_keywords 只用于补召回，最多 {keyword_limit} 个，不要重复 primary_queries。
- 只返回 JSON。

格式：
{{
  "primary_queries": ["主查询"],
  "expanded_keywords": ["扩展关键词"]
}}
""".strip()


def _build_reference_resolution_prompt(reference_text, target_text, structure_digest, candidate_manifest, processed_pages):
    return f"""
请根据合同引用内容定位可能的目标物理页。

引用原文：{reference_text}
引用目标：{target_text}
已读取页面：{processed_pages}
候选页清单：{json.dumps(candidate_manifest, ensure_ascii=False)}
文档结构摘要：
{structure_digest}

规则：
- 只能选择文档结构摘要中已有页码范围内的页面。
- 根据引用信息和文档树判断目标，不要仅因某一页包含引用字样就选择该页。
- 最多返回 3 个页面；无法可靠定位时返回空数组。
- 只返回 JSON：{{"pages": [1, 2], "reason": "..."}}。
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


def _setting(client, name, default):
    return getattr(getattr(client, "retrieval_config", None), name, default)


def _tree_location_limit(client):
    config = getattr(client, "retrieval_config", None)
    if config is not None and hasattr(config, "tree_location_limit"):
        return getattr(config, "tree_location_limit")
    return _setting(client, "tree_page_limit", 3)


def _query_cache_path(client):
    workspace = getattr(client, "workspace", None)
    if workspace is None:
        return None
    return Path(workspace) / "_retrieval" / "query_specs.json"


def _load_query_cache(client):
    if not _setting(client, "retrieval_query_cache_enabled", True):
        return {}
    path = _query_cache_path(client)
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_query_cache(client, cache):
    if not _setting(client, "retrieval_query_cache_enabled", True):
        return
    path = _query_cache_path(client)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_path, path)


def _query_cache_key(client, field):
    payload = {
        "version": QUERY_SPEC_PROMPT_VERSION,
        "model": getattr(client, "retrieve_model", ""),
        "name": field.name,
        "description": field.description,
        "type": field.type,
        "instruction": field.instruction,
        "primary_query_limit": _setting(client, "retrieval_primary_query_limit", 3),
        "expanded_keyword_limit": _setting(client, "retrieval_expanded_keyword_limit", 8),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _string_list(value, limit):
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        item = str(item or "").strip()
        if item and item not in result:
            result.append(item)
        if len(result) >= limit:
            break
    return result


async def _generate_query_spec(client, field, query_cache, retries, timeout_seconds):
    default_spec = {"primary_queries": [field.description], "expanded_keywords": []}
    if not _setting(client, "retrieval_query_generation_enabled", True):
        return {**default_spec, "generation_source": "description_fallback"}
    cache_enabled = _setting(client, "retrieval_query_cache_enabled", True)
    primary_limit = _setting(client, "retrieval_primary_query_limit", 3)
    keyword_limit = _setting(client, "retrieval_expanded_keyword_limit", 8)
    key = _query_cache_key(client, field)
    if cache_enabled and isinstance(query_cache.get(key), dict):
        logger.debug("Retrieval query cache hit for field=%s", field.name)
        cached = query_cache[key]
        return {
            "primary_queries": _string_list(cached.get("primary_queries"), primary_limit) or [field.description],
            "expanded_keywords": _string_list(cached.get("expanded_keywords"), keyword_limit),
            "generation_source": "cache_hit",
        }
    try:
        payload = await _run_json_prompt(
            client.retrieve_model,
            _build_query_generation_prompt(field, primary_limit, keyword_limit),
            retries=retries,
            timeout_seconds=timeout_seconds,
        )
        primary_queries = _string_list(payload.get("primary_queries"), primary_limit) or [field.description]
        expanded_keywords = _string_list(payload.get("expanded_keywords"), keyword_limit)
        result = {"primary_queries": primary_queries, "expanded_keywords": expanded_keywords}
        generation_source = "llm_generated"
    except Exception:
        result = default_spec
        generation_source = "description_fallback"
        logger.debug("Retrieval query generation fell back to description for field=%s", field.name)
    if cache_enabled:
        query_cache[key] = result
    return {**result, "generation_source": generation_source}


def _make_candidate(
    page,
    source,
    page_documents_by_page,
    reason="",
    score=0.0,
    reference_chain=None,
    reference_depth=0,
    tree_location_id="",
    tree_location_title="",
):
    document = page_documents_by_page.get(page, {})
    return {
        "page": page,
        "candidate_id": f"p{page}",
        "content_unit": str(page),
        "sources": [source],
        "section_path": document.get("section_path", ""),
        "bm25_score": score,
        "reason": reason,
        "read_status": "pending",
        "is_chunk": False,
        "reference_chain": list(reference_chain or []),
        "reference_depth": reference_depth,
        "tree_location_id": tree_location_id,
        "tree_location_title": tree_location_title,
    }


def _count_context_tokens(client, content):
    try:
        return count_tokens(content, model=getattr(client, "retrieve_model", None))
    except Exception:
        return max(1, len(content) // 2)


def _ensure_candidate_metadata(candidate):
    page = candidate["page"]
    candidate.setdefault("candidate_id", f"p{page}")
    candidate.setdefault("content_unit", str(page))
    candidate.setdefault("read_status", "pending")
    candidate.setdefault("is_chunk", False)
    candidate.setdefault("reference_chain", [])
    candidate.setdefault("reference_depth", 0)
    candidate.setdefault("evaluation_history", [])
    candidate.setdefault("tree_location_id", "")
    candidate.setdefault("tree_location_title", "")
    return candidate


def _candidate_content(candidate, page_documents_by_page):
    return candidate.get("content_override", page_documents_by_page[candidate["page"]].get("content", ""))


def _split_extraction_content(client, content, token_budget):
    if _count_context_tokens(client, content) <= token_budget:
        return [content]

    chunks = []
    remaining = content
    overlap_budget = min(50, token_budget // 6)
    while remaining:
        if _count_context_tokens(client, remaining) <= token_budget:
            chunks.append(remaining)
            break
        low, high = 1, len(remaining)
        fit = 0
        while low <= high:
            middle = (low + high) // 2
            if _count_context_tokens(client, remaining[:middle]) <= token_budget:
                fit = middle
                low = middle + 1
            else:
                high = middle - 1
        if fit <= 0:
            return []
        boundary = remaining.rfind("\n", 0, fit + 1)
        if boundary >= fit // 2:
            fit = boundary + 1
        chunk = remaining[:fit]
        chunks.append(chunk)
        overlap = ""
        if overlap_budget and fit < len(remaining):
            low, high = 1, len(chunk)
            while low <= high:
                middle = (low + high) // 2
                if _count_context_tokens(client, chunk[-middle:]) <= overlap_budget:
                    overlap = chunk[-middle:]
                    low = middle + 1
                else:
                    high = middle - 1
        remaining = overlap + remaining[fit:]
    return chunks


def _expand_oversized_candidates(client, candidate_pool, page_documents_by_page):
    token_budget = max(1, int(_setting(client, "initial_context_token_budget", 12000)))
    expanded_pool = []
    expanded_pages = []
    for candidate in candidate_pool:
        candidate = _ensure_candidate_metadata(candidate)
        content = _candidate_content(candidate, page_documents_by_page)
        if candidate.get("is_chunk") or _count_context_tokens(client, content) <= token_budget:
            expanded_pool.append(candidate)
            continue
        chunks = _split_extraction_content(client, content, token_budget)
        if not chunks:
            candidate["read_status"] = "unreadable_over_budget"
            expanded_pool.append(candidate)
            continue
        expanded_pages.append(
            {
                "page": candidate["page"],
                "content_units": [f"{candidate['page']}-{index}" for index in range(1, len(chunks) + 1)],
            }
        )
        for index, chunk in enumerate(chunks, start=1):
            chunk_candidate = dict(candidate)
            chunk_candidate.update(
                {
                    "candidate_id": f"p{candidate['page']}-c{index}",
                    "content_unit": f"{candidate['page']}-{index}",
                    "content_override": chunk,
                    "chunk_index": index,
                    "chunk_count": len(chunks),
                    "is_chunk": True,
                    "standalone_batch": True,
                    "read_status": "pending",
                }
            )
            expanded_pool.append(chunk_candidate)
    return expanded_pool, expanded_pages


def _requeue_candidate(candidate):
    previous_status = candidate.get("read_status")
    if previous_status not in {"pending", "loaded"}:
        candidate.setdefault("evaluation_history", []).append(previous_status)
    candidate["read_status"] = "pending"


def _merge_candidate(candidate_pool, candidate):
    candidate = _ensure_candidate_metadata(candidate)
    chunked_page_candidates = [
        existing for existing in candidate_pool if existing["page"] == candidate["page"] and existing.get("is_chunk")
    ]
    if chunked_page_candidates and not candidate.get("is_chunk"):
        for existing in chunked_page_candidates:
            for source in candidate.get("sources", []):
                if source not in existing["sources"]:
                    existing["sources"].append(source)
            existing["bm25_score"] = max(existing.get("bm25_score", 0.0), candidate.get("bm25_score", 0.0))
        return chunked_page_candidates[0]
    for existing in candidate_pool:
        existing = _ensure_candidate_metadata(existing)
        if existing["candidate_id"] != candidate["candidate_id"]:
            continue
        for source in candidate.get("sources", []):
            if source not in existing["sources"]:
                existing["sources"].append(source)
        existing["bm25_score"] = max(existing.get("bm25_score", 0.0), candidate.get("bm25_score", 0.0))
        if candidate.get("reason") and candidate["reason"] not in existing.get("reason", ""):
            existing["reason"] = "；".join(filter(None, [existing.get("reason", ""), candidate["reason"]]))
        candidate_chain = candidate.get("reference_chain") or []
        if candidate_chain and (
            not existing.get("reference_chain")
            or candidate.get("reference_depth", 0) < existing.get("reference_depth", 0)
        ):
            existing["reference_chain"] = list(candidate_chain)
            existing["reference_depth"] = candidate.get("reference_depth", 0)
        if candidate.get("tree_location_id") and not existing.get("tree_location_id"):
            existing["tree_location_id"] = candidate["tree_location_id"]
            existing["tree_location_title"] = candidate.get("tree_location_title", "")
        return existing
    candidate_pool.append(candidate)
    return candidate


def _select_tree_candidates(tree_locations, page_documents_by_page):
    candidate_pool = []
    for location in tree_locations:
        for page in location["pages"]:
            reason = location.get("reason") or "摘要树定位"
            if location.get("title"):
                reason = "；".join([location["title"], reason])
            _merge_candidate(
                candidate_pool,
                _make_candidate(
                    page,
                    "tree",
                    page_documents_by_page,
                    reason=reason,
                    tree_location_id=location["location_id"],
                    tree_location_title=location.get("title", ""),
                ),
            )
    return candidate_pool


def _select_initial_candidates(client, tree_pages, bm25_candidates, page_documents_by_page):
    candidate_pool = []
    page_limit = _setting(client, "initial_candidate_page_limit", 9)

    ordered = [
        _make_candidate(page, "tree", page_documents_by_page, reason="摘要树定位")
        for page in tree_pages[: _tree_location_limit(client)]
    ]
    ordered.extend(bm25_candidates[: _setting(client, "bm25_selected_page_limit", 6)])
    for candidate in ordered:
        existing = next((item for item in candidate_pool if item["page"] == candidate["page"]), None)
        if existing is not None:
            _merge_candidate(candidate_pool, candidate)
            continue
        if len(candidate_pool) >= page_limit:
            break
        _merge_candidate(candidate_pool, candidate)
    return candidate_pool


def _select_extraction_batch(client, pending, page_documents_by_page, reference_context_pages):
    page_limit = max(1, int(_setting(client, "extract_batch_page_limit", 4)))
    token_budget = max(1, int(_setting(client, "initial_context_token_budget", 12000)))
    context_pages = [
        page for page in reference_context_pages if page in page_documents_by_page
    ][:page_limit]
    context_tokens = sum(
        _count_context_tokens(client, page_documents_by_page[page].get("content", "")) for page in context_pages
    )
    if context_tokens > token_budget or len(context_pages) >= page_limit:
        context_pages = []
        context_tokens = 0

    selected = []
    used_tokens = context_tokens
    for candidate in pending:
        candidate = _ensure_candidate_metadata(candidate)
        page = candidate["page"]
        page_tokens = _count_context_tokens(client, _candidate_content(candidate, page_documents_by_page))
        if page_tokens > token_budget:
            candidate["read_status"] = "unreadable_over_budget"
            logger.info("Unable to split oversized extraction unit field_page=%s tokens=%s budget=%s", page, page_tokens, token_budget)
            continue
        if candidate.get("standalone_batch"):
            if selected:
                break
            return [candidate], []
        if not selected and context_pages and used_tokens + page_tokens > token_budget:
            context_pages = []
            used_tokens = 0
        if len(context_pages) + len(selected) >= page_limit or used_tokens + page_tokens > token_budget:
            break
        selected.append(candidate)
        used_tokens += page_tokens
    return selected, context_pages


def _candidate_manifest(candidate_pool):
    return [
        {
            "page": candidate["page"],
            "content_unit": candidate.get("content_unit", str(candidate["page"])),
            "candidate_id": candidate.get("candidate_id", f"p{candidate['page']}"),
            "sources": candidate["sources"],
            "section_path": candidate.get("section_path", ""),
            "tree_location_id": candidate.get("tree_location_id", ""),
            "tree_location_title": candidate.get("tree_location_title", ""),
            "read_status": candidate["read_status"],
            "chunk_note": (
                f"物理页 {candidate['page']} 的第 {candidate['chunk_index']}/{candidate['chunk_count']} 个正文分块"
                if candidate.get("is_chunk")
                else ""
            ),
            "prior_evaluations": candidate.get("evaluation_history", []),
        }
        for candidate in candidate_pool
    ]


def _metadata_result(result, resolution_method, retrieval_sources, evidence_chain, fallback_status=None):
    enriched = dict(result)
    enriched["resolution_method"] = resolution_method
    enriched["retrieval_sources"] = retrieval_sources
    if evidence_chain:
        enriched["evidence_chain"] = evidence_chain
    if fallback_status:
        enriched["fallback_status"] = fallback_status
    return enriched


def _reference_state_for_pages(candidate_pool, pages, preferred_candidates=None):
    search_order = [*(preferred_candidates or []), *candidate_pool]
    for page in pages:
        candidate = next((item for item in search_order if item["page"] == page), None)
        if candidate is not None:
            return list(candidate.get("reference_chain", [])), int(candidate.get("reference_depth", 0))
    return [], 0


def _has_pending_tree_work(candidate_pool):
    return any(
        "tree" in candidate.get("sources", []) and candidate.get("read_status") == "pending"
        for candidate in candidate_pool
    )


def _candidate_pages(candidate_pool):
    return {candidate["page"] for candidate in candidate_pool}


def _exclude_candidate_pages(candidates, excluded_pages):
    excluded_pages = set(excluded_pages)
    return [candidate for candidate in candidates if candidate["page"] not in excluded_pages]


def _emit_retrieval_log(retrieval_logger, event, field_name, **details):
    if retrieval_logger is not None:
        retrieval_logger.info({"event": event, "field": field_name, **details})


async def _extract_payload(client, field, extraction_context, retries, timeout_seconds):
    last_error = None
    for _ in range(retries + 1):
        payload = await _run_json_prompt(
            client.retrieve_model,
            _build_extraction_prompt(field, json.dumps(extraction_context, ensure_ascii=False, indent=2)),
            retries=0,
            timeout_seconds=timeout_seconds,
        )
        try:
            return _normalize_field_result(
                field.name,
                payload,
                default_pages=extraction_context["current_loaded_pages"],
                allowed_pages=extraction_context["current_loaded_pages"],
            )
        except ValueError as exc:
            last_error = exc
    raise last_error or ValueError("field extraction validation failed")


async def _extract_one_field_legacy(client, doc_id, field, structure_digest, semaphore, retries=1, timeout_seconds=45):
    async with semaphore:
        try:
            locator_payload = await _run_json_prompt(
                client.retrieve_model,
                _build_locator_prompt(
                    field,
                    structure_digest,
                    _tree_location_limit(client),
                    _setting(client, "tree_location_page_limit", 3),
                ),
                retries=retries,
                timeout_seconds=timeout_seconds,
            )
            tree_locations = _normalize_tree_locations(
                locator_payload,
                location_limit=_tree_location_limit(client),
                pages_per_location=_setting(client, "tree_location_page_limit", 3),
            )
            pages = []
            for location in tree_locations:
                for page in location["pages"]:
                    if page not in pages:
                        pages.append(page)
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
                    return field.name, _normalize_field_result(
                        field.name,
                        extraction_payload,
                        default_pages=pages,
                        allowed_pages=pages,
                    )
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


async def _extract_one_field_long_context(client, field, pages, semaphore, retries=1, timeout_seconds=45):
    async with semaphore:
        try:
            allowed_pages = [page["page"] for page in pages]
            long_context_model = getattr(client, "long_context_model", None) or client.retrieve_model
            last_error = None
            for _ in range(retries + 1):
                payload = await _run_json_prompt(
                    long_context_model,
                    _build_long_context_extraction_prompt(field, json.dumps(pages, ensure_ascii=False, indent=2)),
                    retries=0,
                    timeout_seconds=timeout_seconds,
                )
                try:
                    result = _normalize_field_result(field.name, payload, allowed_pages=allowed_pages)
                    return field.name, _metadata_result(result, "long_context", ["full_document"], [])
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


async def _extract_one_field_enhanced(
    client,
    doc_id,
    field,
    structure,
    structure_digest,
    retrieval_payload,
    page_documents,
    query_cache,
    vector_lock,
    semaphore,
    retrieval_logger=None,
    retries=1,
    timeout_seconds=45,
):
    async with semaphore:
        try:
            page_documents_by_page = {document["page"]: document for document in page_documents}
            valid_pages = set(page_documents_by_page)
            structure_pages = set()
            for row in _flatten_structure(structure):
                start_page = row.get("start_page")
                end_page = row.get("end_page")
                if isinstance(start_page, int) and isinstance(end_page, int):
                    structure_pages.update(range(start_page, end_page + 1))
            structure_pages &= valid_pages
            query_spec = None
            bm25_candidates = []
            bm25_candidates_loaded = False

            async def ensure_query_spec():
                nonlocal query_spec
                if query_spec is None:
                    query_spec = await _generate_query_spec(client, field, query_cache, retries, timeout_seconds)
                    _emit_retrieval_log(
                        retrieval_logger,
                        "retrieval_query_spec_resolved",
                        field.name,
                        generation_source=query_spec["generation_source"],
                        primary_queries=query_spec["primary_queries"],
                        expanded_keywords=query_spec["expanded_keywords"],
                    )
                return query_spec

            def load_bm25_candidates(spec):
                return retrieve_bm25_candidates(
                    page_documents,
                    spec["primary_queries"],
                    spec["expanded_keywords"],
                    primary_top_k=_setting(client, "bm25_primary_top_k", 10),
                    expanded_top_k=_setting(client, "bm25_expanded_top_k", 20),
                    primary_weight=_setting(client, "bm25_primary_rrf_weight", 3.0),
                    expanded_weight=_setting(client, "bm25_expanded_rrf_weight", 1.0),
                    rrf_constant=_setting(client, "bm25_rrf_constant", 60),
                )

            try:
                tree_location_limit = _tree_location_limit(client)
                tree_location_page_limit = _setting(client, "tree_location_page_limit", 3)
                locator_payload = await _run_json_prompt(
                    client.retrieve_model,
                    _build_locator_prompt(
                        field,
                        structure_digest,
                        tree_location_limit,
                        tree_location_page_limit,
                    ),
                    retries=retries,
                    timeout_seconds=timeout_seconds,
                )
                tree_locations = _normalize_tree_locations(
                    locator_payload,
                    valid_pages,
                    location_limit=tree_location_limit,
                    pages_per_location=tree_location_page_limit,
                )
            except Exception:
                tree_locations = []
            tree_pages = []
            for location in tree_locations:
                for page in location["pages"]:
                    if page not in tree_pages:
                        tree_pages.append(page)

            if tree_locations:
                candidate_pool = _select_tree_candidates(tree_locations, page_documents_by_page)
            else:
                query_spec = await ensure_query_spec()
                bm25_candidates = load_bm25_candidates(query_spec)
                bm25_candidates_loaded = True
                candidate_pool = _select_initial_candidates(client, [], bm25_candidates, page_documents_by_page)
            logger.debug(
                "Initial retrieval field=%s tree_pages=%s bm25_pages=%s selected_pages=%s",
                field.name,
                tree_pages,
                [candidate["page"] for candidate in bm25_candidates],
                [candidate["page"] for candidate in candidate_pool],
            )
            _emit_retrieval_log(
                retrieval_logger,
                "retrieval_candidates_selected",
                field.name,
                tree_pages=tree_pages,
                tree_locations=tree_locations,
                bm25_candidates=[
                    {
                        "page": candidate["page"],
                        "sources": candidate["sources"],
                        "section_path": candidate.get("section_path", ""),
                        "bm25_score": candidate.get("bm25_score", 0.0),
                        "bm25_ranks": candidate.get("bm25_ranks", {}),
                        "bm25_raw_scores": candidate.get("bm25_raw_scores", {}),
                    }
                    for candidate in bm25_candidates
                ],
                candidate_pages=[candidate["page"] for candidate in candidate_pool],
            )
            page_content_cache: dict[int, str] = {}
            processed_pages: set[int] = set()
            followed_reference_keys: set[str] = set()
            reference_target_pages: set[int] = set()
            reference_context_pages: list[int] = []
            unresolved_targets: list[str] = []
            logged_chunked_pages: set[int] = set()
            logged_unreadable_units: set[str] = set()
            vector_fallback_used = False
            deferred_found_result = None
            deferred_found_event = None
            attempt_index = 0
            last_reason = "no candidate pages yielded a supported answer"

            while True:
                candidate_pool, expanded_pages = _expand_oversized_candidates(
                    client,
                    candidate_pool,
                    page_documents_by_page,
                )
                new_expanded_pages = [
                    item for item in expanded_pages if item["page"] not in logged_chunked_pages
                ]
                if new_expanded_pages:
                    logged_chunked_pages.update(item["page"] for item in new_expanded_pages)
                    _emit_retrieval_log(
                        retrieval_logger,
                        "candidate_pages_chunked_for_extraction",
                        field.name,
                        pages=new_expanded_pages,
                    )
                pending = [candidate for candidate in candidate_pool if candidate["read_status"] == "pending"]
                if not pending:
                    if deferred_found_result is not None:
                        if deferred_found_event is not None:
                            _emit_retrieval_log(
                                retrieval_logger,
                                "field_extraction_completed",
                                field.name,
                                **deferred_found_event,
                            )
                        return deferred_found_result
                    if not bm25_candidates_loaded:
                        query_spec = await ensure_query_spec()
                        bm25_candidates = load_bm25_candidates(query_spec)
                        bm25_candidates_loaded = True
                        excluded_pages = _candidate_pages(candidate_pool) | processed_pages
                        new_bm25_candidates = _exclude_candidate_pages(
                            bm25_candidates,
                            excluded_pages,
                        )
                        for candidate in _select_initial_candidates(
                            client,
                            [],
                            new_bm25_candidates,
                            page_documents_by_page,
                        ):
                            _merge_candidate(candidate_pool, candidate)
                        _emit_retrieval_log(
                            retrieval_logger,
                            "bm25_fallback_candidates_selected",
                            field.name,
                            bm25_candidates=[
                                {
                                    "page": candidate["page"],
                                    "sources": candidate["sources"],
                                    "section_path": candidate.get("section_path", ""),
                                    "bm25_score": candidate.get("bm25_score", 0.0),
                                    "bm25_ranks": candidate.get("bm25_ranks", {}),
                                    "bm25_raw_scores": candidate.get("bm25_raw_scores", {}),
                                }
                                for candidate in bm25_candidates
                            ],
                            excluded_pages=sorted(excluded_pages),
                            candidate_pages=[candidate["page"] for candidate in candidate_pool],
                        )
                        if any(candidate["read_status"] == "pending" for candidate in candidate_pool):
                            continue
                    vector_allowed = (
                        _setting(client, "vector_enabled", True)
                        and not vector_fallback_used
                        and _setting(client, "vector_fallback_max_runs", 1) > 0
                    )
                    if vector_allowed:
                        vector_fallback_used = True
                        vector_query = "\n".join(
                            [
                                field.description,
                                field.instruction,
                                *query_spec["primary_queries"],
                                *query_spec["expanded_keywords"],
                                *unresolved_targets,
                            ]
                        )
                        try:
                            async with vector_lock:
                                vector_candidates = await retrieve_vector_candidates(
                                    doc_id=doc_id,
                                    source_sha256=retrieval_payload.get("source_sha256", ""),
                                    page_documents=page_documents,
                                    query_text=vector_query,
                                    processed_pages=set(processed_pages) | _candidate_pages(candidate_pool),
                                    workspace=getattr(client, "workspace", None),
                                    embedding_model=_setting(client, "embedding_model", "openai/bge-m3:latest"),
                                    chunk_tokens=_setting(client, "vector_chunk_tokens", 300),
                                    overlap_tokens=_setting(client, "vector_chunk_overlap_tokens", 50),
                                    chunk_top_k=_setting(client, "vector_chunk_top_k", 30),
                                    page_limit=_setting(client, "vector_page_limit", 5),
                                    context_token_budget=_setting(client, "vector_context_token_budget", 10000),
                                    embedding_batch_size=_setting(client, "embedding_batch_size", 64),
                                    embedding_request_token_budget=_setting(
                                        client,
                                        "embedding_request_token_budget",
                                        8192,
                                    ),
                                    cache_enabled=_setting(client, "vector_index_cache_enabled", True),
                                )
                            logger.debug(
                                "Vector fallback field=%s selected_pages=%s",
                                field.name,
                                [candidate["page"] for candidate in vector_candidates],
                            )
                            _emit_retrieval_log(
                                retrieval_logger,
                                "vector_fallback_completed",
                                field.name,
                                candidate_pages=[candidate["page"] for candidate in vector_candidates],
                            )
                            for candidate in vector_candidates:
                                _merge_candidate(candidate_pool, candidate)
                            if vector_candidates:
                                continue
                        except Exception as exc:
                            logger.info("Vector fallback unavailable for %s: %s", field.name, exc)
                            _emit_retrieval_log(
                                retrieval_logger,
                                "vector_fallback_unavailable",
                                field.name,
                                error_type=type(exc).__name__,
                            )
                            return field.name, _metadata_result(
                                {
                                    "status": ExtractionStatus.ERROR.value,
                                    "value": "",
                                    "evidence": "",
                                    "pages": [],
                                    "confidence": ConfidenceLevel.LOW.value,
                                    "reason": "vector fallback is enabled but unavailable",
                                },
                                "error",
                                [],
                                [],
                                fallback_status="unavailable",
                            )
                    _emit_retrieval_log(
                        retrieval_logger,
                        "field_extraction_completed",
                        field.name,
                        status=ExtractionStatus.NOT_FOUND.value,
                        resolution_method="not_found",
                    )
                    return field.name, _metadata_result(
                        {
                            "status": ExtractionStatus.NOT_FOUND.value,
                            "value": "",
                            "evidence": "",
                            "pages": [],
                            "confidence": ConfidenceLevel.LOW.value,
                            "reason": last_reason,
                        },
                        "not_found",
                        [],
                        [],
                    )

                batch, batch_reference_context_pages = _select_extraction_batch(
                    client,
                    pending,
                    page_documents_by_page,
                    reference_context_pages,
                )
                unreadable_units = [
                    candidate["content_unit"]
                    for candidate in candidate_pool
                    if candidate["read_status"] == "unreadable_over_budget"
                    and candidate["content_unit"] not in logged_unreadable_units
                ]
                if unreadable_units:
                    logged_unreadable_units.update(unreadable_units)
                    _emit_retrieval_log(
                        retrieval_logger,
                        "candidate_units_unreadable_over_budget",
                        field.name,
                        content_units=unreadable_units,
                    )
                if not batch:
                    if any(candidate["read_status"] == "unreadable_over_budget" for candidate in pending):
                        last_reason = "candidate pages exceeded extraction context token budget"
                    continue
                logger.debug(
                    "Extraction batch field=%s content_units=%s",
                    field.name,
                    [candidate["content_unit"] for candidate in batch],
                )
                _emit_retrieval_log(
                    retrieval_logger,
                    "extraction_batch_loaded",
                    field.name,
                    pages=[candidate["page"] for candidate in batch],
                    content_units=[candidate["content_unit"] for candidate in batch],
                )
                for candidate in batch:
                    candidate["read_status"] = "loaded"
                    processed_pages.add(candidate["page"])
                    page_content_cache[candidate["page"]] = _candidate_content(candidate, page_documents_by_page)
                loaded_pages = []
                for page in [*batch_reference_context_pages, *[candidate["page"] for candidate in batch]]:
                    if page not in loaded_pages and page in page_documents_by_page:
                        loaded_pages.append(page)
                        page_content_cache.setdefault(page, page_documents_by_page[page]["content"])
                current_page_content = [
                    {"page": page, "content_unit": str(page), "content": page_content_cache[page]}
                    for page in batch_reference_context_pages
                    if page in loaded_pages
                ]
                current_page_content.extend(
                    {
                        "page": candidate["page"],
                        "content_unit": candidate["content_unit"],
                        "content": _candidate_content(candidate, page_documents_by_page),
                    }
                    for candidate in batch
                )
                extraction_context = {
                    "candidate_page_manifest": _candidate_manifest(candidate_pool),
                    "current_loaded_pages": loaded_pages,
                    "previously_read_pages": sorted(processed_pages - set(loaded_pages)),
                    "current_page_content": current_page_content,
                }
                try:
                    attempt_index += 1
                    result = await _extract_payload(client, field, extraction_context, retries, timeout_seconds)
                except Exception as exc:
                    if deferred_found_result is not None:
                        last_reason = str(exc)
                        for candidate in batch:
                            candidate["read_status"] = "evaluation_error"
                        continue
                    raise
                if result["status"] == ExtractionStatus.FOUND.value:
                    for candidate in batch:
                        candidate["read_status"] = "evaluated_found"
                    supporting_pages = result["pages"] or loaded_pages
                    evidence_chain, _ = _reference_state_for_pages(candidate_pool, supporting_pages, batch)
                    sources = []
                    for candidate in candidate_pool:
                        if candidate["page"] in supporting_pages:
                            for source in candidate["sources"]:
                                if source not in sources:
                                    sources.append(source)
                    if evidence_chain:
                        method = "reference_followup"
                        evidence_chain = [
                            *evidence_chain,
                            {
                                "role": "value_source",
                                "page_number": supporting_pages,
                                "original_quote": result["evidence"],
                            },
                        ]
                    elif "vector_fallback" in sources:
                        method = "vector_fallback"
                    elif "tree" in sources:
                        method = "tree"
                    else:
                        method = "bm25"
                    pending_tree_pages = [
                        candidate["page"]
                        for candidate in candidate_pool
                        if "tree" in candidate.get("sources", []) and candidate.get("read_status") == "pending"
                    ]
                    found_candidate_event = {
                        "attempt_index": attempt_index,
                        "loaded_pages": loaded_pages,
                        "content_units": [candidate["content_unit"] for candidate in batch],
                        "evidence_pages": supporting_pages,
                        "resolution_method": method,
                        "retrieval_sources": sources,
                        "deferred_due_to_pending_tree_work": bool(pending_tree_pages),
                        "pending_tree_pages": pending_tree_pages,
                        "confidence": result.get("confidence"),
                    }
                    if _setting(client, "retrieval_log_evidence_preview_enabled", False):
                        found_candidate_event["evidence_preview"] = str(result.get("evidence", ""))[:80]
                    _emit_retrieval_log(
                        retrieval_logger,
                        "field_extraction_found_candidate",
                        field.name,
                        **found_candidate_event,
                    )
                    found_event = {
                        "status": result["status"],
                        "resolution_method": method,
                        "evidence_pages": supporting_pages,
                        "retrieval_sources": sources,
                    }
                    found_result = (field.name, _metadata_result(result, method, sources, evidence_chain))
                    if _has_pending_tree_work(candidate_pool):
                        if deferred_found_result is None:
                            deferred_found_result = found_result
                            deferred_found_event = found_event
                        reference_context_pages = []
                        continue
                    if deferred_found_result is not None:
                        if deferred_found_event is not None:
                            _emit_retrieval_log(
                                retrieval_logger,
                                "field_extraction_completed",
                                field.name,
                                **deferred_found_event,
                            )
                        return deferred_found_result
                    _emit_retrieval_log(
                        retrieval_logger,
                        "field_extraction_completed",
                        field.name,
                        **found_event,
                    )
                    return found_result

                last_reason = result.get("reason", last_reason)
                if result["status"] == ExtractionStatus.ERROR.value:
                    for candidate in batch:
                        candidate["read_status"] = "evaluation_error"
                    _emit_retrieval_log(
                        retrieval_logger,
                        "field_extraction_error",
                        field.name,
                        pages=loaded_pages,
                    )
                    if deferred_found_result is not None:
                        if _has_pending_tree_work(candidate_pool):
                            reference_context_pages = []
                            continue
                        if deferred_found_event is not None:
                            _emit_retrieval_log(
                                retrieval_logger,
                                "field_extraction_completed",
                                field.name,
                                **deferred_found_event,
                            )
                        return deferred_found_result
                    return field.name, _metadata_result(result, "error", [], [])
                if result["status"] != ExtractionStatus.REFERENCE_FOUND.value:
                    for candidate in batch:
                        candidate["read_status"] = "evaluated_irrelevant"
                    reference_context_pages = []
                    continue
                for candidate in batch:
                    candidate["read_status"] = "evaluated_reference"
                source_chain, source_depth = _reference_state_for_pages(candidate_pool, result["pages"], batch)
                if source_depth >= _setting(client, "max_reference_depth", 3):
                    for reference in result.get("references", []):
                        target_text = str(reference.get("target_text", "") or "").strip()
                        if target_text and target_text not in unresolved_targets:
                            unresolved_targets.append(target_text)
                    logger.debug("Reference depth limit reached for field=%s depth=%s", field.name, source_depth)
                    reference_context_pages = []
                    continue

                next_reference_chain = [
                    *source_chain,
                    {
                        "role": "reference_source",
                        "page_number": result["pages"],
                        "original_quote": result["evidence"],
                    },
                ]
                prioritized_candidates = []
                references = result.get("references", [])[: _setting(client, "max_reference_targets_per_step", 2)]
                for reference in references:
                    target_text = reference["target_text"]
                    key = "|".join(
                        [
                            ",".join(str(page) for page in result.get("pages", [])),
                            str(reference.get("reference_type", "")),
                            normalize_reference_text(target_text),
                        ]
                    )
                    if key in followed_reference_keys:
                        continue
                    followed_reference_keys.add(key)
                    resolution_payload = await _run_json_prompt(
                        client.retrieve_model,
                        _build_reference_resolution_prompt(
                            result["evidence"],
                            target_text,
                            structure_digest,
                            _candidate_manifest(candidate_pool),
                            sorted(processed_pages),
                        ),
                        retries=retries,
                        timeout_seconds=timeout_seconds,
                    )
                    valid_target_pages = structure_pages - set(result["pages"])
                    located_pages = filter_valid_reference_pages(
                        _normalize_ranked_page_list(resolution_payload.get("pages")),
                        valid_target_pages,
                    )
                    located = [{"page": page, "source": "reference_llm"} for page in located_pages]
                    if not located:
                        if target_text not in unresolved_targets:
                            unresolved_targets.append(target_text)
                        continue
                    logger.debug(
                        "Reference resolution field=%s source_pages=%s target_pages=%s sources=%s",
                        field.name,
                        result["pages"],
                        [location["page"] for location in located],
                        [location["source"] for location in located],
                    )
                    _emit_retrieval_log(
                        retrieval_logger,
                        "reference_targets_selected",
                        field.name,
                        source_pages=result["pages"],
                        target_pages=[location["page"] for location in located],
                    )
                    total_page_limit = max(0, int(_setting(client, "reference_total_page_limit", 9)))
                    enqueued_target = False
                    for location in located:
                        page = location["page"]
                        if page not in reference_target_pages and len(reference_target_pages) >= total_page_limit:
                            continue
                        reference_target_pages.add(page)
                        _merge_candidate(
                            candidate_pool,
                            _make_candidate(
                                page,
                                location["source"],
                                page_documents_by_page,
                                reason="引用目标定位",
                                reference_chain=next_reference_chain,
                                reference_depth=source_depth + 1,
                            ),
                        )
                        target_candidates = [
                            candidate for candidate in candidate_pool if candidate["page"] == page
                        ]
                        for candidate in target_candidates:
                            candidate["reference_chain"] = list(next_reference_chain)
                            candidate["reference_depth"] = source_depth + 1
                            _requeue_candidate(candidate)
                            if candidate not in prioritized_candidates:
                                prioritized_candidates.append(candidate)
                        enqueued_target = True
                    if not enqueued_target and target_text not in unresolved_targets:
                        unresolved_targets.append(target_text)
                if prioritized_candidates:
                    candidate_pool[:] = prioritized_candidates + [
                        candidate for candidate in candidate_pool if candidate not in prioritized_candidates
                    ]
                    reference_context_pages = result["pages"]
                else:
                    reference_context_pages = []
        except Exception as exc:
            _emit_retrieval_log(
                retrieval_logger,
                "field_extraction_failed",
                field.name,
                error_type=type(exc).__name__,
            )
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
    include_retrieval_metadata=False,
    retrieval_logger=None,
    long_context_mode=False,
):
    fields = normalize_schema(schema)
    semaphore = asyncio.Semaphore(max(1, min(max_concurrency, len(fields))))
    structure = []
    structure_digest = ""
    retrieval_payload = None
    page_documents = []
    query_cache = {}
    long_context_pages = []
    if long_context_mode:
        if not hasattr(client, "get_long_context_payload"):
            raise ValueError("长上下文模式缺少独立分页原文产物，请重新上传并完成建树后重试")
        long_context_payload = client.get_long_context_payload(doc_id)
        candidate_pages = long_context_payload.get("pages") if isinstance(long_context_payload, dict) else None
        if not isinstance(candidate_pages, list) or not candidate_pages:
            raise ValueError("长上下文模式缺少独立分页原文产物，请重新上传并完成建树后重试")
        long_context_pages = sorted(
            [
                {"page": page["page"], "content": str(page.get("content", ""))}
                for page in candidate_pages
                if isinstance(page, dict) and isinstance(page.get("page"), int)
            ],
            key=lambda page: page["page"],
        )
        if not long_context_pages:
            raise ValueError("长上下文模式的独立分页原文产物无有效页面，请重新上传并完成建树后重试")
    else:
        structure = json.loads(client.get_document_structure(doc_id))
        structure_digest = _build_structure_digest(structure)
        if hasattr(client, "get_retrieval_payload"):
            retrieval_payload = client.get_retrieval_payload(doc_id)
    enhanced_enabled = bool(
        not long_context_mode
        and retrieval_payload
        and retrieval_payload.get("type") == "pdf"
        and isinstance(retrieval_payload.get("pages"), list)
        and retrieval_payload["pages"]
    )
    if enhanced_enabled:
        page_documents = build_page_documents(retrieval_payload["pages"], retrieval_payload.get("structure", structure))
        query_cache = _load_query_cache(client)
    vector_lock = asyncio.Lock()
    total_fields = len(fields)
    completed_count = 0

    async def run_field(field):
        nonlocal completed_count
        if long_context_mode:
            result = await _extract_one_field_long_context(
                client,
                field,
                long_context_pages,
                semaphore,
                retries=retries,
                timeout_seconds=timeout_seconds,
            )
        elif enhanced_enabled:
            result = await _extract_one_field_enhanced(
                client,
                doc_id,
                field,
                structure,
                structure_digest,
                retrieval_payload,
                page_documents,
                query_cache,
                vector_lock,
                semaphore,
                retrieval_logger=retrieval_logger,
                retries=retries,
                timeout_seconds=timeout_seconds,
            )
        else:
            result = await _extract_one_field_legacy(
                client,
                doc_id,
                field,
                structure_digest,
                semaphore,
                retries=retries,
                timeout_seconds=timeout_seconds,
            )
        if not include_retrieval_metadata:
            result = (
                result[0],
                {
                    key: value
                    for key, value in result[1].items()
                    if key not in {"resolution_method", "retrieval_sources", "evidence_chain"}
                },
            )
        completed_count += 1
        if progress_callback is not None:
            progress_callback(completed_count, total_fields)
        return result

    tasks = [run_field(field) for field in fields]
    results = await asyncio.gather(*tasks)
    if enhanced_enabled:
        _save_query_cache(client, query_cache)
    return {name: payload for name, payload in results}


def extract_contract_fields(
    client,
    doc_id,
    schema,
    max_concurrency=8,
    timeout_seconds=45,
    retries=1,
    progress_callback=None,
    include_retrieval_metadata=False,
    retrieval_logger=None,
    long_context_mode=False,
):
    coro = _extract_contract_fields_async(
        client,
        doc_id,
        schema,
        max_concurrency=max_concurrency,
        timeout_seconds=timeout_seconds,
        retries=retries,
        progress_callback=progress_callback,
        include_retrieval_metadata=include_retrieval_metadata,
        retrieval_logger=retrieval_logger,
        long_context_mode=long_context_mode,
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

import asyncio
import concurrent.futures
import json
import os
from pathlib import Path

from .contract_extraction import ConfidenceLevel, ExtractionStatus, normalize_schema
from .identity import build_doc_id, compute_file_sha256


def _safe_json_loads(payload):
    try:
        return json.loads(payload)
    except (TypeError, json.JSONDecodeError):
        return None


def _preview(value, limit=220):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "..."


def _normalize_pages(value):
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, list):
        return [int(item) for item in value if isinstance(item, int) or (isinstance(item, str) and item.isdigit())]
    return []


def _normalize_worker_result(field_name, raw_payload):
    payload = raw_payload if isinstance(raw_payload, dict) else {}
    status = str(payload.get("status", "")).strip().lower()
    if status not in {item.value for item in ExtractionStatus}:
        status = ExtractionStatus.ERROR.value
    candidate_pages = _normalize_pages(payload.get("candidate_pages"))
    pages = _normalize_pages(payload.get("pages")) or list(candidate_pages)

    if status != ExtractionStatus.FOUND.value:
        return {
            "field": field_name,
            "candidate_pages": candidate_pages,
            "status": status,
            "value": "",
            "evidence": "",
            "pages": pages,
            "confidence": ConfidenceLevel.LOW.value,
            "reason": str(payload.get("reason", "")).strip() or "工作 Agent 未返回有效结果",
        }

    confidence = str(payload.get("confidence", "")).strip()
    if confidence not in {item.value for item in ConfidenceLevel}:
        confidence = ConfidenceLevel.LOW.value
    return {
        "field": field_name,
        "candidate_pages": candidate_pages,
        "status": ExtractionStatus.FOUND.value,
        "value": str(payload.get("value", "")).strip(),
        "evidence": str(payload.get("evidence", "")).strip(),
        "pages": pages,
        "confidence": confidence,
        "reason": str(payload.get("reason", "")).strip() or None,
    }


def _build_worker_prompt(field):
    return f"""
你是白盒合同抽取演示中的字段工作 Agent，只负责一个字段。

字段定义：
{json.dumps(field.__dict__, ensure_ascii=False, indent=2)}

要求：
- 所有自然语言思考摘要、工具调用前说明、以及最终自由文本都必须使用简体中文。
- 必须先调用 get_document_structure() 查看目录树和摘要。
- 根据目录树摘要推理这个字段最可能出现的最小候选页范围。
- 然后调用 get_page_content()，只读取紧凑页码范围，不要抓取全文。
- 最终只能返回 JSON。
- 返回结果中除了最终抽取结果，还必须包含 candidate_pages。
- status 只能是：found、not_found、error。
- confidence 只能是：High、Medium、Low。
- 如果 status 是 not_found 或 error，则 confidence 必须是 Low，且 value/evidence 为空字符串。
- 如果找到字段，evidence 尽量引用原文中的关键证据。

返回 JSON：
{{
  "field": "{field.name}",
  "candidate_pages": [4, 5],
  "status": "found",
  "value": "示例值",
  "evidence": "原文中的精确证据",
  "pages": [4],
  "confidence": "High",
  "reason": null
}}
""".strip()


def _build_orchestrator_prompt(schema_fields):
    return f"""
你是白盒合同抽取演示中的中控调度 Agent。

字段列表：
{json.dumps([field.__dict__ for field in schema_fields], ensure_ascii=False, indent=2)}

要求：
- 所有自然语言思考摘要、工具调用前说明、以及最终自由文本都必须使用简体中文。
- 必须先调用 get_document() 获取文档元信息。
- 再调用 get_document_structure() 理解文档树。
- 为每个字段生成一个独立的 worker assignment。
- handoff_message 要简洁、明确、字段导向，使用中文表达。
- 最终只能返回 JSON。

返回 JSON：
{{
  "assignments": [
    {{
      "field": "contract_amount",
      "handoff_message": "优先检查价格与支付相关条款。"
    }}
  ]
}}
""".strip()


def _build_context_prompt():
    return """
你是白盒合同抽取演示中的索引/上下文 Agent。

要求：
- 所有自然语言思考摘要、工具调用前说明、以及最终自由文本都必须使用简体中文。
- 必须先调用 get_document()。
- 再调用一次 get_document_structure()。
- 用 3 条简短中文要点总结缓存状态、doc_id、tree_id 和顶层文档树线索。
""".strip()


def _build_merge_prompt(worker_results):
    return f"""
你是白盒合同抽取演示中的聚合/复核 Agent。

各 Worker 结果：
{json.dumps(worker_results, ensure_ascii=False, indent=2)}

要求：
- 所有自然语言思考摘要、工具调用前说明、以及最终自由文本都必须使用简体中文。
- 不要再调用任何文档工具。
- 用简短中文总结整体抽取情况。
- 保留 worker 输出，作为最终 results 对象。
- 最终只能返回 JSON。

返回 JSON：
{{
  "summary": "简短中文总结",
  "results": {{}}
}}
""".strip()


async def _run_agent_with_sdk(name, instructions, prompt, tools, *, model=None, verbose=False, printer=print):
    try:
        from agents import Agent, ModelSettings, Runner, set_tracing_disabled
        from agents.extensions.models.litellm_model import LitellmModel
        from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
        from openai.types.responses import ResponseReasoningSummaryTextDeltaEvent, ResponseTextDeltaEvent
    except ImportError as exc:
        raise ImportError(
            "Whitebox demo requires the optional 'openai-agents' package. "
            "Please activate the PageIndex environment and install it first."
        ) from exc

    set_tracing_disabled(True)
    model_name = str(model or "").removeprefix("litellm/")
    agent_model = LitellmModel(
        model=model_name,
        base_url=os.getenv("OPENAI_API_BASE") or None,
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY") or None,
    )
    agent = Agent(
        name=name,
        instructions=instructions,
        tools=tools,
        model=agent_model,
        model_settings=ModelSettings(parallel_tool_calls=False),
    )
    streamed_run = Runner.run_streamed(agent, prompt)
    reasoning_chunks = []
    text_chunks = []
    async for event in streamed_run.stream_events():
        if isinstance(event, RawResponsesStreamEvent):
            if isinstance(event.data, ResponseReasoningSummaryTextDeltaEvent):
                reasoning_chunks.append(event.data.delta)
            elif isinstance(event.data, ResponseTextDeltaEvent):
                text_chunks.append(event.data.delta)
        elif isinstance(event, RunItemStreamEvent):
            item = event.item
            if item.type == "tool_call_item":
                raw = item.raw_item
                arguments = getattr(raw, "arguments", "{}")
                printer(f"[{name}] tool call: {raw.name} {arguments if verbose else ''}".rstrip())
            elif item.type == "tool_call_output_item" and verbose:
                printer(f"[{name}] tool output: {_preview(item.output)}")

    reasoning_text = "".join(reasoning_chunks).strip()
    text_output = "".join(text_chunks).strip()
    if reasoning_text:
        printer(f"[{name}] reasoning: {reasoning_text}")
    if text_output:
        printer(f"[{name}] text: {text_output}")
    final_output = "" if not streamed_run.final_output else str(streamed_run.final_output)
    return {
        "name": name,
        "reasoning": reasoning_text,
        "text": text_output,
        "final_output": final_output,
    }


def _make_tools(client, doc_id, decorate=True):
    def get_document() -> str:
        """Get document metadata including doc_id, tree_id, status, and page count."""
        return client.get_document(doc_id)

    def get_document_structure() -> str:
        """Get the cached tree structure without large text fields."""
        return client.get_document_structure(doc_id)

    def get_page_content(pages: str) -> str:
        """Get text for a compact PDF page selection like '4-5' or '7'."""
        return client.get_page_content(doc_id, pages)

    if decorate:
        try:
            from agents import function_tool
        except ImportError:
            function_tool = None
        if function_tool is not None:
            get_document_tool = function_tool(get_document)
            get_document_structure_tool = function_tool(get_document_structure)
            get_page_content_tool = function_tool(get_page_content)
            return [get_document_tool, get_document_structure_tool, get_page_content_tool]

    return [get_document, get_document_structure, get_page_content]


def _format_page_selection(pages):
    normalized = sorted({page for page in pages if isinstance(page, int) and page > 0})
    if not normalized:
        return ""
    ranges = []
    start = end = normalized[0]
    for page in normalized[1:]:
        if page == end + 1:
            end = page
            continue
        ranges.append(f"{start}-{end}" if start != end else str(start))
        start = end = page
    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ",".join(ranges)


async def _run_whitebox_demo_async(
    client,
    doc_id,
    schema,
    *,
    cache_hit=False,
    verbose=False,
    max_worker_concurrency=2,
    printer=print,
    agent_executor=None,
):
    schema_fields = normalize_schema(schema)
    doc_meta = _safe_json_loads(client.get_document(doc_id)) or {}
    tree_id = client.get_tree_id(doc_id)
    printer("=" * 60)
    printer("Whitebox Contract Extraction Demo")
    printer("=" * 60)
    printer(f"[indexing] doc_id={doc_id}")
    printer(f"[indexing] tree_id={tree_id}")
    printer(f"[indexing] cache_hit={str(cache_hit).lower()}")
    printer(f"[indexing] schema_fields={[field.name for field in schema_fields]}")
    printer(f"[orchestrator] worker_max_concurrency={max(1, min(max_worker_concurrency, len(schema_fields) or 1))}")

    if agent_executor is None:
        async def execute_agent(name, instructions, prompt, tools, verbose=False, printer=print):
            return await _run_agent_with_sdk(
                name,
                instructions,
                prompt,
                tools,
                model=client.retrieve_model,
                verbose=verbose,
                printer=printer,
            )
    else:
        execute_agent = agent_executor

    tools = _make_tools(client, doc_id, decorate=agent_executor is None)

    context_result = await execute_agent(
        "context",
        "你是上下文 Agent。请用中文简要说明缓存状态和文档树顶层线索。",
        _build_context_prompt(),
        tools,
        verbose=verbose,
        printer=printer,
    )

    orchestrator_result = await execute_agent(
        "orchestrator",
        "你是调度 Agent。请用中文读取文档元信息和文档树，并为每个字段分配一个独立 Worker 任务。",
        _build_orchestrator_prompt(schema_fields),
        tools,
        verbose=verbose,
        printer=printer,
    )
    orchestrator_payload = _safe_json_loads(orchestrator_result["final_output"]) or {"assignments": []}
    assignments = orchestrator_payload.get("assignments", [])
    if not assignments:
        assignments = [{"field": field.name, "handoff_message": field.description} for field in schema_fields]

    printer(f"[orchestrator] assignments={json.dumps(assignments, ensure_ascii=False)}")

    worker_semaphore = asyncio.Semaphore(max(1, min(max_worker_concurrency, len(schema_fields) or 1)))

    async def run_worker(field):
        async with worker_semaphore:
            printer(f"[orchestrator] handoff -> worker:{field.name}")
            worker_result = await execute_agent(
                f"worker:{field.name}",
                f"你是字段 {field.name} 的工作 Agent。请用中文展示过程，谨慎使用工具，并把页码读取范围控制得尽量小。",
                _build_worker_prompt(field),
                tools,
                verbose=verbose,
                printer=printer,
            )
            worker_payload = _safe_json_loads(worker_result["final_output"]) or {}
            normalized = _normalize_worker_result(field.name, worker_payload)
            printer(f"[worker:{field.name}] candidate_pages={normalized['candidate_pages']}")
            if verbose:
                printer(f"[worker:{field.name}] normalized_result={json.dumps(normalized, ensure_ascii=False)}")
            return field.name, normalized

    worker_results = dict(await asyncio.gather(*(run_worker(field) for field in schema_fields)))

    merge_result = await execute_agent(
        "merge",
        "你是聚合/复核 Agent。请用中文总结各 Worker 结果，不要再调用任何文档工具。",
        _build_merge_prompt(worker_results),
        [],
        verbose=verbose,
        printer=printer,
    )
    merge_payload = _safe_json_loads(merge_result["final_output"]) or {}
    final_results = merge_payload.get("results") if isinstance(merge_payload.get("results"), dict) else worker_results
    summary = str(merge_payload.get("summary", "")).strip() or "已完成 Worker 结果聚合。"
    printer(f"[merge] summary={summary}")
    printer("[final] extracted_json=")
    printer(json.dumps(final_results, ensure_ascii=False, indent=2))

    return {
        "doc_id": doc_id,
        "tree_id": tree_id,
        "cache_hit": cache_hit,
        "document": doc_meta,
        "context": context_result,
        "orchestrator": orchestrator_result,
        "workers": worker_results,
        "merge": {"summary": summary, "raw": merge_result},
        "results": final_results,
    }



def run_contract_extraction_whitebox_demo(
    pdf_path,
    schema_path,
    workspace,
    *,
    verbose=False,
    max_worker_concurrency=2,
    printer=print,
    client=None,
    agent_executor=None,
):
    from .client import PageIndexClient

    pdf_path = Path(pdf_path)
    schema_path = Path(schema_path)
    workspace = Path(workspace)
    client = client or PageIndexClient(workspace=workspace)
    source_sha256 = compute_file_sha256(pdf_path)
    doc_id = build_doc_id(source_sha256)
    cached_doc = client.documents.get(doc_id)
    cache_hit = bool(cached_doc and cached_doc.get("source_sha256") == source_sha256 and cached_doc.get("index_strategy") == "hybrid")

    def demo_progress(event):
        cache_suffix = " cache-hit" if cache_hit and event["stage"] == "completed" else ""
        printer(f"[indexing] {event['percent']:>3}% {event['stage']}: {event['message']}{cache_suffix}")

    doc_id = client.index(
        str(pdf_path),
        strategy="hybrid",
        progress_callback=demo_progress,
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    coro = _run_whitebox_demo_async(
        client,
        doc_id,
        schema,
        cache_hit=cache_hit,
        verbose=verbose,
        max_worker_concurrency=max_worker_concurrency,
        printer=printer,
        agent_executor=agent_executor,
    )
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)


__all__ = ["run_contract_extraction_whitebox_demo"]

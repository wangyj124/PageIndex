import asyncio
import copy
import logging
import uuid

from .llm import extract_json, llm_acompletion
from .tree_utils import generate_doc_description
from .llm import count_tokens


def _append_text(target_text, extra_text):
    target_text = (target_text or "").strip()
    extra_text = (extra_text or "").strip()
    if not extra_text:
        return target_text
    if not target_text:
        return extra_text
    return f"{target_text}\n\n{extra_text}"


def _make_merge_paragraph(node):
    title = (node.get("title") or "").strip()
    text = (node.get("text") or "").strip()
    if title and text:
        return f"{title}\n{text}"
    return title or text


def thin_small_nodes(nodes, min_tokens, model_name=None, token_counter_fn=None):
    token_counter_fn = token_counter_fn or count_tokens

    def _thin(current_nodes):
        for node in current_nodes:
            _thin(node.get("nodes", []))

        kept_nodes = []
        for node in current_nodes:
            children = node.get("nodes", [])
            merged_children = []
            kept_children = []
            for child in children:
                child_children = child.get("nodes", [])
                child_tokens = token_counter_fn(child.get("text") or "", model=model_name)
                if not child_children and child_tokens < min_tokens:
                    merged_children.append(_make_merge_paragraph(child))
                else:
                    kept_children.append(child)

            node["nodes"] = kept_children
            for merged_text in merged_children:
                node["text"] = _append_text(node.get("text"), merged_text)
            kept_nodes.append(node)

        return kept_nodes

    return _thin(nodes)


async def refine_large_nodes(nodes, max_tokens=2000, model=None, llm_fn=None, token_counter_fn=None, logger=None):
    llm_fn = llm_fn or llm_acompletion
    token_counter_fn = token_counter_fn or count_tokens
    logger = logger or logging.getLogger(__name__)

    async def _refine(node):
        for child in node.get("nodes", []):
            await _refine(child)

        if node.get("nodes"):
            return

        span = (node.get("end_index") or 0) - (node.get("start_index") or 0)
        token_count = token_counter_fn(node.get("text") or "", model=model)
        if token_count <= max_tokens or span <= 1:
            return

        prompt = f"""
这是一个过长的 Markdown 章节，请根据语义连贯性，将其拆分为 2-4 个逻辑子段落。
请只返回 JSON List，格式如下：
[
  {{"sub_title": "...", "sub_text": "..."}}
]

原章节标题：
{node.get("title", "")}

原章节正文：
{node.get("text", "")}
""".strip()

        response = await llm_fn(model, prompt)
        parsed = extract_json(response)
        if not isinstance(parsed, list) or not parsed:
            logger.warning("Large node refinement returned invalid JSON; keeping original node")
            return

        children = []
        for index, item in enumerate(parsed, start=1):
            sub_title = str(item.get("sub_title", "")).strip()
            sub_text = str(item.get("sub_text", "")).strip()
            if not sub_title and not sub_text:
                continue
            child_node = {
                "node_id": f"{node.get('node_id', 'node')}_ref_{index:02d}_{uuid.uuid4().hex[:8]}",
                "title": sub_title or f"{node.get('title', 'Section')} - Part {index}",
                "start_index": node.get("start_index"),
                "end_index": node.get("end_index"),
                "text": sub_text,
                "nodes": [],
            }
            children.append(child_node)

        if not children:
            return

        node["text"] = ""
        node["nodes"] = children

    for node in nodes:
        await _refine(node)
    return nodes


async def generate_summaries(nodes, model=None, llm_fn=None):
    llm_fn = llm_fn or llm_acompletion
    tasks = []

    async def _summarize_text(text, prompt_prefix):
        clean_text = (text or "").strip()
        if not clean_text:
            return ""
        prompt = f"{prompt_prefix}\n\n{clean_text}".strip()
        return await llm_fn(model, prompt)

    def _collect(node_list):
        for node in node_list:
            if node.get("nodes"):
                tasks.append(
                    (
                        node,
                        "prefix_summary",
                        _summarize_text(
                            node.get("text", ""),
                            "请对以下父级章节本身的正文做一段简洁概括，输出摘要正文即可：",
                        ),
                    )
                )
                _collect(node["nodes"])
            else:
                tasks.append(
                    (
                        node,
                        "summary",
                        _summarize_text(
                            node.get("text", ""),
                            "请对以下叶子章节正文做一段简洁提炼摘要，输出摘要正文即可：",
                        ),
                    )
                )

    _collect(nodes)
    results = await asyncio.gather(*(task[2] for task in tasks))
    for (node, field_name, _), summary in zip(tasks, results):
        node[field_name] = summary.strip() if isinstance(summary, str) else summary
    return nodes


async def optimize_and_summarize_tree(tree_nodes, min_tokens, max_tokens, model=None, llm_fn=None, token_counter_fn=None):
    optimized_tree = copy.deepcopy(tree_nodes)
    token_counter_fn = token_counter_fn or count_tokens

    thin_small_nodes(optimized_tree, min_tokens=min_tokens, model_name=model, token_counter_fn=token_counter_fn)
    await refine_large_nodes(
        optimized_tree,
        max_tokens=max_tokens,
        model=model,
        llm_fn=llm_fn,
        token_counter_fn=token_counter_fn,
    )
    await generate_summaries(optimized_tree, model=model, llm_fn=llm_fn)

    top_level_titles = [node.get("title", "") for node in optimized_tree]
    doc_description_prompt = (
        "请根据以下文档顶级标题，生成一句简洁的全局文档描述，输出描述正文即可：\n\n"
        + "\n".join(title for title in top_level_titles if title)
    )
    doc_description = ""
    if top_level_titles:
        if llm_fn is not None:
            doc_description = await llm_fn(model, doc_description_prompt)
        else:
            doc_description = generate_doc_description([{"title": title} for title in top_level_titles], model=model)

    return {
        "doc_description": doc_description.strip() if isinstance(doc_description, str) else doc_description,
        "structure": optimized_tree,
    }


__all__ = [
    "thin_small_nodes",
    "refine_large_nodes",
    "generate_summaries",
    "optimize_and_summarize_tree",
]

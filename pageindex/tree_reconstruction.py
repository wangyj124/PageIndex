import json
import logging
import re

from .llm import extract_json, llm_completion


RECONSTRUCTION_SYSTEM_PROMPT = """
你是一名专业的长篇法律与工业文档结构裁定专家。

你不是创意写作者，而是对文档结构拥有裁决权的严格审核者。
你的职责是检查每一个候选节点，并给出最终的结构判断。

判断规则：
1. 必须尊重原始输入顺序，绝不能重排节点，也不能遗漏任何 node_id。
2. corrected_level 表示递增层级：1 为最高层，2 是 1 的子层级，依此类推。
3. 有些节点的 needs_llm_fix = true，表示代码无法把它们对齐到物理页。
   对这些节点，你必须做出结构上的保留或剔除判断：
   - 剔除：如果该项不是真正的标题，就强制 corrected_level = -1。
   - 保留：如果该项是真实标题，就结合上下文给出最合理的 corrected_level。
4. 不要虚构页码。physical_index 可以为 null，这是允许的。
5. 在可用时，请结合标题语义、snippet 证据、邻近节点和页序进行判断。
6. 只输出 JSON。不要输出 JSON 之外的说明，也不要使用 markdown 代码块。
7. 每个结果项都必须包含 node_id、corrected_level 和 decision_reason。
8. 在合同文档中，诸如 "此页为合同签字页"、"合同签字页"、"签字页"、"买方"、"卖方"
   这类签字区标题通常属于有效结构标题。除非上下文明确证明它们只是噪声，
   否则应优先保留签字页作为父标题，并将买方/卖方区块作为其子标题。
""".strip()


RECONSTRUCTION_USER_PROMPT_TEMPLATE = """
请审核下面这些来自长篇法律或工业文档的候选节点。

输入字段说明：
- title: 候选标题文本
- snippet: 该标题下方正文的简短预览
- physical_index: 物理页码，可以为 null
- needs_llm_fix: 代码对齐失败，是否需要你做结构判断

对每个节点：
- 如果 needs_llm_fix 为 false，仍然要从全局视角复核它的结构层级。
- 如果 needs_llm_fix 为 true，你必须做出判断：
  - 如果它只是加粗文本、说明、图注或其他非标题噪声，就将 corrected_level 设为 -1 并剔除。
  - 如果它是真实标题但丢失了页码锚点，就保留并赋予合理的 corrected_level。
- 合同签字页需要特别谨慎：
  - 当 "此页为合同签字页"/"合同签字页"/"签字页" 引出签字区或合同主体区块时，应保留为标题。
  - "买方" 和 "卖方" 应保留为签字页或合同签署部分下的子标题。
  - 如果你剔除了签字页父标题，不要让买方/卖方停留在缺少父级的更深层级。

请严格返回如下结构的 JSON 列表：
[
  {{
    "node_id": "001",
    "corrected_level": 1,
    "decision_reason": "简短但具体的结构判断理由。"
  }}
]

候选节点：
{context_json}
""".strip()


class TreeReconstructionError(Exception):
    def __init__(self, message, failure_type=None, node_id=None):
        super().__init__(message)
        self.failure_type = failure_type
        self.node_id = node_id


def build_context_payload(initial_nodes, snippet_length=80):
    payload = []
    for node in initial_nodes:
        raw_text = str(node.get("text", "") or "")
        snippet = re.sub(r"\s+", " ", raw_text.replace("\r", " ").replace("\n", " ")).strip()
        payload.append(
            {
                "node_id": node.get("node_id"),
                "title": node.get("title", ""),
                "physical_index": node.get("physical_index"),
                "needs_llm_fix": bool(node.get("needs_llm_fix", False)),
                "snippet": snippet[:snippet_length] if snippet else "",
            }
        )
    return payload


def build_reconstruction_prompt(context_payload):
    context_json = json.dumps(context_payload, ensure_ascii=False, indent=2)
    return RECONSTRUCTION_USER_PROMPT_TEMPLATE.format(context_json=context_json)


def call_reconstruction_llm(context_payload, model=None, llm_fn=None):
    llm_fn = llm_fn or llm_completion
    user_prompt = build_reconstruction_prompt(context_payload)
    system_message = [{"role": "system", "content": RECONSTRUCTION_SYSTEM_PROMPT}]
    try:
        response = llm_fn(model=model, prompt=user_prompt, chat_history=system_message)
    except TypeError:
        response = llm_fn(model=model, prompt=f"{RECONSTRUCTION_SYSTEM_PROMPT}\n\n{user_prompt}")

    parsed = extract_json(response)
    if not isinstance(parsed, list):
        raise TreeReconstructionError(
            "LLM reconstruction response is not a JSON list",
            failure_type="invalid_llm_response",
        )
    return parsed


def merge_corrected_levels(initial_nodes, llm_result):
    llm_map = {}
    for item in llm_result:
        node_id = item.get("node_id")
        corrected_level = item.get("corrected_level")
        decision_reason = item.get("decision_reason", item.get("reasoning", ""))
        if node_id is None:
            raise TreeReconstructionError(
                "LLM reconstruction item is missing node_id",
                failure_type="invalid_llm_response",
            )
        if not isinstance(corrected_level, int) or corrected_level == 0 or corrected_level < -1:
            raise TreeReconstructionError(
                f"LLM reconstruction item has invalid corrected_level for node_id={node_id}",
                failure_type="invalid_corrected_level",
                node_id=node_id,
            )
        llm_map[node_id] = {
            "corrected_level": corrected_level,
            "decision_reason": decision_reason,
            "reasoning": decision_reason,
        }

    merged_nodes = []
    for node in initial_nodes:
        node_id = node.get("node_id")
        if node_id not in llm_map:
            raise TreeReconstructionError(
                f"LLM reconstruction missing node_id={node_id}",
                failure_type="missing_node_mapping",
                node_id=node_id,
            )
        merged = dict(node)
        merged.update(llm_map[node_id])
        merged_nodes.append(merged)
    return merged_nodes


def validate_tree_logic(nodes):
    valid_nodes = [
        node
        for node in nodes
        if isinstance(node.get("corrected_level"), int) and node.get("corrected_level") >= 1
    ]
    if not valid_nodes:
        raise TreeReconstructionError(
            "No valid corrected levels found after reconstruction",
            failure_type="empty_corrected_levels",
        )

    first_node = valid_nodes[0]
    if first_node["corrected_level"] not in (1, 2):
        raise TreeReconstructionError(
            f"First valid node has unreasonable corrected_level={first_node['corrected_level']}",
            failure_type="invalid_first_node_level",
            node_id=first_node.get("node_id"),
        )

    previous_node = first_node
    for current_node in valid_nodes[1:]:
        if current_node["corrected_level"] > previous_node["corrected_level"] + 1:
            raise TreeReconstructionError(
                (
                    "Detected isolated level jump: "
                    f"prev={previous_node['corrected_level']} current={current_node['corrected_level']}"
                ),
                failure_type="isolated_level_jump",
                node_id=current_node.get("node_id"),
            )
        previous_node = current_node

    return True


def repair_level_jumps_after_killed_nodes(nodes, logger=None):
    """
    Repair level jumps caused by an intermediate node being killed by the LLM.

    Example:
    - Volume I: level 1
    - Signature Page: -1
    - Buyer: level 3
    - Seller: level 3

    After killing "Signature Page", Buyer/Seller would jump from 1 to 3. In this
    narrow case we lower the over-deep run to sit directly under the previous
    valid parent, preserving order and avoiding a hard failure.
    """
    repaired_nodes = [dict(node) for node in nodes]
    previous_valid_level = None
    previous_valid_node = None
    skipped_nodes = []
    index = 0

    while index < len(repaired_nodes):
        node = repaired_nodes[index]
        current_level = node.get("corrected_level")
        is_valid = isinstance(current_level, int) and current_level >= 1

        if not is_valid:
            skipped_nodes.append(node)
            index += 1
            continue

        if (
            previous_valid_level is not None
            and current_level > previous_valid_level + 1
            and skipped_nodes
        ):
            delta = current_level - (previous_valid_level + 1)
            repair_start = index
            repair_end = index

            while repair_end < len(repaired_nodes):
                repair_node = repaired_nodes[repair_end]
                repair_level = repair_node.get("corrected_level")
                if isinstance(repair_level, int) and repair_level >= 1:
                    if repair_level <= previous_valid_level:
                        break
                    new_level = max(previous_valid_level + 1, repair_level - delta)
                    if new_level != repair_level:
                        repair_node["corrected_level"] = new_level
                        repair_node["auto_repair_reason"] = (
                            "Lowered level to avoid isolated jump after killed node(s): "
                            f"previous_node_id={previous_valid_node.get('node_id') if previous_valid_node else None}, "
                            f"previous_level={previous_valid_level}, "
                            f"original_level={repair_level}, repaired_level={new_level}, "
                            f"killed_node_ids={[item.get('node_id') for item in skipped_nodes]}"
                        )
                repair_end += 1

            if logger is not None:
                logger.info(
                    "Repaired isolated level jump after killed node(s)",
                    extra={
                        "previous_node_id": previous_valid_node.get("node_id") if previous_valid_node else None,
                        "previous_level": previous_valid_level,
                        "current_node_id": node.get("node_id"),
                        "repair_start": repair_start,
                        "repair_end": repair_end,
                        "killed_node_ids": [item.get("node_id") for item in skipped_nodes],
                    },
                )

            current_level = repaired_nodes[index].get("corrected_level")
            skipped_nodes = []

        previous_valid_level = current_level
        previous_valid_node = repaired_nodes[index]
        skipped_nodes = []
        index += 1

    return repaired_nodes


def reconstruct_tree_structure(initial_nodes, model=None, llm_fn=None, logger=None):
    logger = logger or logging.getLogger(__name__)
    context_payload = build_context_payload(initial_nodes)
    llm_result = call_reconstruction_llm(context_payload, model=model, llm_fn=llm_fn)
    merged_nodes = merge_corrected_levels(initial_nodes, llm_result)
    merged_nodes = repair_level_jumps_after_killed_nodes(merged_nodes, logger=logger)

    try:
        validate_tree_logic(merged_nodes)
    except TreeReconstructionError as exc:
        logger.error(
            "Tree reconstruction validation failed",
            extra={
                "failure_type": exc.failure_type,
                "node_id": exc.node_id,
            },
        )
        raise

    return merged_nodes


__all__ = [
    "TreeReconstructionError",
    "RECONSTRUCTION_SYSTEM_PROMPT",
    "RECONSTRUCTION_USER_PROMPT_TEMPLATE",
    "build_context_payload",
    "build_reconstruction_prompt",
    "call_reconstruction_llm",
    "merge_corrected_levels",
    "repair_level_jumps_after_killed_nodes",
    "validate_tree_logic",
    "reconstruct_tree_structure",
]

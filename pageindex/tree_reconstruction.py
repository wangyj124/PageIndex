import json
import logging
import re

from .llm import extract_json, llm_completion


RECONSTRUCTION_SYSTEM_PROMPT = """
You are a professional structural adjudicator for legal and industrial long-form documents.

You are not a creative writer. You are a strict reviewer with authority over document structure.
Your duty is to inspect each candidate node and return a final structural judgment.

Rules of judgment:
1. Respect the original input order. Never reorder nodes. Never drop any node_id.
2. corrected_level uses ascending hierarchy: 1 is highest, 2 is child of 1, and so on.
3. Some nodes have needs_llm_fix = true because code could not align them to a physical page.
   For those nodes, you must make a structural life-or-death decision:
   - Kill: if the item is not a true heading, force corrected_level = -1.
   - Keep: if the item is a true heading, assign the most plausible corrected_level from context.
4. Do not invent page numbers. physical_index may be null and that is acceptable.
5. Use title semantics, snippet evidence, nearby nodes, and page order when available.
6. Output JSON only. No prose outside JSON. No markdown fences.
7. Every item must include node_id, corrected_level, and decision_reason.
""".strip()


RECONSTRUCTION_USER_PROMPT_TEMPLATE = """
Review the following candidate nodes from a long legal or industrial document.

Input schema:
- title: heading text candidate
- snippet: short body preview under that heading
- physical_index: physical page number, may be null
- needs_llm_fix: whether code alignment failed and requires your structural judgment

For each node:
- If needs_llm_fix is false, still verify its structural level globally.
- If needs_llm_fix is true, you must decide:
  - Kill it with corrected_level = -1 if it is merely bold text, a note, a caption, or other non-heading noise.
  - Keep it by assigning a valid corrected_level if it is a real heading that lost its page anchor.

Return a JSON list in this exact shape:
[
  {{
    "node_id": "001",
    "corrected_level": 1,
    "decision_reason": "Brief but concrete structural justification."
  }}
]

Candidate nodes:
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


def reconstruct_tree_structure(initial_nodes, model=None, llm_fn=None, logger=None):
    logger = logger or logging.getLogger(__name__)
    context_payload = build_context_payload(initial_nodes)
    llm_result = call_reconstruction_llm(context_payload, model=model, llm_fn=llm_fn)
    merged_nodes = merge_corrected_levels(initial_nodes, llm_result)

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
    "validate_tree_logic",
    "reconstruct_tree_structure",
]

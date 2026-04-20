import logging

from .markdown import (
    DEBUG_LOG_DIR,
    build_pdf_page_text_map,
    build_hybrid_headings_from_markdown_and_json,
    dump_debug_json,
    emit_debug_log,
    extract_node_text_content,
    extract_nodes_from_markdown,
    get_page_range_text,
)
from .tree_reconstruction import reconstruct_tree_structure
from .tree_utils import build_tree_and_intervals


PREFACE_TITLE = "\u524d\u8a00/\u5c01\u9762"


def append_text_block(base_text, extra_text):
    base_text = str(base_text or "").strip()
    extra_text = str(extra_text or "").strip()
    if not extra_text:
        return base_text
    if not base_text:
        return extra_text
    if extra_text in base_text:
        return base_text
    if base_text in extra_text:
        return extra_text
    return f"{base_text}\n\n{extra_text}"


def collapse_demoted_nodes(flat_nodes):
    if not flat_nodes:
        return flat_nodes

    collapsed_nodes = []
    synthetic_preface = None

    for node in flat_nodes:
        corrected_level = node.get("corrected_level")
        if corrected_level != -1:
            collapsed_nodes.append(dict(node))
            continue

        demoted_text = str(node.get("text", "") or "").strip() or str(node.get("title", "") or "").strip()
        if not demoted_text:
            continue

        if collapsed_nodes:
            collapsed_nodes[-1]["text"] = append_text_block(collapsed_nodes[-1].get("text", ""), demoted_text)
            continue

        if synthetic_preface is None:
            synthetic_preface = {
                "node_id": "preface_00",
                "corrected_level": 1,
                "title": PREFACE_TITLE,
                "start_index": 1,
                "physical_index": 1,
                "text": demoted_text,
                "needs_llm_fix": False,
                "source_physical_index": 1,
            }
        else:
            synthetic_preface["text"] = append_text_block(synthetic_preface.get("text", ""), demoted_text)

    if synthetic_preface is not None:
        collapsed_nodes.insert(0, synthetic_preface)

    return collapsed_nodes


def add_preface_node_if_needed(flat_nodes):
    if not flat_nodes:
        return flat_nodes

    first_node = flat_nodes[0]
    first_start_index = first_node.get("start_index", first_node.get("physical_index"))
    if not isinstance(first_start_index, int) or first_start_index <= 1:
        return flat_nodes

    preface_level = first_node.get("corrected_level")
    if not isinstance(preface_level, int) or preface_level < 1:
        preface_level = 1

    preface_node = {
        "node_id": "preface_00",
        "corrected_level": preface_level,
        "title": PREFACE_TITLE,
        "start_index": 1,
        "physical_index": 1,
        "text": "",
    }

    if "original_level" in first_node:
        preface_node["original_level"] = first_node.get("original_level") or 1
    if "needs_llm_fix" in first_node:
        preface_node["needs_llm_fix"] = False
    if "source_physical_index" in first_node:
        preface_node["source_physical_index"] = 1

    return [preface_node] + list(flat_nodes)


def fill_preface_text_if_needed(flat_nodes, pdf_json_payload):
    if not flat_nodes or flat_nodes[0].get("node_id") != "preface_00":
        return flat_nodes

    if len(flat_nodes) == 1:
        return flat_nodes

    next_start_index = flat_nodes[1].get("start_index", flat_nodes[1].get("physical_index"))
    if not isinstance(next_start_index, int) or next_start_index <= 1:
        return flat_nodes

    page_text_map = build_pdf_page_text_map(pdf_json_payload)
    preface_end_index = next_start_index - 1
    preface_text = get_page_range_text(page_text_map, 1, preface_end_index)
    flat_nodes[0]["text"] = append_text_block(preface_text, flat_nodes[0].get("text", ""))
    return flat_nodes


def build_initial_flat_nodes(markdown_text, pdf_json_payload, default_page=1, logger=None):
    markdown_nodes, markdown_lines = extract_nodes_from_markdown(markdown_text)
    markdown_sections = extract_node_text_content(markdown_nodes, markdown_lines)

    hybrid_headings = build_hybrid_headings_from_markdown_and_json(
        markdown_text=markdown_text,
        pdf_json_payload=pdf_json_payload,
        markdown_sections=markdown_sections,
        default_page=default_page,
        logger=logger,
    )

    flat_nodes = []
    for index, item in enumerate(hybrid_headings, start=1):
        flat_nodes.append(
            {
                "node_id": str(index).zfill(3),
                "original_level": item["level"],
                "title": item["title"],
                "start_index": item["page_number"],
                "physical_index": item["page_number"],
                "text": item.get("md_text", ""),
                "line_num": item["line_num"],
                "needs_llm_fix": item["needs_llm_fix"],
                "source_physical_index": item["physical_index"],
            }
        )

    return flat_nodes


def attach_tree_metadata(tree_nodes, reconstructed_nodes):
    metadata_by_node_id = {node["node_id"]: node for node in reconstructed_nodes}

    def enrich(nodes):
        enriched_nodes = []
        for node in nodes:
            enriched = dict(node)
            metadata = metadata_by_node_id.get(node.get("node_id"), {})
            for field in ("line_num", "original_level", "corrected_level", "needs_llm_fix", "source_physical_index"):
                if field in metadata:
                    enriched[field] = metadata[field]
            enriched["nodes"] = enrich(node.get("nodes", []))
            enriched_nodes.append(enriched)
        return enriched_nodes

    return enrich(tree_nodes)


def build_hybrid_tree_pipeline(markdown_text, pdf_json_payload, total_pages=None, model=None, llm_fn=None, logger=None, progress_callback=None):
    logger = logger or logging.getLogger(__name__)
    derived_total_pages = total_pages or pdf_json_payload.get("number of pages") or 0
    if derived_total_pages < 1:
        raise ValueError("total_pages must be provided or derivable from pdf_json_payload")

    if progress_callback:
        progress_callback("aligning_headings", "Aligning markdown headings with PDF JSON")
    emit_debug_log(logger, "Building initial flat nodes from markdown and JSON")
    flat_nodes = build_initial_flat_nodes(markdown_text, pdf_json_payload, default_page=1, logger=logger)
    if not flat_nodes:
        dump_debug_json(f"{DEBUG_LOG_DIR}/debug_03_reconstructed_nodes.json", [], logger=logger)
        dump_debug_json(f"{DEBUG_LOG_DIR}/debug_04_initial_tree.json", [], logger=logger)
        return {
            "flat_nodes": [],
            "reconstructed_nodes": [],
            "tree": [],
            "total_pages": derived_total_pages,
        }

    if progress_callback:
        progress_callback("reconstructing_tree", "Reconstructing corrected hierarchy and page intervals")
    emit_debug_log(logger, "Reconstructing corrected levels with guardrails")
    reconstructed_nodes = reconstruct_tree_structure(flat_nodes, model=model, llm_fn=llm_fn, logger=logger)
    reconstructed_nodes = add_preface_node_if_needed(reconstructed_nodes)
    reconstructed_nodes = collapse_demoted_nodes(reconstructed_nodes)
    reconstructed_nodes = fill_preface_text_if_needed(reconstructed_nodes, pdf_json_payload)

    changed_level_nodes = []
    for node in reconstructed_nodes:
        original_level = node.get("original_level")
        corrected_level = node.get("corrected_level")
        if node.get("node_id") == "preface_00":
            print(f"Reconstruction added preface node: {node.get('title')}")
            changed_level_nodes.append(
                {
                    "title": node.get("title"),
                    "node_id": node.get("node_id"),
                    "original_level": original_level,
                    "corrected_level": corrected_level,
                }
            )
            continue
        if original_level != corrected_level:
            print(
                f"Reconstruction level change: {node.get('title')} | "
                f"{original_level} -> {corrected_level}"
            )
            changed_level_nodes.append(
                {
                    "title": node.get("title"),
                    "node_id": node.get("node_id"),
                    "original_level": original_level,
                    "corrected_level": corrected_level,
                }
            )

    emit_debug_log(
        logger,
        "Reconstruction completed",
        changed_level_nodes=changed_level_nodes,
        reconstructed_node_count=len(reconstructed_nodes),
    )
    dump_debug_json(f"{DEBUG_LOG_DIR}/debug_03_reconstructed_nodes.json", reconstructed_nodes, logger=logger)

    emit_debug_log(logger, "Building final tree and intervals")
    tree = build_tree_and_intervals(reconstructed_nodes, total_pages=derived_total_pages)
    tree = attach_tree_metadata(tree, reconstructed_nodes)
    top_level_titles = [node.get("title", "") for node in tree]
    print(f"Initial hybrid tree: top_level_count={len(tree)}, top_level_titles={top_level_titles}")
    emit_debug_log(
        logger,
        "Initial hybrid tree built",
        top_level_count=len(tree),
        top_level_titles=top_level_titles,
    )
    dump_debug_json(f"{DEBUG_LOG_DIR}/debug_04_initial_tree.json", tree, logger=logger)

    return {
        "flat_nodes": flat_nodes,
        "reconstructed_nodes": reconstructed_nodes,
        "tree": tree,
        "total_pages": derived_total_pages,
    }


__all__ = [
    "PREFACE_TITLE",
    "append_text_block",
    "add_preface_node_if_needed",
    "collapse_demoted_nodes",
    "fill_preface_text_if_needed",
    "build_initial_flat_nodes",
    "build_hybrid_tree_pipeline",
]

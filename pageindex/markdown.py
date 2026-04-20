import asyncio
import difflib
import json
import logging
import os
import re
import unicodedata
from collections import defaultdict

from .llm import count_tokens
from .tree_utils import (
    create_clean_structure_for_description,
    format_structure,
    generate_doc_description,
    generate_node_summary,
    print_json,
    print_toc,
    structure_to_list,
    write_node_id,
)


TOC_TITLE_PATTERNS = ("目录", "contents", "table of contents")
FALLBACK_HEADING_LEVELS = {
    "title": 1,
    "subtitle": 2,
    "heading": 3,
    "subheading": 4,
}
DEBUG_LOG_DIR = "logs"


def _make_json_serializable(data):
    if isinstance(data, dict):
        return {key: _make_json_serializable(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_make_json_serializable(item) for item in data]
    if isinstance(data, tuple):
        return [_make_json_serializable(item) for item in data]
    if isinstance(data, set):
        return sorted(_make_json_serializable(item) for item in data)
    if isinstance(data, os.PathLike):
        return os.fspath(data)
    return data


def emit_debug_log(logger, message, **payload):
    log_payload = {"message": message, **payload} if payload else {"message": message}
    if logger is None:
        logging.getLogger(__name__).info("%s", log_payload)
        return
    logger.info(log_payload)


def dump_debug_json(filepath, data, logger=None):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(_make_json_serializable(data), f, indent=2, ensure_ascii=False)
    emit_debug_log(logger, "Wrote debug JSON", path=filepath)


def normalize_title(text):
    if text is None:
        return ""
    normalized = []
    for char in str(text).strip().lower():
        if char.isspace():
            continue
        if unicodedata.category(char).startswith("P"):
            continue
        normalized.append(char)
    return "".join(normalized)


def clean_heading_title(text):
    if text is None:
        return ""
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    return cleaned


def normalize_probe_text(text):
    if text is None:
        return ""
    normalized = []
    for char in str(text):
        if char.isspace():
            continue
        if unicodedata.category(char).startswith("P"):
            continue
        normalized.append(char.lower())
    return "".join(normalized)


def resolve_hybrid_json_path(md_path, json_path=None):
    if json_path:
        if not os.path.isfile(json_path):
            raise ValueError(f"JSON file not found: {json_path}")
        return json_path

    candidate = os.path.splitext(md_path)[0] + ".json"
    if not os.path.isfile(candidate):
        raise ValueError(f"JSON file not found for markdown file: {candidate}")
    return candidate


def load_pdf_json_payload(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_pdf_heading_level(item):
    heading_level = item.get("heading level")
    if isinstance(heading_level, int):
        return heading_level
    if isinstance(heading_level, str) and heading_level.isdigit():
        return int(heading_level)

    level_name = str(item.get("level", "")).strip().lower()
    return FALLBACK_HEADING_LEVELS.get(level_name, 6)


def build_pdf_page_text_map(payload):
    page_chunks = defaultdict(list)
    for child in payload.get("kids", []):
        page_number = child.get("page number")
        content = str(child.get("content", "")).strip()
        if not isinstance(page_number, int) or not content:
            continue
        page_chunks[page_number].append(content)
    return {page: "\n".join(chunks) for page, chunks in sorted(page_chunks.items())}


def extract_headings_from_pdf_json(payload):
    headings = []
    for child in payload.get("kids", []):
        if child.get("type") != "heading":
            continue

        title = clean_heading_title(child.get("content", ""))
        normalized_title = normalize_title(title)
        page_number = child.get("page number")
        if not normalized_title or not isinstance(page_number, int):
            continue

        heading = {
            "title": title,
            "normalized_title": normalized_title,
            "level": parse_pdf_heading_level(child),
            "page_number": page_number,
        }

        if headings:
            previous = headings[-1]
            duplicate_on_same_page = (
                previous["normalized_title"] == heading["normalized_title"]
                and previous["level"] == heading["level"]
                and previous["page_number"] == heading["page_number"]
            )
            repeated_across_adjacent_pages = (
                previous["normalized_title"] == heading["normalized_title"]
                and previous["level"] == heading["level"]
                and heading["page_number"] == previous["page_number"] + 1
            )
            if duplicate_on_same_page or repeated_across_adjacent_pages:
                continue

        headings.append(heading)
    return headings


def extract_hybrid_toc_with_fallback(markdown_headings, json_headings=None, page_text_map=None):
    if isinstance(markdown_headings, str):
        markdown_text = markdown_headings
        pdf_json_payload = json_headings or {}
        markdown_nodes, markdown_lines = extract_nodes_from_markdown(markdown_text)
        markdown_headings = extract_node_text_content(markdown_nodes, markdown_lines)
        json_headings = extract_headings_from_pdf_json(pdf_json_payload)
        page_text_map = build_pdf_page_text_map(pdf_json_payload)

    markdown_headings = markdown_headings or []
    json_headings = json_headings or []
    page_text_map = page_text_map or {}

    unmatched_indices = set(range(len(json_headings)))
    normalized_page_text_map = {
        page_number: normalize_probe_text(text)
        for page_number, text in sorted(page_text_map.items())
        if isinstance(page_number, int) and text
    }

    result = []
    for markdown_heading in markdown_headings:
        title = markdown_heading.get("title", "")
        normalized_markdown_title = normalize_title(title)
        physical_index = None
        matched_index = None

        if normalized_markdown_title:
            for index, json_heading in enumerate(json_headings):
                if index not in unmatched_indices:
                    continue
                if json_heading.get("normalized_title") == normalized_markdown_title:
                    matched_index = index
                    physical_index = json_heading.get("page_number")
                    break

        if matched_index is None and normalized_markdown_title:
            for index, json_heading in enumerate(json_headings):
                if index not in unmatched_indices:
                    continue
                normalized_json_title = json_heading.get("normalized_title", "")
                if not normalized_json_title:
                    continue
                contains_match = (
                    normalized_markdown_title in normalized_json_title
                    or normalized_json_title in normalized_markdown_title
                )
                similarity = difflib.SequenceMatcher(
                    None,
                    normalized_markdown_title,
                    normalized_json_title,
                ).ratio()
                if contains_match or similarity > 0.85:
                    matched_index = index
                    physical_index = json_heading.get("page_number")
                    break

        if matched_index is None:
            raw_snippet = str(markdown_heading.get("md_text", "") or "")[:80]
            normalized_snippet = normalize_probe_text(raw_snippet)
            if len(normalized_snippet) > 10:
                for page_number, normalized_page_text in normalized_page_text_map.items():
                    if normalized_snippet in normalized_page_text:
                        physical_index = page_number
                        break

        if matched_index is not None:
            unmatched_indices.discard(matched_index)

        result.append(
            {
                "title": title,
                "level": markdown_heading.get("level"),
                "line_num": markdown_heading.get("line_num"),
                "physical_index": physical_index,
                "needs_llm_fix": physical_index is None,
            }
        )

    return result


def resolve_fallback_physical_indices(flat_toc_items, default_page=1):
    resolved_items = [dict(item) for item in flat_toc_items]

    for item in resolved_items:
        item["resolved_physical_index"] = item["physical_index"]

    index = 0
    while index < len(resolved_items):
        if resolved_items[index]["resolved_physical_index"] is not None:
            index += 1
            continue

        start = index
        while index < len(resolved_items) and resolved_items[index]["resolved_physical_index"] is None:
            index += 1
        end = index - 1

        previous_page = None
        for previous_index in range(start - 1, -1, -1):
            candidate = resolved_items[previous_index]["resolved_physical_index"]
            if candidate is not None:
                previous_page = candidate
                break

        next_page = None
        for next_index in range(index, len(resolved_items)):
            candidate = resolved_items[next_index]["resolved_physical_index"]
            if candidate is not None:
                next_page = candidate
                break

        missing_count = end - start + 1
        if previous_page is None and next_page is None:
            inferred_pages = [default_page] * missing_count
        elif previous_page is None:
            inferred_pages = [next_page] * missing_count
        elif next_page is None:
            inferred_pages = [previous_page] * missing_count
        else:
            gap = max(next_page - previous_page, 0)
            if gap == 0:
                inferred_pages = [previous_page] * missing_count
            else:
                inferred_pages = [
                    previous_page + ((gap * (offset + 1)) // (missing_count + 1))
                    for offset in range(missing_count)
                ]

        for offset, inferred_page in enumerate(inferred_pages):
            resolved_items[start + offset]["resolved_physical_index"] = inferred_page

    return resolved_items


def build_hybrid_headings_from_markdown_and_json(
    markdown_text,
    pdf_json_payload,
    markdown_sections,
    default_page=1,
    logger=None,
    debug_dir=DEBUG_LOG_DIR,
):
    json_headings = extract_headings_from_pdf_json(pdf_json_payload)
    page_text_map = build_pdf_page_text_map(pdf_json_payload)
    parsed_sources_path = os.path.join(debug_dir, "debug_01_parsed_sources.json")
    print(
        f"Dual source parsing: markdown headings={len(markdown_sections)}, "
        f"json headings={len(json_headings)}"
    )
    emit_debug_log(
        logger,
        "Dual source parsing completed",
        markdown_heading_count=len(markdown_sections),
        json_heading_count=len(json_headings),
    )
    dump_debug_json(
        parsed_sources_path,
        {
            "markdown_sections": markdown_sections,
            "json_headings": json_headings,
        },
        logger=logger,
    )

    flat_toc_items = extract_hybrid_toc_with_fallback(markdown_sections, json_headings, page_text_map)
    resolved_items = resolve_fallback_physical_indices(flat_toc_items, default_page=default_page)
    markdown_sections_by_line = {section["line_num"]: section for section in markdown_sections}

    hybrid_headings = []
    for item in resolved_items:
        heading = {
            "title": item["title"],
            "normalized_title": normalize_title(item["title"]),
            "level": item["level"],
            "line_num": item["line_num"],
            "page_number": item["resolved_physical_index"],
            "physical_index": item["physical_index"],
            "needs_llm_fix": item["needs_llm_fix"],
        }

        markdown_section = markdown_sections_by_line.get(item["line_num"])
        if markdown_section is not None:
            heading["md_text"] = markdown_section["text"]

        hybrid_headings.append(heading)

    precise_match_count = sum(1 for item in resolved_items if item["physical_index"] is not None and not item["needs_llm_fix"])
    inferred_match_count = sum(1 for item in resolved_items if item["physical_index"] is None or item["needs_llm_fix"])
    print(
        f"Hybrid alignment: precise_matches={precise_match_count}, "
        f"inferred_matches={inferred_match_count}"
    )
    emit_debug_log(
        logger,
        "Hybrid alignment completed",
        precise_match_count=precise_match_count,
        inferred_match_count=inferred_match_count,
        total_nodes=len(hybrid_headings),
    )
    dump_debug_json(
        os.path.join(debug_dir, "debug_02_aligned_flat_nodes.json"),
        {
            "resolved_items": resolved_items,
            "hybrid_headings": hybrid_headings,
        },
        logger=logger,
    )

    return hybrid_headings


def text_contains_toc_keyword(text):
    normalized = normalize_title(text)
    return any(pattern in normalized for pattern in ("目录", "contents", "tableofcontents"))


def page_looks_like_toc(text, known_titles=None):
    normalized = normalize_title(text)
    if not normalized:
        return False
    if text_contains_toc_keyword(text):
        return True
    if re.search(r"(?:\.{2,}|…{2,}|·{2,}|\-{2,})\s*\d+", text):
        return True
    if re.search(r"\b\d+\b", text):
        matched_titles = 0
        for title in known_titles or []:
            if title and len(title) > 1 and title in normalized:
                matched_titles += 1
                if matched_titles >= 2:
                    return True
    return False


def detect_toc_pages(page_text_map, headings):
    if not page_text_map:
        return []

    known_titles = [heading["normalized_title"] for heading in headings if len(heading["normalized_title"]) > 1]
    toc_start = None
    for page_number, text in page_text_map.items():
        if text_contains_toc_keyword(text):
            toc_start = page_number
            break

    if toc_start is None:
        return []

    toc_pages = []
    for page_number in sorted(page_text_map):
        if page_number < toc_start:
            continue
        text = page_text_map[page_number]
        if page_number == toc_start or page_looks_like_toc(text, known_titles=known_titles):
            toc_pages.append(page_number)
            continue
        if toc_pages:
            break
    return toc_pages


def is_toc_heading(title):
    normalized = normalize_title(title)
    return any(pattern in normalized for pattern in ("目录", "contents", "tableofcontents"))


def extract_toc_analysis(headings, page_text_map):
    toc_pages = detect_toc_pages(page_text_map, headings)
    toc_text = "\n".join(page_text_map[page] for page in toc_pages)
    content_headings = [
        heading for heading in headings if heading["page_number"] not in toc_pages and not is_toc_heading(heading["title"])
    ]

    if not content_headings:
        return {
            "toc_pages": toc_pages,
            "toc_detected": bool(toc_pages),
            "toc_complete": False,
            "major_level": None,
            "covered_major_indices": set(),
            "content_headings": content_headings,
        }

    major_level = min(heading["level"] for heading in content_headings)
    normalized_toc = normalize_title(toc_text)
    covered_major_indices = {
        index
        for index, heading in enumerate(content_headings)
        if heading["level"] == major_level and heading["normalized_title"] and heading["normalized_title"] in normalized_toc
    }
    total_major_count = sum(1 for heading in content_headings if heading["level"] == major_level)
    toc_complete = bool(total_major_count) and total_major_count == len(covered_major_indices)

    return {
        "toc_pages": toc_pages,
        "toc_detected": bool(toc_pages),
        "toc_complete": toc_complete,
        "major_level": major_level,
        "covered_major_indices": covered_major_indices,
        "content_headings": content_headings,
    }


def attach_markdown_chunks_to_headings(headings, markdown_nodes):
    cursor = 0
    for heading in headings:
        matched_node = None
        for index in range(cursor, len(markdown_nodes)):
            candidate = markdown_nodes[index]
            if normalize_title(candidate["title"]) == heading["normalized_title"]:
                matched_node = candidate
                cursor = index + 1
                break
        if matched_node is None:
            continue
        heading["line_num"] = matched_node["line_num"]
        heading["md_text"] = matched_node["text"]
    return headings


def get_page_range_text(page_text_map, start_page, end_page):
    texts = []
    for page_number in range(start_page, end_page + 1):
        text = page_text_map.get(page_number, "").strip()
        if text:
            texts.append(text)
    return "\n\n".join(texts).strip()


def make_orphan_node(start_page, end_page, page_text_map, title=None):
    node = {
        "title": title or f"Untitled section (pages {start_page}-{end_page})",
        "start_page": start_page,
        "end_page": end_page,
        "text": get_page_range_text(page_text_map, start_page, end_page),
    }
    return node


def build_root_segments(headings, major_level, covered_major_indices):
    if not headings:
        return []

    major_indices = [index for index, heading in enumerate(headings) if heading["level"] == major_level]
    if not major_indices:
        return [{"covered": False, "headings": headings}]

    segments = []
    if major_indices[0] > 0:
        segments.append({"covered": False, "headings": headings[: major_indices[0]]})

    for order, major_index in enumerate(major_indices):
        next_major_index = major_indices[order + 1] if order + 1 < len(major_indices) else len(headings)
        current_slice = headings[major_index:next_major_index]
        covered = major_index in covered_major_indices
        if segments and segments[-1]["covered"] == covered:
            segments[-1]["headings"].extend(current_slice)
        else:
            segments.append({"covered": covered, "headings": current_slice})
    return segments


def build_tree_from_hybrid_headings(headings, segment_end_page, page_text_map):
    if not headings:
        return []

    stack = []
    root_nodes = []
    flat_nodes = []

    for index, heading in enumerate(headings):
        next_page = headings[index + 1]["page_number"] if index + 1 < len(headings) else segment_end_page + 1
        if next_page > heading["page_number"]:
            end_page = next_page - 1
        else:
            end_page = heading["page_number"]

        node = {
            "title": heading["title"],
            "start_page": heading["page_number"],
            "end_page": max(end_page, heading["page_number"]),
            "text": heading.get("md_text") or get_page_range_text(page_text_map, heading["page_number"], max(end_page, heading["page_number"])),
            "nodes": [],
        }
        if heading.get("line_num") is not None:
            node["line_num"] = heading["line_num"]

        while stack and stack[-1][1] >= heading["level"]:
            stack.pop()

        if stack:
            stack[-1][0]["nodes"].append(node)
        else:
            root_nodes.append(node)
        stack.append((node, heading["level"]))
        flat_nodes.append(node)

    def extend_parent_ranges(nodes):
        max_page = 0
        for node in nodes:
            current_end = node["end_page"]
            if node.get("nodes"):
                child_end = extend_parent_ranges(node["nodes"])
                current_end = max(current_end, child_end)
                node["end_page"] = current_end
            max_page = max(max_page, current_end)
        return max_page

    extend_parent_ranges(root_nodes)
    return root_nodes


def build_hybrid_structure(content_headings, toc_analysis, total_pages, page_text_map, doc_name, full_markdown_text):
    if not content_headings:
        return [
            {
                "title": doc_name,
                "start_page": 1,
                "end_page": max(total_pages, 1),
                "text": full_markdown_text.strip(),
            }
        ]

    major_level = toc_analysis["major_level"] or min(heading["level"] for heading in content_headings)
    covered_major_indices = toc_analysis["covered_major_indices"] if toc_analysis["toc_detected"] else set()
    root_segments = build_root_segments(content_headings, major_level, covered_major_indices)

    structure = []
    current_page = 1
    for segment_index, segment in enumerate(root_segments):
        segment_headings = segment["headings"]
        if not segment_headings:
            continue

        segment_start = segment_headings[0]["page_number"]
        next_segment_start = (
            root_segments[segment_index + 1]["headings"][0]["page_number"]
            if segment_index + 1 < len(root_segments) and root_segments[segment_index + 1]["headings"]
            else total_pages + 1
        )
        segment_end = max(segment_start, next_segment_start - 1)

        if current_page < segment_start:
            structure.append(make_orphan_node(current_page, segment_start - 1, page_text_map))

        structure.extend(build_tree_from_hybrid_headings(segment_headings, segment_end, page_text_map))
        current_page = segment_end + 1

    if current_page <= total_pages:
        structure.append(make_orphan_node(current_page, total_pages, page_text_map))

    return structure

async def get_node_summary(node, summary_token_threshold=200, model=None):
    node_text = node.get('text')
    num_tokens = count_tokens(node_text, model=model)
    if num_tokens < summary_token_threshold:
        return node_text
    else:
        return await generate_node_summary(node, model=model)


async def _get_node_summary_with_limit(node, summary_token_threshold, model=None, semaphore=None):
    if semaphore is None:
        return await get_node_summary(node, summary_token_threshold=summary_token_threshold, model=model)
    async with semaphore:
        return await get_node_summary(node, summary_token_threshold=summary_token_threshold, model=model)


async def generate_summaries_for_structure_md(
    structure,
    summary_token_threshold,
    model=None,
    max_concurrency=None,
):
    nodes = structure_to_list(structure)
    normalized_concurrency = max(1, int(max_concurrency or len(nodes) or 1))
    semaphore = None if normalized_concurrency >= len(nodes) else asyncio.Semaphore(normalized_concurrency)
    tasks = [
        _get_node_summary_with_limit(
            node,
            summary_token_threshold=summary_token_threshold,
            model=model,
            semaphore=semaphore,
        )
        for node in nodes
    ]
    summaries = await asyncio.gather(*tasks)
    
    for node, summary in zip(nodes, summaries):
        if not node.get('nodes'):
            node['summary'] = summary
        else:
            node['prefix_summary'] = summary
    return structure


def extract_nodes_from_markdown(markdown_content):
    header_pattern = r'^(#{1,6})\s+(.+)$'
    code_block_pattern = r'^```'
    node_list = []
    
    lines = markdown_content.split('\n')
    in_code_block = False
    
    for line_num, line in enumerate(lines, 1):
        stripped_line = line.strip()
        
        # Check for code block delimiters (triple backticks)
        if re.match(code_block_pattern, stripped_line):
            in_code_block = not in_code_block
            continue
        
        # Skip empty lines
        if not stripped_line:
            continue
        
        # Only look for headers when not inside a code block
        if not in_code_block:
            match = re.match(header_pattern, stripped_line)
            if match:
                title = match.group(2).strip()
                node_list.append({'node_title': title, 'line_num': line_num})

    return node_list, lines


def extract_node_text_content(node_list, markdown_lines):    
    all_nodes = []
    for node in node_list:
        line_content = markdown_lines[node['line_num'] - 1]
        header_match = re.match(r'^(#{1,6})', line_content)
        
        if header_match is None:
            print(f"Warning: Line {node['line_num']} does not contain a valid header: '{line_content}'")
            continue
            
        processed_node = {
            'title': node['node_title'],
            'line_num': node['line_num'],
            'level': len(header_match.group(1))
        }
        all_nodes.append(processed_node)
    
    for i, node in enumerate(all_nodes):
        start_line = node['line_num'] - 1 
        if i + 1 < len(all_nodes):
            end_line = all_nodes[i + 1]['line_num'] - 1 
        else:
            end_line = len(markdown_lines)
        
        node['text'] = '\n'.join(markdown_lines[start_line:end_line]).strip()    
    return all_nodes

def update_node_list_with_text_token_count(node_list, model=None):

    def find_all_children(parent_index, parent_level, node_list):
        """Find all direct and indirect children of a parent node"""
        children_indices = []
        
        # Look for children after the parent
        for i in range(parent_index + 1, len(node_list)):
            current_level = node_list[i]['level']
            
            # If we hit a node at same or higher level than parent, stop
            if current_level <= parent_level:
                break
                
            # This is a descendant
            children_indices.append(i)
        
        return children_indices
    
    # Make a copy to avoid modifying the original
    result_list = node_list.copy()
    
    # Process nodes from end to beginning to ensure children are processed before parents
    for i in range(len(result_list) - 1, -1, -1):
        current_node = result_list[i]
        current_level = current_node['level']
        
        # Get all children of this node
        children_indices = find_all_children(i, current_level, result_list)
        
        # Start with the node's own text
        node_text = current_node.get('text', '')
        total_text = node_text
        
        # Add all children's text
        for child_index in children_indices:
            child_text = result_list[child_index].get('text', '')
            if child_text:
                total_text += '\n' + child_text
        
        # Calculate token count for combined text
        result_list[i]['text_token_count'] = count_tokens(total_text, model=model)
    
    return result_list


def tree_thinning_for_index(node_list, min_node_token=None, model=None):
    def find_all_children(parent_index, parent_level, node_list):
        children_indices = []
        
        for i in range(parent_index + 1, len(node_list)):
            current_level = node_list[i]['level']
            
            if current_level <= parent_level:
                break
                
            children_indices.append(i)
        
        return children_indices
    
    result_list = node_list.copy()
    nodes_to_remove = set()
    
    for i in range(len(result_list) - 1, -1, -1):
        if i in nodes_to_remove:
            continue
            
        current_node = result_list[i]
        current_level = current_node['level']
        
        total_tokens = current_node.get('text_token_count', 0)
        
        if total_tokens < min_node_token:
            children_indices = find_all_children(i, current_level, result_list)
            
            children_texts = []
            for child_index in sorted(children_indices):
                if child_index not in nodes_to_remove:
                    child_text = result_list[child_index].get('text', '')
                    if child_text.strip():
                        children_texts.append(child_text)
                    nodes_to_remove.add(child_index)
            
            if children_texts:
                parent_text = current_node.get('text', '')
                merged_text = parent_text
                for child_text in children_texts:
                    if merged_text and not merged_text.endswith('\n'):
                        merged_text += '\n\n'
                    merged_text += child_text
                
                result_list[i]['text'] = merged_text
                
                result_list[i]['text_token_count'] = count_tokens(merged_text, model=model)
    
    for index in sorted(nodes_to_remove, reverse=True):
        result_list.pop(index)
    
    return result_list


def build_tree_from_nodes(node_list):
    if not node_list:
        return []
    
    stack = []
    root_nodes = []
    node_counter = 1
    
    for node in node_list:
        current_level = node['level']
        
        tree_node = {
            'title': node['title'],
            'node_id': str(node_counter).zfill(4),
            'text': node['text'],
            'line_num': node['line_num'],
            'nodes': []
        }
        node_counter += 1
        
        while stack and stack[-1][1] >= current_level:
            stack.pop()
        
        if not stack:
            root_nodes.append(tree_node)
        else:
            parent_node, parent_level = stack[-1]
            parent_node['nodes'].append(tree_node)
        
        stack.append((tree_node, current_level))
    
    return root_nodes


def clean_tree_for_output(tree_nodes):
    cleaned_nodes = []
    
    for node in tree_nodes:
        cleaned_node = {
            'title': node['title'],
            'node_id': node['node_id'],
            'text': node['text'],
            'line_num': node['line_num']
        }
        
        if node['nodes']:
            cleaned_node['nodes'] = clean_tree_for_output(node['nodes'])
        
        cleaned_nodes.append(cleaned_node)
    
    return cleaned_nodes


async def md_to_tree(
    md_path,
    if_thinning=False,
    min_token_threshold=None,
    if_add_node_summary='no',
    summary_token_threshold=None,
    model=None,
    if_add_doc_description='no',
    if_add_node_text='no',
    if_add_node_id='yes',
    summary_max_concurrency=None,
):
    with open(md_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    line_count = markdown_content.count('\n') + 1

    print(f"Extracting nodes from markdown...")
    node_list, markdown_lines = extract_nodes_from_markdown(markdown_content)

    print(f"Extracting text content from nodes...")
    nodes_with_content = extract_node_text_content(node_list, markdown_lines)
    
    if if_thinning:
        nodes_with_content = update_node_list_with_text_token_count(nodes_with_content, model=model)
        print(f"Thinning nodes...")
        nodes_with_content = tree_thinning_for_index(nodes_with_content, min_token_threshold, model=model)
    
    print(f"Building tree from nodes...")
    tree_structure = build_tree_from_nodes(nodes_with_content)

    if if_add_node_id == 'yes':
        write_node_id(tree_structure)

    print(f"Formatting tree structure...")
    
    if if_add_node_summary == 'yes':
        # Always include text for summary generation
        tree_structure = format_structure(tree_structure, order = ['title', 'node_id', 'line_num', 'summary', 'prefix_summary', 'text', 'nodes'])
        
        print(f"Generating summaries for each node...")
        tree_structure = await generate_summaries_for_structure_md(
            tree_structure,
            summary_token_threshold=summary_token_threshold,
            model=model,
            max_concurrency=summary_max_concurrency,
        )
        
        if if_add_node_text == 'no':
            # Remove text after summary generation if not requested
            tree_structure = format_structure(tree_structure, order = ['title', 'node_id', 'line_num', 'summary', 'prefix_summary', 'nodes'])
        
        if if_add_doc_description == 'yes':
            print(f"Generating document description...")
            # Create a clean structure without unnecessary fields for description generation
            clean_structure = create_clean_structure_for_description(tree_structure)
            doc_description = generate_doc_description(clean_structure, model=model)
            return {
                'doc_name': os.path.splitext(os.path.basename(md_path))[0],
                'doc_description': doc_description,
                'line_count': line_count,
                'structure': tree_structure,
            }
    else:
        # No summaries needed, format based on text preference
        if if_add_node_text == 'yes':
            tree_structure = format_structure(tree_structure, order = ['title', 'node_id', 'line_num', 'summary', 'prefix_summary', 'text', 'nodes'])
        else:
            tree_structure = format_structure(tree_structure, order = ['title', 'node_id', 'line_num', 'summary', 'prefix_summary', 'nodes'])
    
    return {
        'doc_name': os.path.splitext(os.path.basename(md_path))[0],
        'line_count': line_count,
        'structure': tree_structure,
    }


async def md_to_tree_hybrid(
    md_path,
    json_path=None,
    if_thinning=False,
    min_token_threshold=None,
    if_add_node_summary='no',
    summary_token_threshold=None,
    model=None,
    if_add_doc_description='no',
    if_add_node_text='no',
    if_add_node_id='yes',
    summary_max_concurrency=None,
):
    del if_thinning
    del min_token_threshold
    logger = logging.getLogger(__name__)

    resolved_json_path = resolve_hybrid_json_path(md_path, json_path=json_path)
    with open(md_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    line_count = markdown_content.count('\n') + 1

    pdf_json_payload = load_pdf_json_payload(resolved_json_path)
    page_text_map = build_pdf_page_text_map(pdf_json_payload)
    total_pages = pdf_json_payload.get("number of pages") or max(page_text_map.keys(), default=1)
    doc_name = os.path.splitext(os.path.basename(md_path))[0]

    print("Extracting markdown sections...")
    markdown_nodes, markdown_lines = extract_nodes_from_markdown(markdown_content)
    markdown_sections = extract_node_text_content(markdown_nodes, markdown_lines)

    print("Extracting heading flow from Markdown with JSON fallback...")
    hybrid_headings = build_hybrid_headings_from_markdown_and_json(
        markdown_text=markdown_content,
        pdf_json_payload=pdf_json_payload,
        markdown_sections=markdown_sections,
        default_page=1,
        logger=logger,
    )

    print("Analyzing table of contents coverage...")
    toc_analysis = extract_toc_analysis(hybrid_headings, page_text_map)
    content_headings = toc_analysis["content_headings"]

    print("Building hybrid tree structure...")
    tree_structure = build_hybrid_structure(
        content_headings=content_headings,
        toc_analysis=toc_analysis,
        total_pages=total_pages,
        page_text_map=page_text_map,
        doc_name=doc_name,
        full_markdown_text=markdown_content,
    )

    if if_add_node_id == 'yes':
        write_node_id(tree_structure)

    top_level_titles = [node.get("title", "") for node in tree_structure]
    print(f"Initial hybrid tree: top_level_count={len(tree_structure)}, top_level_titles={top_level_titles}")
    emit_debug_log(
        logger,
        "Initial hybrid tree built",
        top_level_count=len(tree_structure),
        top_level_titles=top_level_titles,
    )
    dump_debug_json(
        os.path.join(DEBUG_LOG_DIR, "debug_04_initial_tree.json"),
        tree_structure,
        logger=logger,
    )

    print("Formatting hybrid tree structure...")
    field_order = ['title', 'node_id', 'start_page', 'end_page', 'line_num', 'summary', 'prefix_summary', 'text', 'nodes']
    if if_add_node_summary == 'yes':
        tree_structure = format_structure(tree_structure, order=field_order)
        print("Generating summaries for hybrid nodes...")
        tree_structure = await generate_summaries_for_structure_md(
            tree_structure,
            summary_token_threshold=summary_token_threshold,
            model=model,
            max_concurrency=summary_max_concurrency,
        )
        if if_add_node_text == 'no':
            tree_structure = format_structure(
                tree_structure,
                order=['title', 'node_id', 'start_page', 'end_page', 'line_num', 'summary', 'prefix_summary', 'nodes'],
            )

        if if_add_doc_description == 'yes':
            print("Generating hybrid document description...")
            clean_structure = create_clean_structure_for_description(tree_structure)
            doc_description = generate_doc_description(clean_structure, model=model)
            return {
                'doc_name': doc_name,
                'doc_description': doc_description,
                'line_count': line_count,
                'structure': tree_structure,
            }
    else:
        if if_add_node_text == 'yes':
            tree_structure = format_structure(tree_structure, order=field_order)
        else:
            tree_structure = format_structure(
                tree_structure,
                order=['title', 'node_id', 'start_page', 'end_page', 'line_num', 'summary', 'prefix_summary', 'nodes'],
            )

    return {
        'doc_name': doc_name,
        'line_count': line_count,
        'structure': tree_structure,
    }


if __name__ == "__main__":
    import os
    import json
    
    # MD_NAME = 'Detect-Order-Construct'
    MD_NAME = 'cognitive-load'
    MD_PATH = os.path.join(os.path.dirname(__file__), '..', 'sample_data', 'documents', f'{MD_NAME}.md')


    MODEL="gpt-4.1"
    IF_THINNING=False
    THINNING_THRESHOLD=5000
    SUMMARY_TOKEN_THRESHOLD=200
    IF_SUMMARY=True

    tree_structure = asyncio.run(md_to_tree(
        md_path=MD_PATH, 
        if_thinning=IF_THINNING, 
        min_token_threshold=THINNING_THRESHOLD, 
        if_add_node_summary='yes' if IF_SUMMARY else 'no', 
        summary_token_threshold=SUMMARY_TOKEN_THRESHOLD, 
        model=MODEL))
    
    print('\n' + '='*60)
    print('TREE STRUCTURE')
    print('='*60)
    print_json(tree_structure)

    print('\n' + '='*60)
    print('TABLE OF CONTENTS')
    print('='*60)
    print_toc(tree_structure['structure'])

    output_path = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'results', f'{MD_NAME}_structure.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tree_structure, f, indent=2, ensure_ascii=False)
    
    print(f"\nTree structure saved to: {output_path}")

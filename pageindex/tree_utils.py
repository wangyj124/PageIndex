import asyncio
import copy
import json
import textwrap

from .llm import count_tokens, llm_acompletion, llm_completion
from .pdf import get_text_of_pdf_pages, get_text_of_pdf_pages_with_labels


def write_node_id(data, node_id=0):
    if isinstance(data, dict):
        data["node_id"] = str(node_id).zfill(4)
        node_id += 1
        for key in list(data.keys()):
            if "nodes" in key:
                node_id = write_node_id(data[key], node_id)
    elif isinstance(data, list):
        for index in range(len(data)):
            node_id = write_node_id(data[index], node_id)
    return node_id


def get_nodes(structure):
    if isinstance(structure, dict):
        structure_node = copy.deepcopy(structure)
        structure_node.pop("nodes", None)
        nodes = [structure_node]
        for key in list(structure.keys()):
            if "nodes" in key:
                nodes.extend(get_nodes(structure[key]))
        return nodes
    if isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(get_nodes(item))
        return nodes
    return []


def structure_to_list(structure):
    if isinstance(structure, dict):
        nodes = [structure]
        if "nodes" in structure:
            nodes.extend(structure_to_list(structure["nodes"]))
        return nodes
    if isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(structure_to_list(item))
        return nodes
    return []


def get_leaf_nodes(structure):
    if isinstance(structure, dict):
        if not structure["nodes"]:
            structure_node = copy.deepcopy(structure)
            structure_node.pop("nodes", None)
            return [structure_node]
        leaf_nodes = []
        for key in list(structure.keys()):
            if "nodes" in key:
                leaf_nodes.extend(get_leaf_nodes(structure[key]))
        return leaf_nodes
    if isinstance(structure, list):
        leaf_nodes = []
        for item in structure:
            leaf_nodes.extend(get_leaf_nodes(item))
        return leaf_nodes
    return []


def is_leaf_node(data, node_id):
    def find_node(node_data, target_node_id):
        if isinstance(node_data, dict):
            if node_data.get("node_id") == target_node_id:
                return node_data
            for key in node_data.keys():
                if "nodes" in key:
                    result = find_node(node_data[key], target_node_id)
                    if result:
                        return result
        elif isinstance(node_data, list):
            for item in node_data:
                result = find_node(item, target_node_id)
                if result:
                    return result
        return None

    node = find_node(data, node_id)
    return bool(node and not node.get("nodes"))


def get_last_node(structure):
    return structure[-1]


def list_to_tree(data):
    def get_parent_structure(structure):
        if not structure:
            return None
        parts = str(structure).split(".")
        return ".".join(parts[:-1]) if len(parts) > 1 else None

    nodes = {}
    root_nodes = []
    for item in data:
        structure = item.get("structure")
        node = {
            "title": item.get("title"),
            "start_index": item.get("start_index"),
            "end_index": item.get("end_index"),
            "nodes": [],
        }
        nodes[structure] = node
        parent_structure = get_parent_structure(structure)
        if parent_structure:
            if parent_structure in nodes:
                nodes[parent_structure]["nodes"].append(node)
            else:
                root_nodes.append(node)
        else:
            root_nodes.append(node)

    def clean_node(node):
        if not node["nodes"]:
            del node["nodes"]
        else:
            for child in node["nodes"]:
                clean_node(child)
        return node

    return [clean_node(node) for node in root_nodes]


def add_preface_if_needed(data):
    if not isinstance(data, list) or not data:
        return data
    if data[0]["physical_index"] is not None and data[0]["physical_index"] > 1:
        data.insert(
            0,
            {
                "structure": "0",
                "title": "Preface",
                "physical_index": 1,
            },
        )
    return data


def post_processing(structure, end_physical_index):
    for i, item in enumerate(structure):
        item["start_index"] = item.get("physical_index")
        if i < len(structure) - 1:
            if structure[i + 1].get("appear_start") == "yes":
                item["end_index"] = structure[i + 1]["physical_index"] - 1
            else:
                item["end_index"] = structure[i + 1]["physical_index"]
        else:
            item["end_index"] = end_physical_index
    tree = list_to_tree(structure)
    if len(tree) != 0:
        return tree
    for node in structure:
        node.pop("appear_start", None)
        node.pop("physical_index", None)
    return structure


def build_tree_and_intervals(flat_nodes, total_pages):
    if total_pages is None or total_pages < 1:
        raise ValueError("total_pages must be a positive integer")

    prepared_nodes = []
    for item in flat_nodes:
        start_index = item.get("physical_index")
        corrected_level = item.get("corrected_level")
        if not isinstance(start_index, int) or start_index < 1:
            raise ValueError(f"Invalid physical_index for node_id={item.get('node_id')}")
        if not isinstance(corrected_level, int) or corrected_level < 1:
            raise ValueError(f"Invalid corrected_level for node_id={item.get('node_id')}")

        prepared_nodes.append(
            {
                "node_id": item.get("node_id"),
                "title": item.get("title"),
                "start_index": start_index,
                "end_index": None,
                "text": item.get("text"),
                "nodes": [],
                "_level": corrected_level,
            }
        )

    root_nodes = []
    stack = []

    def close_node(node, closing_start_index):
        closed_end_index = closing_start_index - 1
        if closed_end_index < node["start_index"]:
            closed_end_index = node["start_index"]
        node["end_index"] = closed_end_index

    for node in prepared_nodes:
        current_level = node["_level"]
        while stack and stack[-1][0]["_level"] >= current_level:
            closing_node = stack.pop()[0]
            close_node(closing_node, node["start_index"])

        if stack:
            stack[-1][0]["nodes"].append(node)
        else:
            root_nodes.append(node)

        stack.append((node, current_level))

    while stack:
        closing_node = stack.pop()[0]
        closing_node["end_index"] = total_pages

    def finalize(nodes):
        finalized = []
        for node in nodes:
            finalized_node = {
                "node_id": node.get("node_id"),
                "title": node.get("title"),
                "start_index": node.get("start_index"),
                "end_index": node.get("end_index"),
                "text": node.get("text"),
                "nodes": finalize(node.get("nodes", [])),
            }
            finalized.append(finalized_node)
        return finalized

    return finalize(root_nodes)


def clean_structure_post(data):
    if isinstance(data, dict):
        data.pop("page_number", None)
        data.pop("start_index", None)
        data.pop("end_index", None)
        if "nodes" in data:
            clean_structure_post(data["nodes"])
    elif isinstance(data, list):
        for section in data:
            clean_structure_post(section)
    return data


def remove_fields(data, fields=None):
    fields = fields or ["text"]
    if isinstance(data, dict):
        return {k: remove_fields(v, fields) for k, v in data.items() if k not in fields}
    if isinstance(data, list):
        return [remove_fields(item, fields) for item in data]
    return data


def print_toc(tree, indent=0):
    for node in tree:
        print("  " * indent + node["title"])
        if node.get("nodes"):
            print_toc(node["nodes"], indent + 1)


def print_json(data, max_len=40, indent=2):
    def simplify_data(obj):
        if isinstance(obj, dict):
            return {k: simplify_data(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [simplify_data(item) for item in obj]
        if isinstance(obj, str) and len(obj) > max_len:
            return obj[:max_len] + "..."
        return obj

    print(json.dumps(simplify_data(data), indent=indent, ensure_ascii=False))


def remove_structure_text(data):
    if isinstance(data, dict):
        data.pop("text", None)
        if "nodes" in data:
            remove_structure_text(data["nodes"])
    elif isinstance(data, list):
        for item in data:
            remove_structure_text(item)
    return data


def check_token_limit(structure, limit=110000):
    flattened = structure_to_list(structure)
    for node in flattened:
        num_tokens = count_tokens(node["text"], model=None)
        if num_tokens > limit:
            print(f"Node ID: {node['node_id']} has {num_tokens} tokens")
            print("Start Index:", node["start_index"])
            print("End Index:", node["end_index"])
            print("Title:", node["title"])
            print("\n")


def convert_physical_index_to_int(data):
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "physical_index" in item and isinstance(item["physical_index"], str):
                if item["physical_index"].startswith("<physical_index_"):
                    item["physical_index"] = int(item["physical_index"].split("_")[-1].rstrip(">").strip())
                elif item["physical_index"].startswith("physical_index_"):
                    item["physical_index"] = int(item["physical_index"].split("_")[-1].strip())
    elif isinstance(data, str):
        if data.startswith("<physical_index_"):
            data = int(data.split("_")[-1].rstrip(">").strip())
        elif data.startswith("physical_index_"):
            data = int(data.split("_")[-1].strip())
        return data if isinstance(data, int) else None
    return data


def convert_page_to_int(data):
    for item in data:
        if "page" in item and isinstance(item["page"], str):
            try:
                item["page"] = int(item["page"])
            except ValueError:
                pass
    return data


def add_node_text(node, pdf_pages):
    if isinstance(node, dict):
        start_page = node.get("start_index")
        end_page = node.get("end_index")
        node["text"] = get_text_of_pdf_pages(pdf_pages, start_page, end_page)
        if "nodes" in node:
            add_node_text(node["nodes"], pdf_pages)
    elif isinstance(node, list):
        for index in range(len(node)):
            add_node_text(node[index], pdf_pages)


def add_node_text_with_labels(node, pdf_pages):
    if isinstance(node, dict):
        start_page = node.get("start_index")
        end_page = node.get("end_index")
        node["text"] = get_text_of_pdf_pages_with_labels(pdf_pages, start_page, end_page)
        if "nodes" in node:
            add_node_text_with_labels(node["nodes"], pdf_pages)
    elif isinstance(node, list):
        for index in range(len(node)):
            add_node_text_with_labels(node[index], pdf_pages)


async def generate_node_summary(node, model=None):
    prompt = f"""你将获得文档的一部分内容，你的任务是生成该部分文档的描述，说明该部分文档涵盖了哪些主要观点。

    部分文档文本：{node['text']}
    
    直接返回描述，不要包含任何其他文本。
    """
    return await llm_acompletion(model, prompt)


async def generate_summaries_for_structure(structure, model=None):
    nodes = structure_to_list(structure)
    summaries = await asyncio.gather(*[generate_node_summary(node, model=model) for node in nodes])
    for node, summary in zip(nodes, summaries):
        node["summary"] = summary
    return structure


def create_clean_structure_for_description(structure):
    if isinstance(structure, dict):
        clean_node = {}
        for key in ["title", "node_id", "summary", "prefix_summary"]:
            if key in structure:
                clean_node[key] = structure[key]
        if "nodes" in structure and structure["nodes"]:
            clean_node["nodes"] = create_clean_structure_for_description(structure["nodes"])
        return clean_node
    if isinstance(structure, list):
        return [create_clean_structure_for_description(item) for item in structure]
    return structure


def generate_doc_description(structure, model=None):
    prompt = f"""你是生成文档描述的专家。
    你将获得文档的结构。你的任务是为该文档生成一句话的描述，使其易于与其他文档区分。
        
    文档结构：{structure}
    
    直接返回描述，不要包含任何其他文本。
    """
    return llm_completion(model, prompt)


def reorder_dict(data, key_order):
    if not key_order:
        return data
    return {key: data[key] for key in key_order if key in data}


def format_structure(structure, order=None):
    if not order:
        return structure
    if isinstance(structure, dict):
        if "nodes" in structure:
            structure["nodes"] = format_structure(structure["nodes"], order)
        if not structure.get("nodes"):
            structure.pop("nodes", None)
        return reorder_dict(structure, order)
    if isinstance(structure, list):
        return [format_structure(item, order) for item in structure]
    return structure


def create_node_mapping(tree):
    mapping = {}

    def _traverse(nodes):
        for node in nodes:
            if node.get("node_id"):
                mapping[node["node_id"]] = node
            if node.get("nodes"):
                _traverse(node["nodes"])

    _traverse(tree)
    return mapping


def print_tree(tree, indent=0):
    for node in tree:
        summary = node.get("summary") or node.get("prefix_summary", "")
        summary_str = f"  - {summary[:60]}..." if summary else ""
        print("  " * indent + f"[{node.get('node_id', '?')}] {node.get('title', '')}{summary_str}")
        if node.get("nodes"):
            print_tree(node["nodes"], indent + 1)


def print_wrapped(text, width=100):
    for line in text.splitlines():
        print(textwrap.fill(line, width=width))


__all__ = [
    "write_node_id",
    "get_nodes",
    "structure_to_list",
    "get_leaf_nodes",
    "is_leaf_node",
    "get_last_node",
    "list_to_tree",
    "add_preface_if_needed",
    "post_processing",
    "build_tree_and_intervals",
    "clean_structure_post",
    "remove_fields",
    "print_toc",
    "print_json",
    "remove_structure_text",
    "check_token_limit",
    "convert_physical_index_to_int",
    "convert_page_to_int",
    "add_node_text",
    "add_node_text_with_labels",
    "generate_node_summary",
    "generate_summaries_for_structure",
    "create_clean_structure_for_description",
    "generate_doc_description",
    "reorder_dict",
    "format_structure",
    "create_node_mapping",
    "print_tree",
    "print_wrapped",
]

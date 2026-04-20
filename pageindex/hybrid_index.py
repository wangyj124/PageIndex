import asyncio
from pathlib import Path

from .hybrid_pipeline import build_hybrid_tree_pipeline
from .logging_utils import emit_progress_event
from .markdown import (
    build_pdf_page_text_map,
    generate_summaries_for_structure_md,
    load_pdf_json_payload,
)
from .tree_utils import create_clean_structure_for_description, format_structure, generate_doc_description


def require_opendataloader_pdf():
    try:
        import opendataloader_pdf  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Hybrid PDF indexing requires 'opendataloader-pdf'. "
            "Please activate the PageIndex environment and install dependencies first."
        ) from exc


def prepare_hybrid_sources_from_pdf(pdf_path, output_dir, progress_callback=None, progress_logger=None):
    require_opendataloader_pdf()
    import opendataloader_pdf

    pdf_path = Path(pdf_path)
    source_dir = Path(output_dir) / "_hybrid_sources" / pdf_path.stem
    source_dir.mkdir(parents=True, exist_ok=True)
    emit_progress_event(
        "converting_pdf",
        "Converting PDF to markdown/json with opendataloader-pdf",
        doc_name=pdf_path.name,
        extra={"output_dir": str(source_dir)},
        progress_callback=progress_callback,
        progress_logger=progress_logger,
    )
    opendataloader_pdf.convert(
        input_path=[str(pdf_path)],
        output_dir=str(source_dir),
        format="markdown,json",
        quiet=True,
    )

    md_path = source_dir / f"{pdf_path.stem}.md"
    json_path = source_dir / f"{pdf_path.stem}.json"
    if not md_path.is_file() or not json_path.is_file():
        raise ValueError(
            f"opendataloader-pdf did not produce expected files: md={md_path.is_file()}, json={json_path.is_file()}"
        )
    return md_path, json_path


def build_pdf_pages_from_json_payload(pdf_json_payload):
    page_text_map = build_pdf_page_text_map(pdf_json_payload)
    return [{"page": page_number, "content": content} for page_number, content in sorted(page_text_map.items())]


def rename_hybrid_intervals_to_pages(data):
    if isinstance(data, list):
        return [rename_hybrid_intervals_to_pages(item) for item in data]
    if isinstance(data, dict):
        renamed = {}
        for key, value in data.items():
            if key == "start_index":
                renamed["start_page"] = value
            elif key == "end_index":
                renamed["end_page"] = value
            elif key == "nodes":
                renamed["nodes"] = rename_hybrid_intervals_to_pages(value)
            else:
                renamed[key] = rename_hybrid_intervals_to_pages(value)
        return renamed
    return data


def finalize_hybrid_payload(
    tree_result,
    source_path,
    line_count,
    opt,
    summary_token_threshold,
    progress_callback=None,
    progress_logger=None,
):
    source_path = Path(source_path)
    doc_name = source_path.stem
    tree_structure = rename_hybrid_intervals_to_pages(tree_result["tree"])
    if opt.if_add_node_id != "yes":
        for node in tree_structure:
            node.pop("node_id", None)

    with_node_id_order = ["title", "node_id", "start_page", "end_page", "line_num", "summary", "prefix_summary", "text", "nodes"]
    without_node_id_order = ["title", "start_page", "end_page", "line_num", "summary", "prefix_summary", "text", "nodes"]
    full_field_order = with_node_id_order if opt.if_add_node_id == "yes" else without_node_id_order
    compact_field_order = [field for field in full_field_order if field != "text"]

    if opt.if_add_node_summary == "yes":
        emit_progress_event(
            "generating_summaries",
            "Generating hybrid node summaries",
            doc_name=source_path.name,
            progress_callback=progress_callback,
            progress_logger=progress_logger,
        )
        tree_structure = format_structure(tree_structure, order=full_field_order)
        tree_structure = asyncio.run(
            generate_summaries_for_structure_md(
                tree_structure,
                summary_token_threshold=summary_token_threshold,
                model=opt.model,
                max_concurrency=getattr(opt, "summary_max_concurrency", None),
            )
        )
        if opt.if_add_node_text == "no":
            tree_structure = format_structure(tree_structure, order=compact_field_order)

        if opt.if_add_doc_description == "yes":
            clean_structure = create_clean_structure_for_description(tree_structure)
            doc_description = generate_doc_description(clean_structure, model=opt.model)
            return {
                "doc_name": doc_name,
                "doc_description": doc_description,
                "line_count": line_count,
                "structure": tree_structure,
            }
    else:
        if opt.if_add_node_text == "yes":
            tree_structure = format_structure(tree_structure, order=full_field_order)
        else:
            tree_structure = format_structure(tree_structure, order=compact_field_order)

    return {
        "doc_name": doc_name,
        "line_count": line_count,
        "structure": tree_structure,
    }


def run_hybrid_pipeline_for_sources(
    source_path,
    md_path,
    json_path,
    opt,
    summary_token_threshold,
    progress_callback=None,
    progress_logger=None,
):
    source_path = Path(source_path)
    emit_progress_event(
        "loading_hybrid_sources",
        "Loading markdown and JSON hybrid sources",
        doc_name=source_path.name,
        extra={"md_path": str(md_path), "json_path": str(json_path)},
        progress_callback=progress_callback,
        progress_logger=progress_logger,
    )
    markdown_text = Path(md_path).read_text(encoding="utf-8")
    line_count = markdown_text.count("\n") + 1
    pdf_json_payload = load_pdf_json_payload(str(json_path))
    tree_result = build_hybrid_tree_pipeline(
        markdown_text,
        pdf_json_payload,
        model=opt.model,
        logger=progress_logger,
        progress_callback=lambda stage, message, extra=None: emit_progress_event(
            stage,
            message,
            doc_name=source_path.name,
            extra=extra,
            progress_callback=progress_callback,
            progress_logger=progress_logger,
        ),
    )
    payload = finalize_hybrid_payload(
        tree_result,
        source_path,
        line_count,
        opt,
        summary_token_threshold,
        progress_callback=progress_callback,
        progress_logger=progress_logger,
    )
    return payload, pdf_json_payload


__all__ = [
    "build_pdf_pages_from_json_payload",
    "finalize_hybrid_payload",
    "prepare_hybrid_sources_from_pdf",
    "rename_hybrid_intervals_to_pages",
    "require_opendataloader_pdf",
    "run_hybrid_pipeline_for_sources",
]

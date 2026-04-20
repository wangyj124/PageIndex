import asyncio
import json
import shutil
import uuid
from pathlib import Path

import pytest

from pageindex.markdown import (
    extract_hybrid_toc_with_fallback,
    extract_nodes_from_markdown,
    generate_summaries_for_structure_md,
    md_to_tree,
    md_to_tree_hybrid,
    normalize_title,
)


def make_temp_dir():
    path = Path("artifacts/test-tmp") / f"md-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json_payload(path, total_pages, kids):
    path.write_text(
        json.dumps(
            {
                "file name": "demo.pdf",
                "number of pages": total_pages,
                "kids": kids,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def test_extract_nodes_ignores_code_block_headers():
    markdown = "# Title\n\n```md\n## Hidden\n```\n\n## Visible\n"
    nodes, _ = extract_nodes_from_markdown(markdown)
    assert [node["node_title"] for node in nodes] == ["Title", "Visible"]


def test_normalize_title_removes_whitespace_and_punctuation():
    assert normalize_title(" 1. Heading (Part A)! ") == "1headingparta"


def test_extract_hybrid_toc_with_fallback_matches_markdown_to_json():
    markdown = "# General\n\nBody\n\n## 1. Delivery Requirements\n\nMore body\n"
    payload = {
        "kids": [
            {"type": "heading", "page number": 3, "content": "General"},
            {"type": "paragraph", "page number": 3, "content": "Body"},
            {"type": "heading", "page number": 12, "content": "1 Delivery Requirements"},
        ]
    }

    result = extract_hybrid_toc_with_fallback(markdown, payload)

    assert result == [
        {
            "title": "General",
            "level": 1,
            "line_num": 1,
            "physical_index": 3,
            "needs_llm_fix": False,
        },
        {
            "title": "1. Delivery Requirements",
            "level": 2,
            "line_num": 5,
            "physical_index": 12,
            "needs_llm_fix": False,
        },
    ]


def test_extract_hybrid_toc_with_fallback_supports_fuzzy_title_matching():
    markdown_headings = [
        {
            "title": "Chapter One Requirements",
            "level": 1,
            "line_num": 1,
            "md_text": "# Chapter One Requirements\n\nDelivery terms.",
        }
    ]
    json_headings = [
        {
            "title": "Chapter 1 Requirements",
            "normalized_title": normalize_title("Chapter 1 Requirements"),
            "page_number": 7,
        }
    ]

    result = extract_hybrid_toc_with_fallback(markdown_headings, json_headings, {})

    assert result == [
        {
            "title": "Chapter One Requirements",
            "level": 1,
            "line_num": 1,
            "physical_index": 7,
            "needs_llm_fix": False,
        }
    ]


def test_extract_hybrid_toc_with_fallback_uses_content_probing():
    markdown_headings = [
        {
            "title": "Unmatched Heading",
            "level": 2,
            "line_num": 5,
            "md_text": "The supplier shall deliver all goods within ten business days after notice.",
        }
    ]
    page_text_map = {
        3: "Cover page text",
        8: "The supplier shall deliver all goods within ten business days after notice and acceptance.",
    }

    result = extract_hybrid_toc_with_fallback(markdown_headings, [], page_text_map)

    assert result[0]["physical_index"] == 8
    assert result[0]["needs_llm_fix"] is False


def test_extract_hybrid_toc_with_fallback_marks_missing_titles_for_llm_fix():
    markdown = "# Chapter One\n\n## Missing Page Anchor\n"
    payload = {
        "kids": [
            {"type": "heading", "page number": 2, "content": "Chapter One"},
        ]
    }

    result = extract_hybrid_toc_with_fallback(markdown, payload)

    assert result[0]["physical_index"] == 2
    assert result[0]["needs_llm_fix"] is False
    assert result[1]["physical_index"] is None
    assert result[1]["needs_llm_fix"] is True


def test_extract_hybrid_toc_with_fallback_keeps_markdown_order_when_json_order_conflicts():
    markdown = "# Chapter One\n\n## Chapter Two\n"
    payload = {
        "kids": [
            {"type": "heading", "page number": 9, "content": "Chapter Two"},
            {"type": "heading", "page number": 4, "content": "Chapter One"},
        ]
    }

    result = extract_hybrid_toc_with_fallback(markdown, payload)

    assert [item["title"] for item in result] == ["Chapter One", "Chapter Two"]
    assert [item["physical_index"] for item in result] == [4, 9]


def test_md_to_tree_smoke():
    temp_dir = make_temp_dir()
    try:
        md_path = temp_dir / "demo.md"
        md_path.write_text("# Title\n\n## Section\n\nhello\n", encoding="utf-8")

        result = asyncio.run(
            md_to_tree(
                md_path=str(md_path),
                if_thinning=False,
                if_add_node_summary="no",
                if_add_doc_description="no",
                if_add_node_text="yes",
                if_add_node_id="yes",
            )
        )

        assert result["doc_name"] == "demo"
        assert result["structure"][0]["title"] == "Title"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_md_to_tree_hybrid_auto_discovers_same_name_json():
    temp_dir = make_temp_dir()
    try:
        md_path = temp_dir / "demo.md"
        json_path = temp_dir / "demo.json"
        md_path.write_text("# Chapter One\n\nbody\n\n# Chapter Two\n\nmore\n", encoding="utf-8")
        write_json_payload(
            json_path,
            total_pages=2,
            kids=[
                {"type": "heading", "page number": 1, "heading level": 1, "content": "Chapter One"},
                {"type": "paragraph", "page number": 1, "content": "body"},
                {"type": "heading", "page number": 2, "heading level": 1, "content": "Chapter Two"},
                {"type": "paragraph", "page number": 2, "content": "more"},
            ],
        )

        result = asyncio.run(
            md_to_tree_hybrid(
                md_path=str(md_path),
                if_add_node_summary="no",
                if_add_doc_description="no",
                if_add_node_text="yes",
                if_add_node_id="yes",
            )
        )

        assert [node["title"] for node in result["structure"]] == ["Chapter One", "Chapter Two"]
        assert result["structure"][0]["start_page"] == 1
        assert result["structure"][1]["start_page"] == 2
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_md_to_tree_hybrid_raises_when_json_missing():
    temp_dir = make_temp_dir()
    try:
        md_path = temp_dir / "missing.md"
        md_path.write_text("# Title\n", encoding="utf-8")

        with pytest.raises(ValueError, match="JSON file not found"):
            asyncio.run(
                md_to_tree_hybrid(
                    md_path=str(md_path),
                    if_add_node_summary="no",
                    if_add_doc_description="no",
                    if_add_node_text="yes",
                    if_add_node_id="yes",
                )
            )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_md_to_tree_hybrid_merges_cross_page_repeated_heading():
    temp_dir = make_temp_dir()
    try:
        md_path = temp_dir / "demo.md"
        json_path = temp_dir / "demo.json"
        md_path.write_text("# Chapter One\n\npart a\n\n# Chapter Two\n\npart b\n", encoding="utf-8")
        write_json_payload(
            json_path,
            total_pages=3,
            kids=[
                {"type": "heading", "page number": 1, "heading level": 1, "content": "Chapter One"},
                {"type": "paragraph", "page number": 1, "content": "part a"},
                {"type": "heading", "page number": 2, "heading level": 1, "content": "Chapter One"},
                {"type": "paragraph", "page number": 2, "content": "continued"},
                {"type": "heading", "page number": 3, "heading level": 1, "content": "Chapter Two"},
                {"type": "paragraph", "page number": 3, "content": "part b"},
            ],
        )

        result = asyncio.run(
            md_to_tree_hybrid(
                md_path=str(md_path),
                json_path=str(json_path),
                if_add_node_summary="no",
                if_add_doc_description="no",
                if_add_node_text="yes",
                if_add_node_id="yes",
            )
        )

        assert [node["title"] for node in result["structure"]] == ["Chapter One", "Chapter Two"]
        assert result["structure"][0]["start_page"] == 1
        assert result["structure"][0]["end_page"] == 2
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_md_to_tree_hybrid_creates_orphan_block_without_heading():
    temp_dir = make_temp_dir()
    try:
        md_path = temp_dir / "demo.md"
        json_path = temp_dir / "demo.json"
        md_path.write_text("plain intro\n\n# Chapter One\n\nbody\n", encoding="utf-8")
        write_json_payload(
            json_path,
            total_pages=2,
            kids=[
                {"type": "paragraph", "page number": 1, "content": "plain intro"},
                {"type": "heading", "page number": 2, "heading level": 1, "content": "Chapter One"},
                {"type": "paragraph", "page number": 2, "content": "body"},
            ],
        )

        result = asyncio.run(
            md_to_tree_hybrid(
                md_path=str(md_path),
                json_path=str(json_path),
                if_add_node_summary="no",
                if_add_doc_description="no",
                if_add_node_text="yes",
                if_add_node_id="yes",
            )
        )

        assert result["structure"][0]["title"] == "Untitled section (pages 1-1)"
        assert result["structure"][0]["start_page"] == 1
        assert result["structure"][1]["title"] == "Chapter One"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_md_to_tree_hybrid_incomplete_toc_keeps_prefix_content():
    temp_dir = make_temp_dir()
    try:
        md_path = temp_dir / "demo.md"
        json_path = temp_dir / "demo.json"
        md_path.write_text(
            "# Volume I\n\nintro\n\n## Section A\n\ncontent\n\n# Volume II\n\nnext\n",
            encoding="utf-8",
        )
        write_json_payload(
            json_path,
            total_pages=5,
            kids=[
                {"type": "heading", "page number": 1, "heading level": 1, "content": "Volume I"},
                {"type": "paragraph", "page number": 1, "content": "intro"},
                {"type": "heading", "page number": 2, "heading level": 2, "content": "Section A"},
                {"type": "paragraph", "page number": 2, "content": "content"},
                {"type": "heading", "page number": 3, "heading level": 1, "content": "contents"},
                {"type": "paragraph", "page number": 3, "content": "Volume II ........ 5"},
                {"type": "heading", "page number": 5, "heading level": 1, "content": "Volume II"},
                {"type": "paragraph", "page number": 5, "content": "next"},
            ],
        )

        result = asyncio.run(
            md_to_tree_hybrid(
                md_path=str(md_path),
                json_path=str(json_path),
                if_add_node_summary="no",
                if_add_doc_description="no",
                if_add_node_text="no",
                if_add_node_id="yes",
            )
        )

        titles = [node["title"] for node in result["structure"]]
        assert "contents" not in titles
        assert titles[:2] == ["Volume I", "Volume II"]
        assert result["structure"][0]["nodes"][0]["title"] == "Section A"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_md_to_tree_hybrid_prefers_markdown_order_over_json_order():
    temp_dir = make_temp_dir()
    try:
        md_path = temp_dir / "demo.md"
        json_path = temp_dir / "demo.json"
        md_path.write_text("# Alpha\n\nbody\n\n# Beta\n\nmore\n", encoding="utf-8")
        write_json_payload(
            json_path,
            total_pages=5,
            kids=[
                {"type": "heading", "page number": 5, "content": "Beta"},
                {"type": "heading", "page number": 2, "content": "Alpha"},
            ],
        )

        result = asyncio.run(
            md_to_tree_hybrid(
                md_path=str(md_path),
                json_path=str(json_path),
                if_add_node_summary="no",
                if_add_doc_description="no",
                if_add_node_text="no",
                if_add_node_id="yes",
            )
        )

        titles = [node["title"] for node in result["structure"]]
        assert titles[-2:] == ["Alpha", "Beta"]
        assert result["structure"][-2]["start_page"] == 2
        assert result["structure"][-1]["start_page"] == 5
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_md_to_tree_hybrid_keeps_missing_markdown_heading_in_tree():
    temp_dir = make_temp_dir()
    try:
        md_path = temp_dir / "demo.md"
        json_path = temp_dir / "demo.json"
        md_path.write_text("# Alpha\n\n## Missing Child\n\ntext\n\n# Omega\n\nend\n", encoding="utf-8")
        write_json_payload(
            json_path,
            total_pages=4,
            kids=[
                {"type": "heading", "page number": 1, "content": "Alpha"},
                {"type": "heading", "page number": 4, "content": "Omega"},
            ],
        )

        result = asyncio.run(
            md_to_tree_hybrid(
                md_path=str(md_path),
                json_path=str(json_path),
                if_add_node_summary="no",
                if_add_doc_description="no",
                if_add_node_text="no",
                if_add_node_id="yes",
            )
        )

        assert [node["title"] for node in result["structure"]] == ["Alpha", "Omega"]
        assert result["structure"][0]["nodes"][0]["title"] == "Missing Child"
        assert result["structure"][0]["nodes"][0]["start_page"] == 2
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_md_to_tree_hybrid_writes_debug_dumps(monkeypatch):
    temp_dir = make_temp_dir().resolve()
    try:
        monkeypatch.chdir(temp_dir)
        md_path = temp_dir / "demo.md"
        json_path = temp_dir / "demo.json"
        md_path.write_text("# Chapter One\n\nbody\n\n# Chapter Two\n\nmore\n", encoding="utf-8")
        write_json_payload(
            json_path,
            total_pages=2,
            kids=[
                {"type": "heading", "page number": 1, "heading level": 1, "content": "Chapter One"},
                {"type": "paragraph", "page number": 1, "content": "body"},
                {"type": "heading", "page number": 2, "heading level": 1, "content": "Chapter Two"},
                {"type": "paragraph", "page number": 2, "content": "more"},
            ],
        )

        asyncio.run(
            md_to_tree_hybrid(
                md_path=str(md_path),
                json_path=str(json_path),
                if_add_node_summary="no",
                if_add_doc_description="no",
                if_add_node_text="no",
                if_add_node_id="yes",
            )
        )

        assert (temp_dir / "logs" / "debug_01_parsed_sources.json").is_file()
        assert (temp_dir / "logs" / "debug_02_aligned_flat_nodes.json").is_file()
        assert (temp_dir / "logs" / "debug_04_initial_tree.json").is_file()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_generate_summaries_for_structure_md_respects_max_concurrency(monkeypatch):
    structure = [
        {"title": "A", "text": "alpha"},
        {"title": "B", "text": "beta"},
        {"title": "C", "text": "gamma"},
        {"title": "D", "text": "delta"},
    ]
    state = {"active": 0, "peak": 0}

    async def fake_get_node_summary(node, summary_token_threshold=200, model=None):
        del summary_token_threshold, model
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0.01)
        state["active"] -= 1
        return f"summary:{node['title']}"

    monkeypatch.setattr("pageindex.markdown.get_node_summary", fake_get_node_summary)

    result = asyncio.run(
        generate_summaries_for_structure_md(
            structure,
            summary_token_threshold=1,
            model="openai/test-model",
            max_concurrency=2,
        )
    )

    assert state["peak"] == 2
    assert [node["summary"] for node in result] == [
        "summary:A",
        "summary:B",
        "summary:C",
        "summary:D",
    ]

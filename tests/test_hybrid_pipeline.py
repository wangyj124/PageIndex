import json
import logging

from pageindex.hybrid_pipeline import (
    add_preface_node_if_needed,
    build_hybrid_tree_pipeline,
    build_initial_flat_nodes,
    collapse_demoted_nodes,
    fill_preface_text_if_needed,
)


def make_payload(total_pages, kids):
    return {
        "file name": "demo.pdf",
        "number of pages": total_pages,
        "kids": kids,
    }


def test_build_initial_flat_nodes_preserves_markdown_order_and_fallback_fields():
    markdown_text = "# Alpha\n\nbody\n\n## Missing Child\n\nmore\n\n# Omega\n\nend\n"
    payload = make_payload(
        4,
        [
            {"type": "heading", "page number": 1, "content": "Alpha"},
            {"type": "heading", "page number": 4, "content": "Omega"},
        ],
    )

    flat_nodes = build_initial_flat_nodes(markdown_text, payload)

    assert [node["title"] for node in flat_nodes] == ["Alpha", "Missing Child", "Omega"]
    assert [node["physical_index"] for node in flat_nodes] == [1, 2, 4]
    assert flat_nodes[1]["needs_llm_fix"] is True
    assert flat_nodes[1]["source_physical_index"] is None
    assert flat_nodes[1]["original_level"] == 2


def test_build_hybrid_tree_pipeline_preserves_leading_contract_number_before_first_page_heading():
    leading_text = "买方合同编号：BUY-001 卖方合同编号：SELL-002"
    markdown_text = f"{leading_text}\n\n# 机岛设备采购合同\n\n正文\n"
    payload = make_payload(
        1,
        [
            {"type": "paragraph", "page number": 1, "content": leading_text},
            {"type": "heading", "page number": 1, "content": "机岛设备采购合同"},
            {"type": "paragraph", "page number": 1, "content": "正文"},
        ],
    )

    result = build_hybrid_tree_pipeline(
        markdown_text,
        payload,
        llm_fn=lambda model, prompt, chat_history=None: json.dumps(
            [{"node_id": "001", "corrected_level": 1, "decision_reason": "Contract title."}]
        ),
        cover_classifier_fn=lambda model, prompt: '{"is_contract_cover": false}',
    )

    assert result["tree"][0]["title"] == "首页前导信息"
    assert result["tree"][0]["text"] == leading_text
    assert result["tree"][0]["start_index"] == 1
    assert result["tree"][0]["end_index"] == 1
    assert result["tree"][1]["title"] == "机岛设备采购合同"


def test_build_hybrid_tree_pipeline_names_detected_cover_node_contract_cover():
    leading_text = "买方合同编号：BUY-001 卖方合同编号：SELL-002"
    markdown_text = f"{leading_text}\n\n# 机岛设备采购合同\n"
    payload = make_payload(
        1,
        [
            {"type": "paragraph", "page number": 1, "content": leading_text},
            {"type": "heading", "page number": 1, "content": "机岛设备采购合同"},
        ],
    )

    result = build_hybrid_tree_pipeline(
        markdown_text,
        payload,
        llm_fn=lambda model, prompt, chat_history=None: json.dumps(
            [{"node_id": "001", "corrected_level": 1, "decision_reason": "Contract title."}]
        ),
        cover_classifier_fn=lambda model, prompt: '{"is_contract_cover": true}',
    )

    assert result["tree"][0]["title"] == "合同封面"
    assert result["tree"][1]["title"] == "机岛设备采购合同"


def test_build_hybrid_tree_pipeline_keeps_non_cover_leading_text_without_cover_llm_for_long_page():
    leading_text = "内部说明：本页包含业务正文，不是封面。"
    long_body = "履约说明" * 160
    markdown_text = f"{leading_text}\n\n# 执行说明\n\n{long_body}\n"
    payload = make_payload(
        1,
        [
            {"type": "paragraph", "page number": 1, "content": leading_text},
            {"type": "heading", "page number": 1, "content": "执行说明"},
            {"type": "paragraph", "page number": 1, "content": long_body},
        ],
    )

    def fail_if_called(model, prompt):
        raise AssertionError("Long first pages must not invoke cover classification.")

    result = build_hybrid_tree_pipeline(
        markdown_text,
        payload,
        llm_fn=lambda model, prompt, chat_history=None: json.dumps(
            [{"node_id": "001", "corrected_level": 1, "decision_reason": "Section."}]
        ),
        cover_classifier_fn=fail_if_called,
    )

    assert result["tree"][0]["title"] == "首页前导信息"
    assert result["tree"][0]["text"] == leading_text
    assert result["tree"][1]["title"] == "执行说明"


def test_build_hybrid_tree_pipeline_creates_first_page_node_without_any_heading():
    first_page_text = "买方合同编号：BUY-001\n卖方合同编号：SELL-002"
    payload = make_payload(1, [{"type": "paragraph", "page number": 1, "content": first_page_text}])

    result = build_hybrid_tree_pipeline(
        first_page_text,
        payload,
        cover_classifier_fn=lambda model, prompt: '{"is_contract_cover": false}',
    )

    assert result["flat_nodes"] == []
    assert result["tree"] == [
        {
            "node_id": "front_page_00",
            "title": "首页前导信息",
            "start_index": 1,
            "end_index": 1,
            "text": first_page_text,
            "nodes": [],
        }
    ]


def test_add_preface_node_if_needed_handles_empty_list():
    assert add_preface_node_if_needed([]) == []


def test_add_preface_node_if_needed_inserts_preface_when_first_node_starts_after_page_one():
    flat_nodes = [
        {
            "node_id": "001",
            "corrected_level": 1,
            "title": "Volume One",
            "start_index": 3,
            "physical_index": 3,
            "text": "...",
        }
    ]

    result = add_preface_node_if_needed(flat_nodes)

    assert result[0]["node_id"] == "preface_00"
    assert result[0]["corrected_level"] == 1
    assert result[0]["start_index"] == 1
    assert result[0]["physical_index"] == 1
    assert result[0]["text"] == ""
    assert result[1:] == flat_nodes


def test_add_preface_node_if_needed_skips_when_first_node_already_starts_at_page_one():
    flat_nodes = [
        {
            "node_id": "001",
            "corrected_level": 1,
            "title": "Volume One",
            "start_index": 1,
            "physical_index": 1,
            "text": "...",
        }
    ]

    result = add_preface_node_if_needed(flat_nodes)

    assert result == flat_nodes


def test_fill_preface_text_if_needed_populates_preface_pages():
    payload = make_payload(
        5,
        [
            {"type": "paragraph", "page number": 1, "content": "Cover page"},
            {"type": "paragraph", "page number": 2, "content": "Disclaimer page"},
            {"type": "heading", "page number": 3, "content": "Volume One"},
        ],
    )
    flat_nodes = [
        {
            "node_id": "preface_00",
            "corrected_level": 1,
            "title": "Preface",
            "start_index": 1,
            "physical_index": 1,
            "text": "",
        },
        {
            "node_id": "001",
            "corrected_level": 1,
            "title": "Volume One",
            "start_index": 3,
            "physical_index": 3,
            "text": "Body",
        },
    ]

    result = fill_preface_text_if_needed(flat_nodes, payload)

    assert result[0]["text"] == "Cover page\n\nDisclaimer page"


def test_collapse_demoted_nodes_merges_into_previous_structural_node():
    flat_nodes = [
        {
            "node_id": "001",
            "corrected_level": 1,
            "title": "Volume One",
            "start_index": 2,
            "physical_index": 2,
            "text": "# Volume One\n\nintro",
        },
        {
            "node_id": "002",
            "corrected_level": -1,
            "title": "Fake Heading",
            "start_index": 3,
            "physical_index": 3,
            "text": "### Fake Heading\n\nThis should become body text.",
        },
        {
            "node_id": "003",
            "corrected_level": 2,
            "title": "Chapter One",
            "start_index": 4,
            "physical_index": 4,
            "text": "## Chapter One\n\nbody",
        },
    ]

    result = collapse_demoted_nodes(flat_nodes)

    assert [node["node_id"] for node in result] == ["001", "003"]
    assert "This should become body text." in result[0]["text"]
    assert result[1]["title"] == "Chapter One"


def test_collapse_demoted_nodes_creates_preface_for_leading_demoted_content():
    flat_nodes = [
        {
            "node_id": "001",
            "corrected_level": -1,
            "title": "Front Note",
            "start_index": 1,
            "physical_index": 1,
            "text": "Important notice before the first real heading.",
        },
        {
            "node_id": "002",
            "corrected_level": 1,
            "title": "Volume One",
            "start_index": 3,
            "physical_index": 3,
            "text": "# Volume One\n\nintro",
        },
    ]

    result = collapse_demoted_nodes(flat_nodes)

    assert result[0]["node_id"] == "preface_00"
    assert result[0]["physical_index"] == 1
    assert "Important notice" in result[0]["text"]
    assert result[1]["node_id"] == "002"


def test_build_hybrid_tree_pipeline_builds_full_tree():
    markdown_text = (
        "# Volume One\n\nintro\n\n"
        "## Chapter One\n\nbody one\n\n"
        "## Chapter Two\n\nbody two\n\n"
        "# Volume Two\n\nnext\n"
    )
    payload = make_payload(
        12,
        [
            {"type": "paragraph", "page number": 1, "content": "Cover note"},
            {"type": "heading", "page number": 2, "content": "Volume One"},
            {"type": "heading", "page number": 2, "content": "Chapter One"},
            {"type": "heading", "page number": 5, "content": "Chapter Two"},
            {"type": "heading", "page number": 10, "content": "Volume Two"},
        ],
    )
    llm_response = json.dumps(
        [
            {"node_id": "001", "corrected_level": 1, "reasoning": "Volume root."},
            {"node_id": "002", "corrected_level": 2, "reasoning": "Chapter child."},
            {"node_id": "003", "corrected_level": 2, "reasoning": "Chapter child."},
            {"node_id": "004", "corrected_level": 1, "reasoning": "Second volume root."},
        ]
    )

    result = build_hybrid_tree_pipeline(
        markdown_text,
        payload,
        llm_fn=lambda model, prompt, chat_history=None: llm_response,
    )

    assert [node["title"] for node in result["flat_nodes"]] == [
        "Volume One",
        "Chapter One",
        "Chapter Two",
        "Volume Two",
    ]
    assert [node["node_id"] for node in result["reconstructed_nodes"]] == [
        "preface_00",
        "001",
        "002",
        "003",
        "004",
    ]
    assert [node["corrected_level"] for node in result["reconstructed_nodes"]] == [1, 1, 2, 2, 1]
    assert result["tree"][0]["node_id"] == "preface_00"
    assert result["tree"][0]["text"] == "Cover note"
    assert result["tree"][1]["title"] == "Volume One"
    assert [node["title"] for node in result["tree"][1]["nodes"]] == ["Chapter One", "Chapter Two"]
    assert result["tree"][2]["title"] == "Volume Two"


def test_build_hybrid_tree_pipeline_collapses_killed_nodes_before_tree_building():
    markdown_text = "# Volume One\n\nintro\n\n## Fake Heading\n\nloose body\n\n## Chapter One\n\nbody one\n"
    payload = make_payload(
        6,
        [
            {"type": "paragraph", "page number": 1, "content": "Cover note"},
            {"type": "heading", "page number": 2, "content": "Volume One"},
            {"type": "heading", "page number": 4, "content": "Chapter One"},
        ],
    )
    llm_response = json.dumps(
        [
            {"node_id": "001", "corrected_level": 1, "decision_reason": "Volume root."},
            {"node_id": "002", "corrected_level": -1, "decision_reason": "This is only emphasized body text."},
            {"node_id": "003", "corrected_level": 2, "decision_reason": "Real chapter."},
        ]
    )

    result = build_hybrid_tree_pipeline(
        markdown_text,
        payload,
        llm_fn=lambda model, prompt, chat_history=None: llm_response,
    )

    assert [node["node_id"] for node in result["reconstructed_nodes"]] == ["preface_00", "001", "003"]
    assert [node["title"] for node in result["tree"][1]["nodes"]] == ["Chapter One"]
    assert "loose body" in result["tree"][1]["text"]


def test_build_hybrid_tree_pipeline_writes_debug_artifacts(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    markdown_text = "# Volume One\n\nintro\n\n## Chapter One\n\nbody one\n"
    payload = make_payload(
        4,
        [
            {"type": "paragraph", "page number": 1, "content": "Cover note"},
            {"type": "heading", "page number": 2, "content": "Volume One"},
            {"type": "heading", "page number": 3, "content": "Chapter One"},
        ],
    )
    llm_response = json.dumps(
        [
            {"node_id": "001", "corrected_level": 1, "decision_reason": "Volume root."},
            {"node_id": "002", "corrected_level": 2, "decision_reason": "Chapter child."},
        ]
    )

    build_hybrid_tree_pipeline(
        markdown_text,
        payload,
        llm_fn=lambda model, prompt, chat_history=None: llm_response,
    )

    assert (tmp_path / "logs" / "debug_01_parsed_sources.json").is_file()
    assert (tmp_path / "logs" / "debug_02_aligned_flat_nodes.json").is_file()
    assert (tmp_path / "logs" / "debug_03_reconstructed_nodes.json").is_file()
    assert (tmp_path / "logs" / "debug_04_initial_tree.json").is_file()


def test_build_hybrid_tree_pipeline_writes_debug_artifacts_to_custom_dir(tmp_path):
    markdown_text = "# Volume One\n\nintro\n\n## Chapter One\n\nbody one\n"
    payload = make_payload(
        4,
        [
            {"type": "paragraph", "page number": 1, "content": "Cover note"},
            {"type": "heading", "page number": 2, "content": "Volume One"},
            {"type": "heading", "page number": 3, "content": "Chapter One"},
        ],
    )
    llm_response = json.dumps(
        [
            {"node_id": "001", "corrected_level": 1, "decision_reason": "Volume root."},
            {"node_id": "002", "corrected_level": 2, "decision_reason": "Chapter child."},
        ]
    )
    debug_dir = tmp_path / "task-output" / "logs"

    build_hybrid_tree_pipeline(
        markdown_text,
        payload,
        llm_fn=lambda model, prompt, chat_history=None: llm_response,
        debug_dir=str(debug_dir),
    )

    assert (debug_dir / "debug_01_parsed_sources.json").is_file()
    assert (debug_dir / "debug_02_aligned_flat_nodes.json").is_file()
    assert (debug_dir / "debug_03_reconstructed_nodes.json").is_file()
    assert (debug_dir / "debug_04_initial_tree.json").is_file()


def test_build_hybrid_tree_pipeline_logs_reconstruction_messages_without_stdout(capsys, caplog):
    markdown_text = "# Volume One\n\nintro\n\n## Chapter One\n\nbody one\n"
    payload = make_payload(
        4,
        [
            {"type": "paragraph", "page number": 1, "content": "Cover note"},
            {"type": "heading", "page number": 2, "content": "Volume One"},
            {"type": "heading", "page number": 3, "content": "Chapter One"},
        ],
    )
    llm_response = json.dumps(
        [
            {"node_id": "001", "corrected_level": 1, "decision_reason": "Volume root."},
            {"node_id": "002", "corrected_level": 2, "decision_reason": "Chapter child."},
        ]
    )
    logger = logging.getLogger("tests.hybrid_pipeline")

    with caplog.at_level(logging.DEBUG, logger="tests.hybrid_pipeline"):
        build_hybrid_tree_pipeline(
            markdown_text,
            payload,
            llm_fn=lambda model, prompt, chat_history=None: llm_response,
            logger=logger,
        )

    stdout = capsys.readouterr().out
    assert "Reconstruction added preface node" not in stdout
    assert "Reconstruction level change" not in stdout
    assert "Initial hybrid tree: top_level_count=" not in stdout
    assert "Reconstruction added preface node" in caplog.text
    assert "Initial hybrid tree: top_level_count=2" in caplog.text

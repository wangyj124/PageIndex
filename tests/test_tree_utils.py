from pageindex.tree_utils import build_tree_and_intervals, normalize_tree_retrieval_segments


def test_build_tree_and_intervals_creates_expected_tree_and_ranges():
    flat_nodes = [
        {"node_id": "001", "corrected_level": 1, "title": "第一卷", "physical_index": 2, "text": "卷一正文"},
        {"node_id": "002", "corrected_level": 2, "title": "第一章", "physical_index": 2, "text": "第一章正文"},
        {"node_id": "003", "corrected_level": 2, "title": "第二章", "physical_index": 5, "text": "第二章正文"},
        {"node_id": "004", "corrected_level": 1, "title": "第二卷", "physical_index": 10, "text": "卷二正文"},
    ]

    tree = build_tree_and_intervals(flat_nodes, total_pages=12)

    assert tree == [
        {
            "node_id": "001",
            "title": "第一卷",
            "start_index": 2,
            "end_index": 9,
            "text": "卷一正文",
            "nodes": [
                {
                    "node_id": "002",
                    "title": "第一章",
                    "start_index": 2,
                    "end_index": 4,
                    "text": "第一章正文",
                    "nodes": [],
                },
                {
                    "node_id": "003",
                    "title": "第二章",
                    "start_index": 5,
                    "end_index": 9,
                    "text": "第二章正文",
                    "nodes": [],
                },
            ],
        },
        {
            "node_id": "004",
            "title": "第二卷",
            "start_index": 10,
            "end_index": 12,
            "text": "卷二正文",
            "nodes": [],
        },
    ]


def test_build_tree_and_intervals_closes_same_page_sibling_to_own_start():
    flat_nodes = [
        {"node_id": "001", "corrected_level": 1, "title": "第一卷", "physical_index": 2, "text": "卷一"},
        {"node_id": "002", "corrected_level": 2, "title": "第一章", "physical_index": 2, "text": "章一"},
        {"node_id": "003", "corrected_level": 2, "title": "第二章", "physical_index": 2, "text": "章二"},
    ]

    tree = build_tree_and_intervals(flat_nodes, total_pages=6)

    assert tree[0]["start_index"] == 2
    assert tree[0]["end_index"] == 6
    assert tree[0]["nodes"][0]["start_index"] == 2
    assert tree[0]["nodes"][0]["end_index"] == 2
    assert tree[0]["nodes"][1]["start_index"] == 2
    assert tree[0]["nodes"][1]["end_index"] == 6


def test_build_tree_and_intervals_rejects_invalid_input():
    flat_nodes = [
        {"node_id": "001", "corrected_level": 0, "title": "坏节点", "physical_index": 1, "text": ""},
    ]

    try:
        build_tree_and_intervals(flat_nodes, total_pages=3)
    except ValueError as exc:
        assert "corrected_level" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid corrected_level")


def test_normalize_tree_retrieval_segments_uses_parent_to_child_prefix_range():
    tree = [
        {
            "title": "父章节",
            "start_page": 16,
            "end_page": 20,
            "nodes": [
                {"title": "子章节一", "start_page": 16, "end_page": 17, "nodes": []},
                {"title": "子章节二", "start_page": 18, "end_page": 20, "nodes": []},
            ],
        }
    ]

    normalize_tree_retrieval_segments(
        tree,
        page_text_getter=lambda start, end: f"p{start}-{end}",
        max_pages_per_segment=5,
        start_key="start_page",
        end_key="end_page",
    )

    parent = tree[0]
    first_child = parent["nodes"][0]
    second_child = parent["nodes"][1]
    assert parent["start_page"] == 16
    assert parent["end_page"] == 20
    assert (parent["content_start_page"], parent["content_end_page"], parent["text"]) == (16, 16, "p16-16")
    assert (first_child["content_start_page"], first_child["content_end_page"], first_child["text"]) == (
        16,
        17,
        "p16-17",
    )
    assert (second_child["content_start_page"], second_child["content_end_page"], second_child["text"]) == (
        18,
        20,
        "p18-20",
    )


def test_normalize_tree_retrieval_segments_respects_existing_sibling_end_page():
    tree = [
        {
            "title": "父章节",
            "start_page": 1,
            "end_page": 4,
            "nodes": [
                {"title": "A", "start_page": 1, "end_page": 2, "nodes": []},
                {"title": "B", "start_page": 3, "end_page": 4, "nodes": []},
            ],
        }
    ]

    normalize_tree_retrieval_segments(
        tree,
        page_text_getter=lambda start, end: f"p{start}-{end}",
        max_pages_per_segment=5,
        start_key="start_page",
        end_key="end_page",
    )

    first_child = tree[0]["nodes"][0]
    second_child = tree[0]["nodes"][1]
    assert (first_child["content_start_page"], first_child["content_end_page"], first_child["text"]) == (
        1,
        2,
        "p1-2",
    )
    assert (second_child["content_start_page"], second_child["content_end_page"], second_child["text"]) == (
        3,
        4,
        "p3-4",
    )


def test_normalize_tree_retrieval_segments_splits_medium_segment_into_virtual_nodes():
    tree = [{"title": "长叶子", "start_page": 1, "end_page": 7, "nodes": []}]

    normalize_tree_retrieval_segments(
        tree,
        page_text_getter=lambda start, end: f"p{start}-{end}",
        max_pages_per_segment=5,
        start_key="start_page",
        end_key="end_page",
    )

    node = tree[0]
    assert node["text"] == ""
    assert (node["content_start_page"], node["content_end_page"]) == (1, 7)
    assert [(child["start_page"], child["end_page"], child["text"]) for child in node["nodes"]] == [
        (1, 2, "p1-2"),
        (3, 4, "p3-4"),
        (5, 7, "p5-7"),
    ]
    assert all(child["is_virtual_node"] is True for child in node["nodes"])
    assert all(child["virtual_reason"] == "medium_segment_split" for child in node["nodes"])

from pageindex.tree_utils import build_tree_and_intervals


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

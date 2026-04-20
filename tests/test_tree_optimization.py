import asyncio

from pageindex.tree_optimization import (
    generate_summaries,
    optimize_and_summarize_tree,
    refine_large_nodes,
    thin_small_nodes,
)


def make_tree():
    return [
        {
            "node_id": "001",
            "title": "Root",
            "start_index": 1,
            "end_index": 5,
            "text": "root text",
            "nodes": [
                {
                    "node_id": "002",
                    "title": "Tiny Leaf",
                    "start_index": 1,
                    "end_index": 1,
                    "text": "tiny",
                    "nodes": [],
                },
                {
                    "node_id": "003",
                    "title": "Large Leaf",
                    "start_index": 2,
                    "end_index": 5,
                    "text": "large " * 100,
                    "nodes": [],
                },
            ],
        }
    ]


def simple_token_counter(text, model=None):
    del model
    return len((text or "").split())


async def fake_llm_fn(model, prompt):
    del model
    if "拆分为 2-4 个逻辑子段落" in prompt:
        return """
        [
          {"sub_title": "Part A", "sub_text": "segment one"},
          {"sub_title": "Part B", "sub_text": "segment two"}
        ]
        """
    if "全局文档描述" in prompt or "顶级标题" in prompt:
        return "A compact document description."
    return "summary text"


def test_thin_small_nodes_merges_tiny_leaf_into_parent():
    tree = make_tree()
    thin_small_nodes(tree, min_tokens=10, model_name=None, token_counter_fn=simple_token_counter)

    root = tree[0]
    assert "Tiny Leaf\ntiny" in root["text"]
    assert [child["title"] for child in root["nodes"]] == ["Large Leaf"]


def test_refine_large_nodes_splits_large_leaf_into_children():
    tree = make_tree()
    asyncio.run(
        refine_large_nodes(
            tree,
            max_tokens=10,
            llm_fn=fake_llm_fn,
            token_counter_fn=simple_token_counter,
        )
    )

    large_leaf = tree[0]["nodes"][1]
    assert large_leaf["text"] == ""
    assert [child["title"] for child in large_leaf["nodes"]] == ["Part A", "Part B"]
    assert all(child["start_index"] == 2 and child["end_index"] == 5 for child in large_leaf["nodes"])


def test_generate_summaries_writes_prefix_and_leaf_summaries():
    tree = make_tree()
    asyncio.run(generate_summaries(tree, llm_fn=fake_llm_fn))

    assert tree[0]["prefix_summary"] == "summary text"
    assert tree[0]["nodes"][0]["summary"] == "summary text"
    assert tree[0]["nodes"][1]["summary"] == "summary text"


def test_optimize_and_summarize_tree_runs_full_pipeline():
    tree = make_tree()
    result = asyncio.run(
        optimize_and_summarize_tree(
            tree,
            min_tokens=10,
            max_tokens=10,
            llm_fn=fake_llm_fn,
            token_counter_fn=simple_token_counter,
        )
    )

    structure = result["structure"]
    assert result["doc_description"] == "A compact document description."
    assert "Tiny Leaf\ntiny" in structure[0]["text"]
    refined_leaf = structure[0]["nodes"][0]
    assert refined_leaf["title"] == "Large Leaf"
    assert [child["title"] for child in refined_leaf["nodes"]] == ["Part A", "Part B"]
    assert structure[0]["prefix_summary"] == "summary text"

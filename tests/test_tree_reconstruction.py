import json

import pytest

from pageindex.tree_reconstruction import (
    RECONSTRUCTION_SYSTEM_PROMPT,
    TreeReconstructionError,
    build_context_payload,
    reconstruct_tree_structure,
    validate_tree_logic,
)


def make_initial_nodes():
    return [
        {
            "node_id": "001",
            "original_level": 6,
            "title": "Volume I",
            "physical_index": 2,
            "text": "This agreement is entered into by the following parties.\nThe snippet should be flattened.",
            "needs_llm_fix": False,
        },
        {
            "node_id": "002",
            "original_level": 4,
            "title": "Chapter 1 Requirements",
            "physical_index": None,
            "text": "The supplier shall deliver the goods on time.",
            "needs_llm_fix": True,
        },
        {
            "node_id": "003",
            "original_level": 5,
            "title": "Signature Page",
            "physical_index": 60,
            "text": "",
            "needs_llm_fix": False,
        },
    ]


def test_build_context_payload_truncates_and_includes_fix_flag():
    long_text = "A" * 40 + "\n" + "B" * 60
    payload = build_context_payload(
        [
            {
                "node_id": "001",
                "title": "Example",
                "physical_index": None,
                "text": long_text,
                "needs_llm_fix": True,
            }
        ]
    )

    assert payload == [
        {
            "node_id": "001",
            "title": "Example",
            "physical_index": None,
            "needs_llm_fix": True,
            "snippet": ("A" * 40 + " " + "B" * 39),
        }
    ]


def test_validate_tree_logic_rejects_invalid_first_level():
    nodes = [
        {"node_id": "001", "corrected_level": 4},
        {"node_id": "002", "corrected_level": 4},
    ]

    with pytest.raises(TreeReconstructionError, match="First valid node"):
        validate_tree_logic(nodes)


def test_validate_tree_logic_rejects_isolated_level_jump():
    nodes = [
        {"node_id": "001", "corrected_level": 1},
        {"node_id": "002", "corrected_level": 4},
    ]

    with pytest.raises(TreeReconstructionError, match="isolated level jump|Detected isolated level jump"):
        validate_tree_logic(nodes)


def test_validate_tree_logic_skips_killed_nodes():
    nodes = [
        {"node_id": "001", "corrected_level": -1},
        {"node_id": "002", "corrected_level": 1},
        {"node_id": "003", "corrected_level": -1},
        {"node_id": "004", "corrected_level": 2},
    ]

    assert validate_tree_logic(nodes) is True


def test_reconstruct_tree_structure_merges_llm_levels_and_decisions():
    initial_nodes = make_initial_nodes()
    llm_response = json.dumps(
        [
            {"node_id": "001", "corrected_level": 1, "decision_reason": "Top-level cover section."},
            {"node_id": "002", "corrected_level": 2, "decision_reason": "A valid chapter under Volume I."},
            {"node_id": "003", "corrected_level": 1, "decision_reason": "Closing signature section."},
        ],
        ensure_ascii=False,
    )

    def fake_llm_fn(model, prompt, chat_history=None):
        assert model == "demo-model"
        assert chat_history[0]["content"] == RECONSTRUCTION_SYSTEM_PROMPT
        assert '"needs_llm_fix": true' in prompt
        return llm_response

    result = reconstruct_tree_structure(initial_nodes, model="demo-model", llm_fn=fake_llm_fn)

    assert [node["corrected_level"] for node in result] == [1, 2, 1]
    assert result[1]["decision_reason"] == "A valid chapter under Volume I."
    assert result[1]["reasoning"] == "A valid chapter under Volume I."


def test_reconstruct_tree_structure_accepts_kill_decision():
    initial_nodes = make_initial_nodes()
    llm_response = json.dumps(
        [
            {"node_id": "001", "corrected_level": 1, "decision_reason": "Top level."},
            {"node_id": "002", "corrected_level": -1, "decision_reason": "This is body text, not a heading."},
            {"node_id": "003", "corrected_level": 1, "decision_reason": "Top level ending."},
        ],
        ensure_ascii=False,
    )

    result = reconstruct_tree_structure(initial_nodes, llm_fn=lambda model, prompt, chat_history=None: llm_response)

    assert [node["corrected_level"] for node in result] == [1, -1, 1]
    assert result[1]["decision_reason"] == "This is body text, not a heading."


def test_reconstruct_tree_structure_raises_on_guardrail_failure():
    initial_nodes = make_initial_nodes()
    llm_response = json.dumps(
        [
            {"node_id": "001", "corrected_level": 1, "decision_reason": "Top level."},
            {"node_id": "002", "corrected_level": 4, "decision_reason": "Wrong jump."},
            {"node_id": "003", "corrected_level": 1, "decision_reason": "Top level."},
        ],
        ensure_ascii=False,
    )

    with pytest.raises(TreeReconstructionError) as exc_info:
        reconstruct_tree_structure(
            initial_nodes,
            llm_fn=lambda model, prompt, chat_history=None: llm_response,
        )

    assert exc_info.value.failure_type == "isolated_level_jump"
    assert exc_info.value.node_id == "002"

import json

import pytest

from pageindex.tree_reconstruction import (
    RECONSTRUCTION_SYSTEM_PROMPT,
    RECONSTRUCTION_USER_PROMPT_TEMPLATE,
    TreeReconstructionError,
    build_context_payload,
    repair_level_jumps_after_killed_nodes,
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


def test_reconstruction_prompt_mentions_contract_signature_headings():
    assert "此页为合同签字页" in RECONSTRUCTION_SYSTEM_PROMPT
    assert "买方" in RECONSTRUCTION_SYSTEM_PROMPT
    assert "卖方" in RECONSTRUCTION_SYSTEM_PROMPT
    assert "缺少父级的更深层级" in RECONSTRUCTION_USER_PROMPT_TEMPLATE


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


def test_repair_level_jumps_after_killed_nodes_lowers_over_deep_run():
    nodes = [
        {"node_id": "005", "title": "Volume I", "corrected_level": 1},
        {"node_id": "006", "title": "Signature Page", "corrected_level": -1},
        {"node_id": "007", "title": "Buyer", "corrected_level": 3},
        {"node_id": "008", "title": "Seller", "corrected_level": 3},
        {"node_id": "009", "title": "Volume II", "corrected_level": 1},
    ]

    repaired = repair_level_jumps_after_killed_nodes(nodes)

    assert [node["corrected_level"] for node in repaired] == [1, -1, 2, 2, 1]
    assert repaired[2]["auto_repair_reason"].startswith("Lowered level")
    assert repaired[3]["auto_repair_reason"].startswith("Lowered level")
    assert validate_tree_logic(repaired) is True


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

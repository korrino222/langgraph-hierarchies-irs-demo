"""Tests for Example 01 — artifact handoff."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph_hierarchies.state.context import BaseContext
from langgraph_hierarchies.state.schema import create_base_state_defaults

from examples.example_01_artifact_handoff.agents import (
    EXTRACTOR_GOAL,
    FORMATTER_GOAL,
    child_graph,
    compile_formatter,
    compile_root,
)
from examples.example_01_artifact_handoff.model import RuleBasedModel, load_raw_text
from examples.example_01_artifact_handoff.policy import ARTIFACT_POLICY
from examples.example_01_artifact_handoff.state import HandoffState
from examples.example_01_artifact_handoff.tools import format_expense, parse_expense

pytestmark = [pytest.mark.scripted]

_FIXTURES = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "examples"
    / "example_01_artifact_handoff"
    / "fixtures"
)


def _invoke_ok_pipeline() -> dict:
    root = compile_root()
    context = BaseContext(model=RuleBasedModel.for_ok())
    return root.invoke(
        create_base_state_defaults(),
        config=RunnableConfig(recursion_limit=50),
        context=context,
    )


def _invoke_fail_pipeline() -> dict:
    root = compile_root()
    context = BaseContext(model=RuleBasedModel.for_fail())
    return root.invoke(
        create_base_state_defaults(),
        config=RunnableConfig(recursion_limit=50),
        context=context,
    )


def _formatter_output_from_pipeline(result: dict) -> dict:
    raw = result.get("pipeline_artifact") or result.get("current_agent_report", "")
    return json.loads(raw)


def _replay_formatter(fixture_path: Path) -> dict:
    formatter = compile_formatter()
    state = create_base_state_defaults()
    state["pipeline_artifact"] = fixture_path.read_text().strip()
    return formatter.invoke(state, config=RunnableConfig(recursion_limit=20))


def _tool_names_in_messages(messages: list) -> list[str]:
    names: list[str] = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            names.extend(call["name"] for call in message.tool_calls)
    return names


def test_boundary_ok() -> None:
    root = compile_root()
    extractor = child_graph(root, "extractor")

    parent_state = create_base_state_defaults()
    parent_state["messages"] = [HumanMessage(content="upstream history must not leak")]
    parent_state["pipeline_artifact"] = "prior-artifact"

    child_input = {
        **parent_state,
        "current_agent_args": {
            "task": f"Parse expense note:\n{load_raw_text('raw_ok.txt')}",
            "task_scope": "extraction only",
            "task_iterations": 0,
        },
        "current_tool_call": {
            "name": "extractor",
            "args": {},
            "id": "call-extractor",
            "type": "tool_call",
        },
    }

    entered = extractor.entry_hook(child_input)
    assert entered["messages"] == []

    child_output = extractor.invoke(entered)
    restored = extractor.exit_hook(child_output)

    assert restored["messages"][0].content == "upstream history must not leak"
    extraction = json.loads(restored["pipeline_artifact"])
    assert extraction["status"] == "ok"
    assert extraction["vendor"] == "XYZ Consulting"


def test_goals_are_single_sentence() -> None:
    assert EXTRACTOR_GOAL.endswith(".")
    assert FORMATTER_GOAL.endswith(".")
    assert "extractor" not in FORMATTER_GOAL.lower()
    assert "formatter" not in EXTRACTOR_GOAL.lower()


def test_failure_structured() -> None:
    result = _invoke_fail_pipeline()
    artifact = _formatter_output_from_pipeline(result)

    assert artifact["status"] == "error"
    assert artifact["stage"] == "extraction"
    assert "missing" in artifact["reason"]

    tool_names = _tool_names_in_messages(result.get("messages", []))
    assert "formatter" not in tool_names


def test_replay_matches_pipeline() -> None:
    pipeline_result = _invoke_ok_pipeline()
    pipeline_artifact = _formatter_output_from_pipeline(pipeline_result)

    replay_result = _replay_formatter(_FIXTURES / "extraction_artifact.json")
    replay_artifact = json.loads(
        replay_result.get("pipeline_artifact")
        or replay_result.get("current_agent_report", "")
    )

    assert pipeline_artifact == replay_artifact
    assert pipeline_artifact["card"] == (
        "XYZ Consulting | $87.50 | 2024-03-15 | business development"
    )


def test_parse_and_format_tools() -> None:
    raw = load_raw_text("raw_ok.txt")
    extraction = parse_expense(raw)
    formatted = format_expense(extraction)
    assert formatted["status"] == "ok"
    assert "XYZ Consulting" in formatted["card"]


def test_subchain_policy_on_extractor() -> None:
    from examples.example_01_artifact_handoff.agents import Extractor

    extractor = Extractor(
        state_schema=HandoffState,
        context_schema=BaseContext,
        subchain_policy=ARTIFACT_POLICY,
    ).compile_graph()
    assert extractor.subchain_policy is not None
    assert extractor.subchain_policy.clear_messages is True
    assert "pipeline_artifact" in extractor.subchain_policy.merge_fields

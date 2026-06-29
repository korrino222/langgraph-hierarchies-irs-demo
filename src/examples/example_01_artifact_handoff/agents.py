"""Graph classes for the artifact handoff example."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph_hierarchies.graphs.react import ReactGraph
from langgraph_hierarchies.graphs.simple import SimpleGraph
from langgraph_hierarchies.state.context import BaseContext
from langgraph_hierarchies.state.schema import create_base_state_defaults

from examples.example_01_artifact_handoff.policy import ARTIFACT_POLICY
from examples.example_01_artifact_handoff.state import HandoffState
from examples.example_01_artifact_handoff.tools import (
    artifact_from_state,
    format_expense,
    parse_expense,
)

EXTRACTOR_GOAL = "Extract structured expense fields from raw text."
FORMATTER_GOAL = "Format a structured expense artifact into a card."


def _task_text(state: dict) -> str:
    args = state.get("current_agent_args") or {}
    task = args.get("task", "")
    prefix = "Parse expense note:\n"
    if task.startswith(prefix):
        return task[len(prefix) :].strip()
    return task.strip()


class Extractor(SimpleGraph):
    """Parse raw expense text into a structured JSON artifact."""

    name = "extractor"
    description = EXTRACTOR_GOAL

    def computation_node(
        self, state: dict, config: RunnableConfig | None = None
    ) -> dict:
        result = parse_expense(_task_text(state))
        artifact = json.dumps(result)
        return {
            "current_agent_report": artifact,
            "pipeline_artifact": artifact,
        }


class Formatter(SimpleGraph):
    """Format a structured extraction artifact into a printable card."""

    name = "formatter"
    description = FORMATTER_GOAL

    def computation_node(
        self, state: dict, config: RunnableConfig | None = None
    ) -> dict:
        extraction = artifact_from_state(state)
        result = format_expense(extraction)
        artifact = json.dumps(result)
        return {
            "current_agent_report": artifact,
            "pipeline_artifact": artifact,
        }


class HandoffRoot(ReactGraph):
    """Orchestrate extraction then formatting with artifact-only handoff."""

    name = "handoff_root"
    description = "Two-stage expense handoff pipeline"

    def compile_graph(self, *args: Any, **kwargs: Any):
        extractor = Extractor(
            state_schema=HandoffState,
            context_schema=BaseContext,
            subchain_policy=ARTIFACT_POLICY,
        ).compile_graph()
        formatter = Formatter(
            state_schema=HandoffState,
            context_schema=BaseContext,
            subchain_policy=ARTIFACT_POLICY,
        ).compile_graph()
        return super().compile_graph(
            *args,
            compiled_subgraphs=[extractor, formatter],
            **kwargs,
        )


def compile_root() -> Any:
    """Compile the handoff root graph without subchain_policy on root."""
    root = HandoffRoot(
        state_schema=HandoffState,
        context_schema=BaseContext,
        reports_to_supervisor=False,
        message_system=(
            "Orchestrate expense extraction, then formatting when extraction succeeds."
        ),
        message_reasoning="Which unit should run next?",
    )
    return root.compile_as_root(state_defaults=create_base_state_defaults())


def compile_formatter() -> Any:
    """Compile the formatter unit alone for replay / benchmark runs."""
    return Formatter(
        state_schema=HandoffState,
        context_schema=BaseContext,
        subchain_policy=ARTIFACT_POLICY,
    ).compile_graph()


def child_graph(compiled_root: Any, name: str) -> Any:
    """Return a named child compiled subgraph from the root."""
    for child in compiled_root.compiled_subgraphs:
        if child.name == name:
            return child
    msg = f"Child graph not found: {name}"
    raise KeyError(msg)

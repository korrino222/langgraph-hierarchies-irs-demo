"""Graph classes for the artifact handoff example."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph_hierarchies.graphs.react import ReactGraph
from langgraph_hierarchies.graphs.simple import SimpleGraph
from langgraph_hierarchies.state.context import BaseContext
from langgraph_hierarchies.state.schema import create_base_state_defaults
from langgraph_hierarchies.tools.builtins import finish_task

from examples.example_01_artifact_handoff.policy import ARTIFACT_POLICY
from examples.example_01_artifact_handoff.state import HandoffState
from examples.example_01_artifact_handoff.tools import (
    artifact_from_state,
    format_expense,
    parse_expense,
    report_extraction,
)

EXTRACTOR_GOAL = "Extract structured expense fields from raw text."
FORMATTER_GOAL = "Format a structured expense artifact into a card."
ROOT_LLM_SYSTEM = (
    "You orchestrate a two-stage expense pipeline. The full invoice text is in your task.\n"
    "Steps:\n"
    "1. Call extractor with the full invoice text in the task argument.\n"
    "2. If the extraction artifact has status 'ok', call formatter (no args needed).\n"
    "   If status is 'error', skip formatter and go directly to step 3.\n"
    "3. Call finish_task with the formatted card text or the error artifact JSON.\n"
    "Never call raise_exception. Always complete with finish_task.\n"
    "Call exactly one tool per step."
)
EXTRACTOR_LLM_SYSTEM = (
    "You are an expense extraction unit. "
    "Read the invoice or expense note and call report_extraction.\n"
    "On success: set vendor, amount (float), date (YYYY-MM-DD), and category.\n"
    "On failure: set error='<brief reason>' (leave other fields empty).\n"
    "Always call report_extraction — never raise_exception.\n"
    "Call report_extraction exactly once."
)


def _task_text(state: dict) -> str:
    args = state.get("current_agent_args") or {}
    task = args.get("task", "")
    prefix = "Parse expense note:\n"
    if task.startswith(prefix):
        return task[len(prefix) :].strip()
    if task.strip():
        return task.strip()
    return (state.get("raw_input") or "").strip()


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


class LLMExtractor(ReactGraph):
    """Parse an invoice or expense note into a structured JSON artifact using an LLM."""

    name = "extractor"
    description = EXTRACTOR_GOAL

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # report_extraction handles both success and error; raise_exception is not needed
        self.tools = [t for t in self.tools if getattr(t, "name", None) != "raise_exception"]

    def system(self, state: dict) -> dict:
        result = super().system(state)
        raw = (state.get("raw_input") or "").strip()
        if raw:
            result["messages"].append(HumanMessage(content=f"[INVOICE]\n{raw}"))
        return result


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

    def __init__(self, *args: Any, use_llm_extractor: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.use_llm_extractor = use_llm_extractor
        self.tools = [t for t in self.tools if getattr(t, "name", None) != "raise_exception"]

    def compile_graph(self, *args: Any, **kwargs: Any):
        if self.use_llm_extractor:
            extractor = LLMExtractor(
                state_schema=HandoffState,
                context_schema=BaseContext,
                subchain_policy=ARTIFACT_POLICY,
                tools=[report_extraction],
                message_system=EXTRACTOR_LLM_SYSTEM,
                message_reasoning="Extract the expense fields now.",
            ).compile_graph()
        else:
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


def compile_root(*, use_llm_extractor: bool = False) -> Any:
    """Compile the handoff root graph without subchain_policy on root."""
    root = HandoffRoot(
        state_schema=HandoffState,
        context_schema=BaseContext,
        reports_to_supervisor=False,
        message_system=ROOT_LLM_SYSTEM,
        message_reasoning="Which unit should run next?",
        tools=[finish_task],
        use_llm_extractor=use_llm_extractor,
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

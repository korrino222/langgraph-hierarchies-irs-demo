"""Rule-based scripted model for deterministic handoff execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from examples.example_01_artifact_handoff.tools import format_expense, parse_expense

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _tool_call(
    name: str, args: dict[str, Any], *, call_id: str | None = None
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": call_id or str(uuid4()),
                "type": "tool_call",
            }
        ],
    )


def _delegate_args(task: str) -> dict[str, Any]:
    return {
        "task": task,
        "task_scope": "Stay within assigned unit responsibilities.",
        "task_iterations": 0,
    }


def load_raw_text(name: str) -> str:
    return (_FIXTURES / name).read_text().strip()


def build_ok_responses() -> list[AIMessage]:
    raw_text = load_raw_text("raw_ok.txt")
    extraction = parse_expense(raw_text)
    formatted = format_expense(extraction)
    card = formatted["card"]

    return [
        _tool_call("extractor", _delegate_args(f"Parse expense note:\n{raw_text}")),
        _tool_call("formatter", _delegate_args("Format the extracted expense artifact.")),
        _tool_call("finish_task", {"result": card}),
    ]


def build_fail_responses() -> list[AIMessage]:
    raw_text = load_raw_text("raw_fail.txt")
    error_artifact = json.dumps(parse_expense(raw_text))

    return [
        _tool_call("extractor", _delegate_args(f"Parse expense note:\n{raw_text}")),
        _tool_call("finish_task", {"result": error_artifact}),
    ]


class RuleBasedModel(BaseChatModel):
    """Deterministic model that pops pre-built tool-call responses."""

    responses: list[AIMessage]

    @property
    def _llm_type(self) -> str:
        return "rule-based-model"

    def _generate(
        self,
        messages: list,
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.responses:
            msg = "RuleBasedModel response queue exhausted"
            raise RuntimeError(msg)
        message = self.responses.pop(0)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, **kwargs):
        return self

    @classmethod
    def for_ok(cls) -> RuleBasedModel:
        return cls(responses=build_ok_responses())

    @classmethod
    def for_fail(cls) -> RuleBasedModel:
        return cls(responses=build_fail_responses())


def create_openai_model(*, model: str = "gpt-4o-mini", temperature: float = 0):
    """Return a ChatOpenAI model for root orchestration (--llm-* modes)."""
    import os

    if not os.getenv("OPENAI_API_KEY"):
        msg = "OPENAI_API_KEY is required for --llm-ok / --llm-fail"
        raise RuntimeError(msg)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, temperature=temperature)

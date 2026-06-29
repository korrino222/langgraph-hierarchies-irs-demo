"""CLI entry point for Example 01 — artifact handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
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
from examples.example_01_artifact_handoff.model import RuleBasedModel

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _print_boundary_demo(compiled_root: Any) -> None:
    extractor = child_graph(compiled_root, "extractor")
    parent = create_base_state_defaults()
    parent["messages"] = [HumanMessage(content="upstream history must not leak")]
    parent["pipeline_artifact"] = "prior-artifact"

    child_input = {
        **parent,
        "current_agent_args": {
            "task": f"Parse expense note:\n{_FIXTURES.joinpath('raw_ok.txt').read_text().strip()}",
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
    print(f"[Extractor] entered: {len(entered.get('messages', []))} messages")
    print(f"[Extractor] goal:      \"{EXTRACTOR_GOAL}\"")

    child_output = extractor.invoke(entered)
    restored = extractor.exit_hook(child_output)
    print(
        "[Root]      parent messages after Extractor: "
        f"{len(restored.get('messages', []))} (unchanged)"
    )
    print(f"[Extractor] artifact:  {restored.get('pipeline_artifact', '')}")


def _print_result(result: dict, *, replay: bool = False) -> None:
    artifact_raw = result.get("pipeline_artifact") or result.get("current_agent_report", "")
    try:
        artifact = json.loads(artifact_raw)
    except json.JSONDecodeError:
        print(artifact_raw)
        return

    if replay:
        print("[Formatter] entered:   0 messages")
        print(f"[Formatter] goal:      \"{FORMATTER_GOAL}\"")

    if artifact.get("status") == "ok" and "card" in artifact:
        print(f"[Formatter] card:      {artifact['card']}")
    elif artifact.get("status") == "error":
        print(f"[Pipeline]  error:     {artifact.get('reason', artifact)}")
    else:
        print(json.dumps(artifact, indent=2))


def run_scripted(*, ok: bool) -> dict:
    root = compile_root()
    model = RuleBasedModel.for_ok() if ok else RuleBasedModel.for_fail()
    context = BaseContext(model=model)

    print("[Root]      pipeline started")
    if ok:
        _print_boundary_demo(root)

    result = root.invoke(
        create_base_state_defaults(),
        config=RunnableConfig(recursion_limit=50),
        context=context,
    )

    if ok:
        print(f"[Formatter] goal:      \"{FORMATTER_GOAL}\"")
    _print_result(result)
    return result


def run_replay(fixture_path: Path) -> dict:
    formatter = compile_formatter()
    artifact_text = fixture_path.read_text().strip()

    state = create_base_state_defaults()
    state["pipeline_artifact"] = artifact_text

    result = formatter.invoke(state, config=RunnableConfig(recursion_limit=20))
    _print_result(result, replay=True)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Example 01 — artifact handoff")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--scripted-ok",
        action="store_true",
        help="Run full pipeline on raw_ok.txt (happy path)",
    )
    group.add_argument(
        "--scripted-fail",
        action="store_true",
        help="Run full pipeline on raw_fail.txt (structured error)",
    )
    group.add_argument(
        "--replay",
        metavar="FIXTURE",
        type=Path,
        help="Run Formatter only from a committed extraction artifact",
    )
    args = parser.parse_args(argv)

    if args.scripted_ok:
        run_scripted(ok=True)
    elif args.scripted_fail:
        run_scripted(ok=False)
    elif args.replay:
        run_replay(args.replay)


if __name__ == "__main__":
    main()

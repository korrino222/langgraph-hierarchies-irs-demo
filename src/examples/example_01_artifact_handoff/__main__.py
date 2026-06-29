"""CLI entry point for Example 01 — artifact handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph_hierarchies.state.context import BaseContext
from langgraph_hierarchies.state.schema import create_base_state_defaults

from examples.example_01_artifact_handoff.agents import (
    EXTRACTOR_GOAL,
    FORMATTER_GOAL,
    child_graph,
    compile_formatter,
    compile_root,
)
from examples.example_01_artifact_handoff.model import (
    RuleBasedModel,
    create_openai_model,
    load_raw_text,
)
from examples.example_01_artifact_handoff.tracing import (
    apply_project_override,
    build_run_config,
    load_env,
    new_thread_id,
    print_tracing_hint,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _print_boundary_demo(compiled_root: Any, *, config: Any) -> None:
    extractor = child_graph(compiled_root, "extractor")
    parent = create_base_state_defaults()
    parent["messages"] = [HumanMessage(content="upstream history must not leak")]
    parent["pipeline_artifact"] = "prior-artifact"

    child_input = {
        **parent,
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
    print(f"[Extractor] entered: {len(entered.get('messages', []))} messages")
    print(f"[Extractor] goal:      \"{EXTRACTOR_GOAL}\"")

    child_output = extractor.invoke(entered, config=config)
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


def _llm_initial_state(raw_fixture: str) -> dict:
    raw_text = load_raw_text(raw_fixture)
    state = create_base_state_defaults()
    state["raw_input"] = raw_text
    state["current_agent_args"] = {
        "task": (
            "Process this invoice through extraction and formatting.\n\n"
            f"{raw_text}"
        ),
        "task_scope": (
            "Orchestrate extractor and formatter subgraphs only. "
            "Call extractor first. Call formatter only when extraction status is ok. "
            "Finish with the formatted card or error artifact JSON."
        ),
        "task_iterations": 0,
    }
    return state


def run_pipeline(
    *,
    ok: bool,
    use_llm: bool,
    project: str | None,
    thread_id: str,
) -> dict:
    mode = "llm-ok" if use_llm and ok else "llm-fail" if use_llm else "scripted-ok" if ok else "scripted-fail"
    project_name = apply_project_override(project)
    config = build_run_config(
        thread_id=thread_id,
        run_name=f"example-01-{mode}",
        tags=[mode],
    )

    root = compile_root(use_llm_extractor=use_llm)
    if use_llm:
        context = BaseContext(thread_id=thread_id, model=create_openai_model())
        raw_fixture = "invoice_ok.txt" if ok else "invoice_fail.txt"
        state = _llm_initial_state(raw_fixture)
    else:
        context = BaseContext(
            thread_id=thread_id,
            model=RuleBasedModel.for_ok() if ok else RuleBasedModel.for_fail(),
        )
        state = create_base_state_defaults()

    print("[Root]      pipeline started")
    print_tracing_hint(project_name, thread_id)
    if ok and not use_llm:
        _print_boundary_demo(root, config=config)

    result = root.invoke(state, config=config, context=context)

    if ok:
        print(f"[Formatter] goal:      \"{FORMATTER_GOAL}\"")
    _print_result(result)
    return result


def run_replay(fixture_path: Path, *, project: str | None, thread_id: str) -> dict:
    project_name = apply_project_override(project)
    config = build_run_config(
        thread_id=thread_id,
        run_name="example-01-replay",
        tags=["replay"],
        recursion_limit=20,
    )

    formatter = compile_formatter()
    artifact_text = fixture_path.read_text().strip()

    state = create_base_state_defaults()
    state["pipeline_artifact"] = artifact_text

    print_tracing_hint(project_name, thread_id)
    result = formatter.invoke(state, config=config)
    _print_result(result, replay=True)
    return result


def main(argv: list[str] | None = None) -> None:
    load_env()

    parser = argparse.ArgumentParser(description="Example 01 — artifact handoff")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--scripted-ok",
        action="store_true",
        help="Run full pipeline on raw_ok.txt (deterministic, no API key)",
    )
    group.add_argument(
        "--scripted-fail",
        action="store_true",
        help="Run full pipeline on raw_fail.txt (structured error, no API key)",
    )
    group.add_argument(
        "--llm-ok",
        action="store_true",
        help="Run full pipeline on raw_ok.txt with gpt-4o-mini orchestration (OPENAI_API_KEY)",
    )
    group.add_argument(
        "--llm-fail",
        action="store_true",
        help="Run full pipeline on raw_fail.txt with gpt-4o-mini orchestration",
    )
    group.add_argument(
        "--replay",
        metavar="FIXTURE",
        type=Path,
        help="Run Formatter only from a committed extraction artifact",
    )
    parser.add_argument(
        "--project",
        metavar="NAME",
        help="Override LANGCHAIN_PROJECT for LangSmith (default: langgraph-hierarchies-examples-01)",
    )
    args = parser.parse_args(argv)
    thread_id = new_thread_id()

    if args.scripted_ok:
        run_pipeline(ok=True, use_llm=False, project=args.project, thread_id=thread_id)
    elif args.scripted_fail:
        run_pipeline(ok=False, use_llm=False, project=args.project, thread_id=thread_id)
    elif args.llm_ok:
        run_pipeline(ok=True, use_llm=True, project=args.project, thread_id=thread_id)
    elif args.llm_fail:
        run_pipeline(ok=False, use_llm=True, project=args.project, thread_id=thread_id)
    elif args.replay:
        run_replay(args.replay, project=args.project, thread_id=thread_id)


if __name__ == "__main__":
    main()

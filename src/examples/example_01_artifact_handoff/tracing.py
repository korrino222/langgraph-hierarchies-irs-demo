"""LangSmith tracing helpers for Example 01."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph_hierarchies.tracing import build_invoke_config

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PROJECT = "langgraph-hierarchies-examples-01"


def load_env() -> None:
    """Load `.env` from the repo root when python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_REPO_ROOT / ".env", override=False)


def new_thread_id() -> str:
    """Return a LangSmith-friendly thread ID (UUID v7 when langsmith is available)."""
    try:
        from langsmith import uuid7

        return str(uuid7())
    except ImportError:
        return str(uuid4())


def langsmith_enabled() -> bool:
    value = os.getenv("LANGCHAIN_TRACING_V2", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def apply_project_override(project: str | None) -> str:
    """Set LANGCHAIN_PROJECT for this process; return the effective project name."""
    effective = project or os.getenv("LANGCHAIN_PROJECT") or _DEFAULT_PROJECT
    os.environ["LANGCHAIN_PROJECT"] = effective
    return effective


def build_run_config(
    *,
    thread_id: str,
    run_name: str,
    tags: list[str],
    recursion_limit: int = 50,
) -> RunnableConfig:
    """Build RunnableConfig with LangSmith thread metadata and tags."""
    return build_invoke_config(
        thread_id=thread_id,
        run_name=run_name,
        tags=["example-01", *tags],
        recursion_limit=recursion_limit,
    )


def print_tracing_hint(project: str, thread_id: str) -> None:
    if langsmith_enabled():
        print(f"[Trace]     LangSmith project: {project}")
        print(f"[Trace]     Thread ID:         {thread_id}")
        print("[Trace]     View runs at https://smith.langchain.com (Threads tab)")
    else:
        print(
            "[Trace]     LangSmith off — set LANGCHAIN_TRACING_V2=true and "
            "LANGCHAIN_API_KEY in .env to capture runs"
        )
        print(f"[Trace]     Thread ID (local): {thread_id}")

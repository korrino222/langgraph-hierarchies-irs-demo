"""LangSmith tracing helpers for Example 01."""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.runnables import RunnableConfig

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PROJECT = "langgraph-hierarchies-examples-01"


def load_env() -> None:
    """Load `.env` from the repo root when python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_REPO_ROOT / ".env", override=False)


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
    run_name: str,
    tags: list[str],
    recursion_limit: int = 50,
) -> RunnableConfig:
    """Build RunnableConfig with LangSmith-friendly run name and tags."""
    return RunnableConfig(
        recursion_limit=recursion_limit,
        run_name=run_name,
        tags=["example-01", *tags],
    )


def print_tracing_hint(project: str) -> None:
    if langsmith_enabled():
        print(f"[Trace]     LangSmith project: {project}")
        print("[Trace]     View runs at https://smith.langchain.com")
    else:
        print(
            "[Trace]     LangSmith off — set LANGCHAIN_TRACING_V2=true and "
            "LANGCHAIN_API_KEY in .env to capture runs"
        )

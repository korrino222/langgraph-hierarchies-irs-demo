# langgraph-hierarchies-examples

Progressive runnable examples for [`langgraph-hierarchies`](https://github.com/korrino222/langgraph-hierarchies) — decomposability, responsibility boundaries, and batch processing, culminating in a full IRS reporting workflow.

> **Not tax advice.** Synthetic fixtures only. For the minimal deterministic pattern reference, see the library's [`examples/irs_reporting/`](https://github.com/korrino222/langgraph-hierarchies/tree/main/examples/irs_reporting).

## Quick start

```bash
git clone https://github.com/korrino222/langgraph-hierarchies-examples.git
cd langgraph-hierarchies-examples
cp .env.example .env   # optional: OPENAI_API_KEY, LANGCHAIN_API_KEY
uv sync
uv run python -m examples.example_01_artifact_handoff --scripted-ok
```

See [DESIGN.md](./DESIGN.md) for `--llm-ok`, `--replay`, and LangSmith tracing.

## Status

Example **01 — Artifact handoff** is implemented. See [DESIGN.md](./DESIGN.md) for run modes, LangSmith tracing, and article mapping. Examples 02–04 are planned.

Planned example ladder:

1. **Artifact handoff** — isolated context, `pipeline_artifact` only
2. **Responsibility move** — smell in monolith, boundary in decomposed
3. **Batch TodoGraph** — flat context at scale (optional v0.1)
4. **IRS full** — monolith vs decomposed comparison

## Development

```bash
uv run ruff check .
uv run pytest -m "not llm"
```

Requires **`langgraph-hierarchies>=0.0.3`** (PyPI) for LangSmith thread propagation via `build_invoke_config`.

## License

MIT — see [LICENSE](./LICENSE).

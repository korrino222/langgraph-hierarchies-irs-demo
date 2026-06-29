# langgraph-hierarchies-examples

Progressive runnable examples for [`langgraph-hierarchies`](https://github.com/korrino222/langgraph-hierarchies) — decomposability, responsibility boundaries, and batch processing, culminating in a full IRS reporting workflow.

> **Not tax advice.** Synthetic fixtures only. For the minimal deterministic pattern reference, see the library's [`examples/irs_reporting/`](https://github.com/korrino222/langgraph-hierarchies/tree/main/examples/irs_reporting).

## Quick start

```bash
git clone https://github.com/korrino222/langgraph-hierarchies-examples.git
cd langgraph-hierarchies-examples
cp .env.example .env   # optional: OPENAI_API_KEY for --llm runs
uv sync
```

## Status

Repository scaffold only — examples not yet implemented. See [DESIGN.md](./DESIGN.md) (coming) for exhibit notes.

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

## License

MIT — see [LICENSE](./LICENSE).

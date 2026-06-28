# langgraph-hierarchies-irs-demo

Article companion for the **decomposability / layered hierarchies** narrative — a realistic small-business tax-prep workflow built on [`langgraph-hierarchies`](https://github.com/korrino222/langgraph-hierarchies).

Demonstrates compiled subgraphs, `SubchainPolicy` artifact isolation, `TodoGraph` gating, and intentional reconciliation failures with fake bank statements and invoices.

> **Not tax advice.** Synthetic fixtures only. For the minimal deterministic pattern reference, see the library's [`examples/irs_reporting/`](https://github.com/korrino222/langgraph-hierarchies/tree/main/examples/irs_reporting).

## Quick start

```bash
git clone https://github.com/korrino222/langgraph-hierarchies-irs-demo.git
cd langgraph-hierarchies-irs-demo
cp .env.example .env   # add OPENAI_API_KEY
uv sync
uv run python -m irs_demo.hierarchy --max-transactions 10
```

## No API key (scripted)

```bash
uv run python -m irs_demo.hierarchy --scripted --max-transactions 5
```

## Status

Repository scaffold only — agent hierarchy and fixtures are not yet implemented. See [DESIGN.md](./DESIGN.md) (coming in D-08) for intentional responsibility smells and exhibit notes.

## Development

Requires a local clone of [`langgraph-hierarchies`](https://github.com/korrino222/langgraph-hierarchies) as a sibling directory (`../2026-06-28-langgraph-hierarchies` or symlink `../langgraph-hierarchies`). After library v0.1 publishes to PyPI, the editable path dependency will be replaced with a version pin.

```bash
uv run ruff check .
uv run pytest -m "not llm"
```

## License

MIT — see [LICENSE](./LICENSE).

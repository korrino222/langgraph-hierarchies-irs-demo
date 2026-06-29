# DESIGN.md

Design notes for `langgraph-hierarchies-examples`. Maps runnable exhibits to the article ladder.

## Example 01 — Artifact handoff

**Article:** [Decomposability in AI workflows](https://medium.com/@ishish222/decomposability-in-ai-workflows-what-it-is-and-why-you-want-it-c12c9a939565)

**Run (no API key):**

```bash
uv run python -m examples.example_01_artifact_handoff --scripted-ok
uv run python -m examples.example_01_artifact_handoff --replay src/examples/example_01_artifact_handoff/fixtures/extraction_artifact.json
```

**Run with real LLM** (orchestration at root only; child units stay deterministic):

```bash
cp .env.example .env   # set OPENAI_API_KEY
uv run python -m examples.example_01_artifact_handoff --llm-ok
uv run python -m examples.example_01_artifact_handoff --llm-fail
```

**LangSmith tracing** (works for scripted, LLM, and replay modes):

```bash
cp .env.example .env   # set LANGCHAIN_TRACING_V2=true, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT
uv run python -m examples.example_01_artifact_handoff --scripted-ok
# Open https://smith.langchain.com → project langgraph-hierarchies-examples-01
# Filter by tag: example-01, scripted-ok | llm-ok | replay
```

Optional: `--project my-traces` overrides `LANGCHAIN_PROJECT` for a single run.

### Four decomposability tests → code

| Test | What the reader should see | Where in code |
|------|---------------------------|---------------|
| **Goal** | Each unit has a one-sentence purpose with no cross-reference | `EXTRACTOR_GOAL` / `FORMATTER_GOAL` in [`agents.py`](src/examples/example_01_artifact_handoff/agents.py); printed in CLI trace |
| **Boundary** | Child enters with `messages=[]`; parent history unchanged; only `pipeline_artifact` merges back | `ARTIFACT_POLICY` in [`policy.py`](src/examples/example_01_artifact_handoff/policy.py); `test_boundary_ok` in [`tests/test_01_artifact_handoff.py`](tests/test_01_artifact_handoff.py) |
| **Failure** | Bad input → structured error artifact; Formatter never invoked | `parse_expense()` in [`tools.py`](src/examples/example_01_artifact_handoff/tools.py); `RuleBasedModel.for_fail()` in [`model.py`](src/examples/example_01_artifact_handoff/model.py); `test_failure_structured` |
| **Replay** | Formatter runs from committed fixture; output matches full pipeline | `--replay` in [`__main__.py`](src/examples/example_01_artifact_handoff/__main__.py); `test_replay_matches_pipeline` |

### Graph topology

```
HandoffRoot (ReactGraph, no subchain_policy)
  ├── Extractor (SimpleGraph, ARTIFACT_POLICY)
  └── Formatter (SimpleGraph, ARTIFACT_POLICY)
```

Root orchestrates via subgraph tool calls. Children clear `messages` on entry and merge only `pipeline_artifact` on exit — the article's “context stays lean” primitive in miniature.

In `--llm-*` modes, **only the root** uses `gpt-4o-mini` for delegation decisions; Extractor and Formatter remain deterministic `SimpleGraph` workers. LangSmith traces show nested `handoff_root` → `extractor` / `formatter` spans either way.

### LangSmith: what to verify

| Mode | Run name | Tags | What to look for |
|------|----------|------|------------------|
| `--scripted-ok` | `example-01-scripted-ok` | `example-01`, `scripted-ok` | Nested extractor → formatter; child inputs have empty message history |
| `--scripted-fail` | `example-01-scripted-fail` | `example-01`, `scripted-fail` | Extractor only; no formatter child |
| `--llm-ok` | `example-01-llm-ok` | `example-01`, `llm-ok` | Root LLM turns + same subgraph nesting as scripted |
| `--replay` | `example-01-replay` | `example-01`, `replay` | Formatter span only |

### Deferred to later examples

- **Responsibility smell** → Example 02
- **Flat context at N items** → Example 03 / IRS matching stage
- **Monolith vs decomposed metrics** → IRS capstone (04)

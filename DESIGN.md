# DESIGN.md

Design notes for `langgraph-hierarchies-examples`. Maps runnable exhibits to the article ladder.

## Example 01 — Artifact handoff

**Article:** [Decomposability in AI workflows](https://medium.com/@ishish222/decomposability-in-ai-workflows-what-it-is-and-why-you-want-it-c12c9a939565)

**Run:**

```bash
uv run python -m examples.example_01_artifact_handoff --scripted-ok
uv run python -m examples.example_01_artifact_handoff --replay src/examples/example_01_artifact_handoff/fixtures/extraction_artifact.json
```

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

### Deferred to later examples

- **Responsibility smell** → Example 02
- **Flat context at N items** → Example 03 / IRS matching stage
- **Monolith vs decomposed metrics** → IRS capstone (04)

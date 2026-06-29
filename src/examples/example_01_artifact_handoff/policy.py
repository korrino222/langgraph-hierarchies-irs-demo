"""Subchain policy for artifact-only handoff."""

from langgraph_hierarchies.graphs.compiled import SubchainPolicy

ARTIFACT_POLICY = SubchainPolicy(
    clear_messages=True,
    merge_fields=["pipeline_artifact"],
)

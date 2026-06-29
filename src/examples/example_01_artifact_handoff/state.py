"""State schema for the artifact handoff example."""

from __future__ import annotations

from typing import Annotated

from langgraph_hierarchies.state.reducers import reduce_current_agent_report
from langgraph_hierarchies.state.schema import BaseState


class HandoffState(BaseState):
    """Extended state carrying structured artifacts between pipeline stages."""

    pipeline_artifact: Annotated[str, reduce_current_agent_report]
    raw_input: str

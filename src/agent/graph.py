"""LangGraph single-node graph definition.

Placeholder graph with a single call_model node.
Add retrieval and generation nodes as agentic logic grows.
"""

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from agent.configuration import Configuration
from agent.state import State


async def call_model(
    state: State, config: RunnableConfig
) -> dict[str, Any]:
    """Process the current state through the model.

    Args:
        state: Current graph state.
        config: LangGraph runtime configuration.

    Returns:
        dict[str, Any]: Partial state update with messages.
    """
    configuration = Configuration.from_runnable_config(config)
    _ = configuration  # will be used when adding model calls
    return {"messages": []}


# Build the graph
builder = StateGraph(State, context_schema=Configuration)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

graph = builder.compile()

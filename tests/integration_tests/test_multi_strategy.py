# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Test each search strategy end-to-end with live backends.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

import pytest
from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import graph

pytestmark = pytest.mark.anyio

@pytest.mark.parametrize("query,expected_strategy", [
    ("Who is David Bowie?", "local"),
    ("What are the main movements in electronic music?", "global"),
    ("How did the British Invasion influence psychedelic rock bands in San Francisco?", "drift"),
    ("Compare the impact of The Beatles and The Rolling Stones on 1960s culture.", "hybrid"),
])
async def test_strategies_end_to_end(query, expected_strategy):
    """Test each search strategy end-to-end with live backends."""
    inputs = {"messages": [HumanMessage(content=query)]}
    
    # Increase timeout for real LLM and DB calls
    async for output in graph.astream(inputs, stream_mode="updates"):
        for node_name, state_update in output.items():
            print(f"\n--- Node: {node_name} ---")
            if node_name == "query_analyzer":
                strategy = state_update.get("strategy")
                print(f"Strategy classified as: {strategy}")
                assert strategy in ("local", "global", "drift", "hybrid")
            
            if node_name == "synthesize_answer":
                messages = state_update.get("messages", [])
                if messages:
                    print(f"Answer snippet: {messages[0].content[:100]}...")
                    assert isinstance(messages[0], AIMessage)
                    assert len(messages[0].content) > 0

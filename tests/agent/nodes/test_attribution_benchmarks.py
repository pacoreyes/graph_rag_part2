import pytest
import re
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage

from agent.nodes.generation import synthesize_answer, _check_faithfulness, _resolve_aku_legend
from agent.state import State

pytestmark = pytest.mark.anyio

def _make_state(**overrides) -> State:
    defaults = {
        "messages": [HumanMessage(content="Test query")],
        "source_urls": {
            "Q1": {"name": "Entity 1", "wikipedia_url": "http://link1"},
            "Q2": {"name": "Entity 2", "wikipedia_url": "http://link2"},
        }
    }
    defaults.update(overrides)
    return State(**defaults)

def _make_config(**configurable):
    return {"configurable": configurable} if configurable else {"configurable": {"model": "gemini-2.0-flash"}}

# --- 1. Cleanliness Check ---

@patch("agent.nodes.generation.gemini_client")
@patch("agent.nodes.generation.gemini_generate")
async def test_cleanliness_check_negative(mock_generate, mock_gc):
    """Verify that forbidden patterns ARE detected."""
    mock_gc.get_client = AsyncMock(return_value=MagicMock())
    # Simulated response with forbidden patterns
    mock_generate.return_value = '{"answer": "Kraftwerk [1] was formed in Germany [Kraftwerk](http://link). [2]", "evidence": {"1": "...", "2": "..."}}'
    
    state = _make_state(
        entities=[{"name": "E1", "pagerank": 1}],
        chunk_evidence=[{"text": "C1", "score": 1}]
    )
    
    result = await synthesize_answer(state, _make_config())
    synthesis = result["messages"][0].content.split("---")[0]
    
    # It SHOULD find a link
    assert re.search(r"\[[^\]]+\]\([^)]+\)", synthesis)
    # It SHOULD find a bracketed name
    assert re.search(r"\[[^\]]*[^\d\s\[\],][^\]]*\]", synthesis)


@patch("agent.nodes.generation.gemini_client")
@patch("agent.nodes.generation.gemini_generate")
async def test_cleanliness_check_positive(mock_generate, mock_gc):
    """Verify that valid numerical citations are ALLOWED."""
    mock_gc.get_client = AsyncMock(return_value=MagicMock())
    # Clean response with only numerical citations
    mock_generate.return_value = '{"answer": "Kraftwerk [1] was a pioneer [1][2].", "evidence": {"1": "...", "2": "..."}}'
    
    state = _make_state(
        entities=[{"name": "E1", "pagerank": 1}, {"name": "E2", "pagerank": 0.9}],
    )
    
    result = await synthesize_answer(state, _make_config())
    synthesis = result["messages"][0].content.split("---")[0]
    
    # NO forbidden links
    assert not re.search(r"\[[^\]]+\]\([^)]+\)", synthesis)
    # NO forbidden bracketed names (only numbers allowed)
    assert not re.search(r"\[[^\]]*[^\d\s\[\],][^\]]*\]", synthesis)
    # BUT numerical citations ARE there
    assert re.search(r"\[1\]", synthesis)
    assert re.search(r"\[2\]", synthesis)

# --- 2. Origin Accuracy Test ---

@patch("agent.nodes.generation.gemini_client")
@patch("agent.nodes.generation.gemini_generate")
async def test_origin_accuracy_in_legend(mock_generate, mock_gc):
    """Verify that the legend correctly maps Neo4j vs. Pinecone origins."""
    mock_gc.get_client = AsyncMock(return_value=MagicMock())
    mock_generate.return_value = '{"answer": "Fact from Graph [1]. Fact from Vector [2].", "evidence": {"1": "G", "2": "V"}}'
    
    state = _make_state(
        entities=[{"name": "E1", "pagerank": 1, "origin": "Graph DB", "method": "M1", "qid": "Q1"}],
        chunk_evidence=[{"text": "C1", "score": 1, "origin": "Vector DB", "method": "M2", "article_id": "Q2", "id": "chunk1"}]
    )
    
    result = await synthesize_answer(state, _make_config())
    answer_text = result["messages"][0].content
    legend = answer_text.split("---")[1]
    
    assert "[1]" in legend and "Origin: Graph DB" in legend
    assert "[2]" in legend and "Origin: Vector DB" in legend

# --- 3. Noise Robustness (Stress Test) ---

@patch("agent.nodes.generation.gemini_client")
@patch("agent.nodes.generation.gemini_generate")
async def test_noise_robustness(mock_generate, mock_gc):
    """Stress test: provide 50% irrelevant AKUs and ensure they aren't cited."""
    mock_gc.get_client = AsyncMock(return_value=MagicMock())
    # LLM correctly ignores the noise
    mock_generate.return_value = '{"answer": "Relevant fact [1].", "evidence": {"1": "..."}}'
    
    state = _make_state(
        entities=[
            {"name": "Relevant", "pagerank": 1, "qid": "Q1"},
            {"name": "Noise 1", "pagerank": 0.1},
            {"name": "Noise 2", "pagerank": 0.05},
        ]
    )
    
    result = await synthesize_answer(state, _make_config())
    answer_text = result["messages"][0].content
    
    # Check that only [1] is cited
    citations = re.findall(r"\[(\d+)\]", answer_text.split("---")[0])
    assert "1" in citations
    assert "2" not in citations
    assert "3" not in citations

# --- 4. Faithfulness Benchmark (NLI Judge) ---

@patch("agent.nodes.generation.gemini_client")
@patch("agent.nodes.generation.gemini_generate")
async def test_faithfulness_nli_logic(mock_generate, mock_gc):
    mock_gc.get_client = AsyncMock(return_value=MagicMock())
    # Hallucinated citation [2]
    mock_generate.return_value = '{"answer": "Einstein was a cat [2].", "evidence": {"2": "..."}}'
    
    # We call our own logic which detects the hallucination
    akus = [{"index": 1, "content": "Einstein was a physicist"}]
    result = _check_faithfulness("Einstein was a cat [2].", akus)
    
    assert result["is_faithful"] is False
    assert "Hallucinated" in result["issue"]

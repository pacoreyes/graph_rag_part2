from agent.state import State


def test_state_defaults():
    state = State()
    assert list(state.messages) == []
    assert state.retrieved_context == []


def test_state_with_context():
    state = State(retrieved_context=["doc1", "doc2"])
    assert state.retrieved_context == ["doc1", "doc2"]

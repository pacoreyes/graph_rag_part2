# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# No description available.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

from agent.configuration import Configuration


def test_configuration_defaults():
    cfg = Configuration()
    assert cfg.model == "gemini-1.5-flash"
    assert cfg.retrieval_k == 5
    assert cfg.community_level == 2
    assert cfg.neighborhood_depth == 1
    assert cfg.similarity_threshold == 0.5


def test_configuration_custom_values():
    cfg = Configuration(model="gemini-2.0-flash", retrieval_k=10)
    assert cfg.model == "gemini-2.0-flash"
    assert cfg.retrieval_k == 10


def test_from_runnable_config_none_returns_defaults():
    cfg = Configuration.from_runnable_config(None)
    assert cfg.model == "gemini-1.5-flash"
    assert cfg.retrieval_k == 5


def test_from_runnable_config_empty_dict_returns_defaults():
    cfg = Configuration.from_runnable_config({})
    assert cfg.model == "gemini-1.5-flash"


def test_from_runnable_config_with_configurable():
    config = {"configurable": {"model": "custom-model", "retrieval_k": 20}}
    cfg = Configuration.from_runnable_config(config)
    assert cfg.model == "custom-model"
    assert cfg.retrieval_k == 20


def test_from_runnable_config_ignores_unknown_keys():
    config = {"configurable": {"model": "custom-model", "unknown_key": "value"}}
    cfg = Configuration.from_runnable_config(config)
    assert cfg.model == "custom-model"
    assert cfg.retrieval_k == 5


def test_configuration_is_frozen():
    cfg = Configuration()
    try:
        cfg.model = "other"  # type: ignore[misc]
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass


def test_configuration_community_level():
    cfg = Configuration(community_level=3)
    assert cfg.community_level == 3


def test_configuration_neighborhood_depth():
    cfg = Configuration(neighborhood_depth=2)
    assert cfg.neighborhood_depth == 2


def test_configuration_similarity_threshold():
    cfg = Configuration(similarity_threshold=0.7)
    assert cfg.similarity_threshold == 0.7


def test_from_runnable_config_with_new_fields():
    config = {"configurable": {"community_level": 4, "similarity_threshold": 0.8}}
    cfg = Configuration.from_runnable_config(config)
    assert cfg.community_level == 4
    assert cfg.similarity_threshold == 0.8

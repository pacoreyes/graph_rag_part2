from agent.configuration import Configuration


def test_configuration_defaults():
    cfg = Configuration()
    assert cfg.model == "gemini-1.5-flash"
    assert cfg.retrieval_k == 5


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

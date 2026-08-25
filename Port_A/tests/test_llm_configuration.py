import pytest

from core import server


def test_llm_configuration_requires_key_url_and_model(monkeypatch):
    monkeypatch.setattr(server, "DASHSCOPE_API_KEY", "your-api-key")
    monkeypatch.setattr(
        server,
        "DASHSCOPE_BASE_URL",
        "https://your-llm-endpoint.example.com/v1",
    )
    monkeypatch.setattr(server, "LLM_MODEL_NAME", "")

    assert server.llm_configuration_errors() == [
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_BASE_URL",
        "LLM_MODEL_NAME",
    ]
    with pytest.raises(RuntimeError, match="LLM_MODEL_NAME"):
        server.require_llm_configuration()


def test_llm_configuration_accepts_explicit_model(monkeypatch):
    monkeypatch.setattr(server, "DASHSCOPE_API_KEY", "secret")
    monkeypatch.setattr(server, "DASHSCOPE_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setattr(server, "LLM_MODEL_NAME", "locally-served-model")

    assert server.llm_configuration_errors() == []
    server.require_llm_configuration()

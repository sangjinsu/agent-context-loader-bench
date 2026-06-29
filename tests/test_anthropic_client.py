from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from agent_loader_bench.config import load_settings
from agent_loader_bench.llm.anthropic_client import AnthropicMessagesClient


def install_fake_anthropic(monkeypatch: pytest.MonkeyPatch, captured: dict) -> None:
    def create(**kwargs):
        captured.update(kwargs)
        text_block = SimpleNamespace(type="text", text="implemented")
        return SimpleNamespace(content=[text_block])

    class FakeAnthropic:
        def __init__(self, *, api_key: str) -> None:
            captured["api_key"] = api_key
            self.messages = SimpleNamespace(create=create)

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))


def test_generate_maps_system_and_user_without_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    install_fake_anthropic(monkeypatch, captured)

    client = AnthropicMessagesClient(api_key="from-settings", model="claude-opus-4-8")
    # The runner passes temperature for interface parity; it must never reach the API.
    result = client.generate(instructions="rules", user_input="request", temperature=0.0)

    assert result == "implemented"
    assert captured["api_key"] == "from-settings"
    assert captured["model"] == "claude-opus-4-8"
    assert captured["system"] == "rules"
    assert captured["messages"] == [{"role": "user", "content": "request"}]
    assert "temperature" not in captured
    assert "top_p" not in captured


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicMessagesClient(api_key=None, model="claude-opus-4-8")


def test_settings_infers_anthropic_provider_from_claude_model(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("LLM_MODEL=claude-opus-4-8\n", encoding="utf-8")
    settings = load_settings(tmp_path)
    assert settings.llm_provider == "anthropic"


def test_settings_defaults_to_openai_provider(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("LLM_MODEL=gpt-5.5\n", encoding="utf-8")
    settings = load_settings(tmp_path)
    assert settings.llm_provider == "openai"


def test_settings_explicit_provider_overrides_inference(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("LLM_MODEL=claude-opus-4-8\nLLM_PROVIDER=openai\n", encoding="utf-8")
    settings = load_settings(tmp_path)
    assert settings.llm_provider == "openai"

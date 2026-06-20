from __future__ import annotations

from importlib import import_module
import os

from .base import LLMClient


class AnthropicMessagesClient(LLMClient):
    """Claude fallback client using the official Anthropic SDK.

    Kept separate from the OpenAI client on purpose: the two providers have
    different request shapes and must not be mixed. Claude Opus 4.8 rejects
    sampling parameters (temperature/top_p) with a 400, so the `temperature`
    argument exists only for interface compatibility and is never forwarded.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "claude-opus-4-8",
        max_tokens: int = 1024,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for --live-llm runs with a Claude model")

        try:
            anthropic_module = import_module("anthropic")
        except ImportError as error:
            raise RuntimeError("The 'anthropic' package is required for --live-llm runs with a Claude model") from error

        self._client = anthropic_module.Anthropic(api_key=self.api_key)
        self.model = model
        self.max_tokens = max_tokens

    def generate(
        self,
        *,
        instructions: str,
        user_input: str,
        model: str | None = None,
        temperature: float | None = None,  # accepted for interface parity; never sent to the API
    ) -> str:
        selected_model = model or self.model
        # No temperature / top_p: Opus 4.8 (and the 4.7+ family) return 400 if sampling params are sent.
        message = self._client.messages.create(
            model=selected_model,
            max_tokens=self.max_tokens,
            system=instructions,
            messages=[{"role": "user", "content": user_input}],
        )
        return _extract_text(message)


def _extract_text(message: object) -> str:
    """Concatenate text blocks from an Anthropic Messages response."""
    content = getattr(message, "content", None) or []
    parts: list[str] = []
    for block in content:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts).strip()

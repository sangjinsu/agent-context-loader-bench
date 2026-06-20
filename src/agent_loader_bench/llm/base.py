from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    def generate(
        self,
        *,
        instructions: str,
        user_input: str,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str: ...

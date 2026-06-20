from __future__ import annotations

from importlib import import_module
import os

from .base import LLMClient


class OpenAIResponsesClient(LLMClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gpt-4.1-mini",
        temperature: float = 0,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for --live-llm runs")

        try:
            openai_module = import_module("openai")
        except ImportError as error:
            raise RuntimeError("The 'openai' package is required for --live-llm runs") from error

        self._client = openai_module.OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature

    def generate(
        self,
        *,
        instructions: str,
        user_input: str,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        selected_model = model or self.model
        selected_temperature = self.temperature if temperature is None else temperature

        if hasattr(self._client, "responses"):
            response = self._client.responses.create(
                model=selected_model,
                instructions=instructions,
                input=user_input,
                temperature=selected_temperature,
            )
            output_text = getattr(response, "output_text", None)
            if isinstance(output_text, str) and output_text.strip():
                return output_text.strip()

        completion = self._client.chat.completions.create(
            model=selected_model,
            temperature=selected_temperature,
            messages=[
                {"role": "developer", "content": instructions},
                {"role": "user", "content": user_input},
            ],
        )
        message = completion.choices[0].message.content
        if isinstance(message, list):
            return "".join(part.get("text", "") for part in message if isinstance(part, dict)).strip()
        return str(message).strip()

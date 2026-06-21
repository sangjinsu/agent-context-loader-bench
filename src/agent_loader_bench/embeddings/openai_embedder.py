from __future__ import annotations

from importlib import import_module
import os

from .base import l2_normalize


class OpenAIEmbedder:
    """Embedder backed by the OpenAI embeddings API (e.g. text-embedding-3-small).

    Vectors are L2-normalized on return so the loader's dot-product cosine stays
    valid. Network calls happen here; the loader only reaches this code when the
    index was built with `embedding_provider="openai"`.
    """

    provider = "openai"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings")

        try:
            openai_module = import_module("openai")
        except ImportError as error:
            raise RuntimeError("The 'openai' package is required for OpenAI embeddings") from error

        self._client = openai_module.OpenAI(api_key=self.api_key)
        self.model = model
        self._requested_dimensions = dimensions
        self.dimensions = dimensions or 0

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        kwargs: dict[str, object] = {"model": self.model, "input": inputs}
        if self._requested_dimensions:
            kwargs["dimensions"] = self._requested_dimensions
        response = self._client.embeddings.create(**kwargs)
        vectors = [l2_normalize([float(value) for value in item.embedding]) for item in response.data]
        if vectors:
            self.dimensions = len(vectors[0])
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

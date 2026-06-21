from __future__ import annotations

from typing import Any

from .base import Embedder, l2_normalize
from .hashing import DEFAULT_HASHING_DIMENSIONS, HashingEmbedder, deterministic_text_vector
from .openai_embedder import OpenAIEmbedder

__all__ = [
    "Embedder",
    "HashingEmbedder",
    "OpenAIEmbedder",
    "deterministic_text_vector",
    "l2_normalize",
    "make_embedder",
    "embedder_from_index",
    "DEFAULT_HASHING_DIMENSIONS",
]


def make_embedder(
    provider: str | None,
    model: str | None = None,
    *,
    api_key: str | None = None,
    dimensions: int | None = None,
) -> Embedder:
    """Construct an embedder by provider name."""
    name = (provider or "openai").strip().lower()
    if name == "hashing":
        return HashingEmbedder(dimensions=dimensions or DEFAULT_HASHING_DIMENSIONS)
    if name == "openai":
        return OpenAIEmbedder(
            api_key=api_key,
            model=model or "text-embedding-3-small",
            dimensions=dimensions,
        )
    raise ValueError(f"Unsupported embedding provider '{provider}'. Choose 'hashing' or 'openai'.")


def embedder_from_index(payload: dict[str, Any], settings: Any | None = None) -> Embedder:
    """Build the embedder recorded in a vector index payload.

    The loader uses this so a query is embedded with the same provider/model and
    dimension that produced the stored document vectors — keeping query and
    document spaces consistent without relying on the current environment.
    """
    provider = str(payload.get("embedding_provider", "hashing"))
    model = payload.get("embedding_model")
    dimensions_value = payload.get("dimensions")
    dimensions = int(dimensions_value) if dimensions_value else None
    api_key = getattr(settings, "openai_api_key", None) if settings is not None else None
    return make_embedder(provider, model, api_key=api_key, dimensions=dimensions)

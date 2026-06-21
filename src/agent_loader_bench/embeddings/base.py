from __future__ import annotations

import math
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors for the vector_search loader.

    Implementations must return L2-normalized vectors so the loader's
    dot-product cosine similarity stays valid. `provider`/`model`/`dimensions`
    are recorded in the index so a query is later embedded with the same model
    the documents were embedded with.
    """

    provider: str
    model: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]

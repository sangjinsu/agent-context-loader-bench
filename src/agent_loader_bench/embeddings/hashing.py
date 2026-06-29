from __future__ import annotations

from hashlib import sha256

from .base import l2_normalize

DEFAULT_HASHING_DIMENSIONS = 24


def deterministic_text_vector(text: str, dimensions: int = DEFAULT_HASHING_DIMENSIONS) -> list[float]:
    """Offline, deterministic feature-hashing vector (no network, no model).

    Each term hashes into one of `dimensions` buckets with a sign and weight;
    the accumulated vector is L2-normalized. This is the default offline
    embedder and the stand-in used by unit tests.
    """
    # Imported lazily to avoid a loaders <-> embeddings import cycle at module load.
    from agent_loader_bench.loaders.base import normalize_terms

    values = [0.0] * dimensions
    for term in normalize_terms(text):
        digest = sha256(term.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        weight = 1.0 + (digest[5] / 255.0)
        values[bucket] += sign * weight
    return l2_normalize(values)


class HashingEmbedder:
    """Embedder backed by `deterministic_text_vector` — offline and reproducible."""

    provider = "hashing"

    def __init__(self, dimensions: int = DEFAULT_HASHING_DIMENSIONS) -> None:
        self.model = "hashing"
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [deterministic_text_vector(text, self.dimensions) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return deterministic_text_vector(text, self.dimensions)

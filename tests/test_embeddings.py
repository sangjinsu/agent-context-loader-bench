from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from textwrap import dedent
from types import SimpleNamespace

import pytest

from agent_loader_bench.embeddings import (
    HashingEmbedder,
    OpenAIEmbedder,
    make_embedder,
)
from agent_loader_bench.loaders import VectorContextLoader, build_vector_index
from agent_loader_bench.loaders.vector_search import default_min_score


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")


@pytest.fixture
def embed_repo(tmp_path: Path) -> Path:
    _write(tmp_path / "AGENTS.md", "# Rules\n\nRead first.")
    _write(
        tmp_path / "skills" / "vllm-benchmark" / "SKILL.md",
        """
        ---
        id: skill.vllm.benchmark
        type: skill
        title: vLLM Benchmark Skill
        description: Use this skill when comparing vLLM throughput and latency benchmarks.
        version: 0.1.0
        status: draft
        tags:
          - vllm
          - benchmark
        activation:
          keywords:
            - vllm
            - benchmark
            - throughput
            - latency
        priority: 80
        ---

        # vLLM Benchmark Skill

        ## When to use

        Use this skill when the user asks to benchmark vLLM performance.

        ## Steps

        1. Prepare prompts.
        2. Measure throughput and latency.
        """,
    )
    return tmp_path


def install_fake_openai(monkeypatch: pytest.MonkeyPatch, captured: list[dict]) -> None:
    class FakeEmbeddings:
        def create(self, *, model: str, input: list[str], dimensions: int | None = None):
            captured.append({"model": model, "input": list(input), "dimensions": dimensions})
            dim = dimensions or 8
            data = []
            for text in input:
                base = sum(ord(char) for char in text) + 1
                data.append(SimpleNamespace(embedding=[float((base + index * 7) % 17 + 1) for index in range(dim)]))
            return SimpleNamespace(data=data)

    class FakeOpenAI:
        def __init__(self, *, api_key: str) -> None:
            self.api_key = api_key
            self.embeddings = FakeEmbeddings()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(dimensions=24)

    first = embedder.embed_query("benchmark vllm throughput")
    second = embedder.embed_query("benchmark vllm throughput")

    assert first == second
    assert len(first) == 24
    assert embedder.provider == "hashing"
    assert _norm(first) == pytest.approx(1.0)


def test_default_min_score_is_provider_aware() -> None:
    # Real embeddings need a higher relevance floor than the near-zero hashing space.
    assert default_min_score("hashing") == 0.2
    assert default_min_score("openai") == 0.28
    assert default_min_score(None) == 0.2


def test_make_embedder_selects_provider_and_rejects_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    assert make_embedder("hashing").provider == "hashing"

    install_fake_openai(monkeypatch, [])
    assert make_embedder("openai", api_key="test-key").provider == "openai"

    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        make_embedder("voyage")


def test_openai_embedder_normalizes_batches_and_passes_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict] = []
    install_fake_openai(monkeypatch, captured)

    embedder = OpenAIEmbedder(api_key="test-key", model="text-embedding-3-small", dimensions=8)
    vectors = embedder.embed_documents(["alpha", "beta", "gamma"])

    assert len(vectors) == 3
    assert all(_norm(vector) == pytest.approx(1.0) for vector in vectors)
    assert captured[0]["model"] == "text-embedding-3-small"
    assert captured[0]["input"] == ["alpha", "beta", "gamma"]
    assert captured[0]["dimensions"] == 8


def test_openai_embedder_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIEmbedder(api_key=None)


def test_build_and_load_openai_index_records_provider_and_reembeds_query(
    embed_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[dict] = []
    install_fake_openai(monkeypatch, captured)

    index_path = embed_repo / ".agentdb" / "vector" / "index.json"
    build_vector_index(
        embed_repo,
        index_path,
        OpenAIEmbedder(api_key="test-key", model="text-embedding-3-small"),
    )

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["embedding_provider"] == "openai"
    assert payload["embedding_model"] == "text-embedding-3-small"
    assert payload["dimensions"] == 8
    document_embed_calls = len(captured)

    loader = VectorContextLoader(
        embed_repo,
        index_path=index_path,
        settings=SimpleNamespace(openai_api_key="test-key"),
    )
    context = loader.load("Benchmark throughput and latency for vllm.")

    # The loader re-embedded the query with the same provider recorded in the index.
    assert context.debug["embedding_provider"] == "openai"
    assert len(captured) == document_embed_calls + 1
    assert captured[-1]["input"] == ["Benchmark throughput and latency for vllm."]

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from agent_loader_bench.corpus import (
    CorpusDocument,
    SkillDocument,
    load_corpus_document,
    load_documents,
    load_skill_document,
)
from agent_loader_bench.tokens import estimate_tokens


def write_file(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")
    return path


@pytest.fixture
def sample_docs(tmp_path: Path) -> dict[str, Path]:
    skill_path = write_file(
        tmp_path / "skills" / "vllm-benchmark" / "SKILL.md",
        """
        ---
        id: skill.vllm.benchmark
        type: skill
        title: vLLM Benchmark Skill
        description: Benchmark vLLM throughput and latency.
        version: 0.1.0
        status: draft
        tags:
          - vllm
          - benchmark
        activation:
          keywords:
            - vllm
            - throughput
            - latency
        priority: 80
        ---

        # vLLM Benchmark Skill

        Intro paragraph.

        ## When to use

        Use this skill for throughput comparisons.

        ### Caveats

        Benchmark on the same hardware.

        ## Steps

        1. Prepare prompt fixtures.
        2. Run repeated measurements.
        """,
    )
    wiki_path = write_file(
        tmp_path / "wiki" / "concepts" / "latency.md",
        """
        # Latency Notes

        Overview text.

        ## Definitions

        Latency is end-to-end delay.

        ## Measurement

        Record percentile values.
        """,
    )
    return {"skill": skill_path, "wiki": wiki_path}


def test_load_skill_document_parses_frontmatter_and_sections(sample_docs: dict[str, Path]) -> None:
    document = load_skill_document(sample_docs["skill"])

    assert isinstance(document, SkillDocument)
    assert document.id == "skill.vllm.benchmark"
    assert document.activation_keywords == ["vllm", "throughput", "latency"]
    assert document.tags == ["vllm", "benchmark"]
    assert [section.title for section in document.sections] == ["When to use", "Caveats", "Steps"]
    assert [section.level for section in document.sections] == [2, 3, 2]
    assert "Use this skill for throughput comparisons." in document.sections[0].content
    assert "Benchmark on the same hardware." in document.sections[1].content


def test_load_corpus_document_reads_general_markdown(sample_docs: dict[str, Path]) -> None:
    document = load_corpus_document(sample_docs["wiki"])

    assert isinstance(document, CorpusDocument)
    assert not isinstance(document, SkillDocument)
    assert document.frontmatter == {}
    assert document.title == "Latency Notes"
    assert [section.title for section in document.sections] == ["Definitions", "Measurement"]
    assert document.path.as_posix().endswith("wiki/concepts/latency.md")


def test_load_documents_sorts_paths_deterministically(sample_docs: dict[str, Path]) -> None:
    documents = load_documents([sample_docs["wiki"], sample_docs["skill"]])

    assert [document.path.name for document in documents] == ["SKILL.md", "latency.md"]


def test_load_skill_document_rejects_missing_required_fields(tmp_path: Path) -> None:
    broken_skill = write_file(
        tmp_path / "skills" / "broken" / "SKILL.md",
        """
        ---
        id: skill.broken
        type: skill
        title: Broken Skill
        description: Missing fields example.
        version: 0.1.0
        status: draft
        tags:
          - broken
        activation: {}
        ---

        # Broken Skill
        """,
    )

    with pytest.raises(ValueError, match="activation.keywords, priority"):
        load_skill_document(broken_skill)


def test_estimate_tokens_is_simple_and_deterministic() -> None:
    text = "alpha beta, gamma"

    assert estimate_tokens(text) == estimate_tokens(text)
    assert estimate_tokens(text) == 4

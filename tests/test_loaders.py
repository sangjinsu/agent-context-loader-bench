from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest

from agent_loader_bench.loaders import (
    FSDirectContextLoader,
    HybridContextLoader,
    JSONDocumentContextLoader,
    ManifestContextLoader,
    SQLiteFTSContextLoader,
    SQLiteFTSSectionContextLoader,
    SQLiteMetadataContextLoader,
    VectorContextLoader,
    build_deterministic_vector_index,
    build_json_document_store,
    build_manifest_index,
    build_sqlite_fts_index,
    build_sqlite_metadata_index,
    sqlite_fts5_available,
)


def write_file(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")
    return path


@pytest.fixture
def sample_loader_repo(tmp_path: Path) -> Path:
    write_file(
        tmp_path / "AGENTS.md",
        """
        # Agent Loader Rules

        Always read this file first.

        Keep Markdown as source of truth.
        """,
    )
    write_file(
        tmp_path / "skills" / "docs-writing" / "SKILL.md",
        """
        ---
        id: skill.docs.writing
        type: skill
        title: Docs Writing Skill
        description: Use this skill when updating technical documentation and examples.
        version: 0.1.0
        status: draft
        tags:
          - docs
          - writing
        activation:
          keywords:
            - docs
            - documentation
            - guide
            - writing
        priority: 50
        ---

        # Docs Writing Skill

        Intro paragraph for docs work.

        ## When to use

        Use this skill when the user asks for docs updates.

        ## Steps

        1. Identify outdated explanations.
        2. Update examples.
        """,
    )
    write_file(
        tmp_path / "skills" / "vllm-benchmark" / "SKILL.md",
        """
        ---
        id: skill.vllm.benchmark
        type: skill
        title: vLLM Benchmark Skill
        description: Use this skill when adding, running, or comparing vLLM throughput and latency benchmarks.
        version: 0.1.0
        status: draft
        tags:
          - vllm
          - benchmark
          - latency
        activation:
          keywords:
            - vllm
            - benchmark
            - throughput
            - latency
            - performance
        priority: 80
        ---

        # vLLM Benchmark Skill

        Benchmark intro.

        ## When to use

        Use this skill when the user asks to benchmark vLLM performance.

        ## Inputs

        Collect hardware and model details.

        ## Steps

        1. Prepare prompt fixtures.
        2. Run repeated measurements.

        ## Safety and Constraints

        Keep comparisons on the same hardware.
        """,
    )
    write_file(
        tmp_path / "wiki" / "concepts" / "latency.md",
        """
        # Latency Notes

        Overview for latency benchmarking.

        ## Definitions

        Latency is end-to-end delay.

        ## Measurement

        Record percentile values for benchmark runs.
        """,
    )
    return tmp_path


def test_fs_direct_selects_expected_skill_and_assembles_context(sample_loader_repo: Path) -> None:
    loader = FSDirectContextLoader(sample_loader_repo)

    context = loader.load("Add a vllm benchmark that compares throughput and latency.", task_type="benchmark")

    assert context.loader_name == "fs_direct"
    assert context.selected_skills == ["skill.vllm.benchmark"]
    assert context.selected_sections == []
    assert "AGENTS.md excerpt" in context.context_text
    assert "User request:" in context.context_text
    assert "[Skill: skill.vllm.benchmark]" in context.context_text
    assert "Path: skills/vllm-benchmark/SKILL.md" in context.context_text
    assert "Latency Notes" in context.context_text
    assert context.context_text.index("AGENTS.md excerpt") < context.context_text.index("User request:")
    assert context.context_text.index("User request:") < context.context_text.index("[Skill: skill.vllm.benchmark]")
    assert context.debug["selection"][0]["reason"]


def test_fs_direct_no_match_returns_empty_context_with_debug(sample_loader_repo: Path) -> None:
    loader = FSDirectContextLoader(sample_loader_repo)

    context = loader.load("Compose a jazz trio setlist.", task_type="music")

    assert context.selected_skills == []
    assert context.context_text == ""
    assert context.context_token_estimate == 0
    assert context.debug["selected"] == []
    assert context.debug["selection"][0]["score"] >= context.debug["selection"][-1]["score"]


def test_fs_direct_is_deterministic_for_repeated_calls(sample_loader_repo: Path) -> None:
    loader = FSDirectContextLoader(sample_loader_repo)

    first = loader.load("Need docs writing help for examples.")
    second = loader.load("Need docs writing help for examples.")

    assert first == second


def test_fs_direct_avoids_benchmark_skill_for_docs_only_trap(sample_loader_repo: Path) -> None:
    loader = FSDirectContextLoader(sample_loader_repo)

    context = loader.load(
        "Write documentation explaining benchmark results without adding a vLLM benchmark implementation.",
        task_type="docs",
    )

    assert context.selected_skills == ["skill.docs.writing"]


def test_manifest_loader_requires_existing_index(sample_loader_repo: Path) -> None:
    loader = ManifestContextLoader(sample_loader_repo, manifest_path=sample_loader_repo / ".agentdb" / "manifest.json")

    with pytest.raises(FileNotFoundError, match="manifest"):
        loader.load("Need docs help.")


def test_manifest_loader_selects_from_manifest_and_loads_markdown(sample_loader_repo: Path) -> None:
    manifest_path = sample_loader_repo / ".agentdb" / "manifest.json"
    build_manifest_index(sample_loader_repo, manifest_path)
    loader = ManifestContextLoader(sample_loader_repo, manifest_path=manifest_path)

    context = loader.load("Need docs writing help for guides.")

    assert context.loader_name == "manifest_json"
    assert context.selected_skills == ["skill.docs.writing"]
    assert "[Skill: skill.docs.writing]" in context.context_text
    assert context.debug["index_path"].endswith(".agentdb/manifest.json")


def test_manifest_loader_rejects_stale_index(sample_loader_repo: Path) -> None:
    manifest_path = sample_loader_repo / ".agentdb" / "manifest.json"
    build_manifest_index(sample_loader_repo, manifest_path)
    skill_path = sample_loader_repo / "skills" / "docs-writing" / "SKILL.md"
    skill_path.write_text(skill_path.read_text(encoding="utf-8") + "\nExtra source change.\n", encoding="utf-8")
    loader = ManifestContextLoader(sample_loader_repo, manifest_path=manifest_path)

    with pytest.raises(RuntimeError, match="stale"):
        loader.load("Need docs writing help.")


def test_sqlite_metadata_missing_index_raises_clear_error(sample_loader_repo: Path) -> None:
    loader = SQLiteMetadataContextLoader(
        sample_loader_repo, index_path=sample_loader_repo / ".agentdb" / "index.sqlite"
    )

    with pytest.raises(FileNotFoundError, match="index.sqlite"):
        loader.load("Need benchmark help.")


def test_sqlite_metadata_selects_expected_skill(sample_loader_repo: Path) -> None:
    index_path = sample_loader_repo / ".agentdb" / "index.sqlite"
    build_sqlite_metadata_index(sample_loader_repo, index_path)
    loader = SQLiteMetadataContextLoader(sample_loader_repo, index_path=index_path)

    context = loader.load("Compare vllm throughput and latency.")

    assert context.selected_skills == ["skill.vllm.benchmark"]
    assert context.debug["index_path"].endswith(".agentdb/index.sqlite")
    assert context.debug["selection"][0]["score"] > 0


def test_sqlite_metadata_loader_rejects_stale_index(sample_loader_repo: Path) -> None:
    index_path = sample_loader_repo / ".agentdb" / "index.sqlite"
    build_sqlite_metadata_index(sample_loader_repo, index_path)
    skill_path = sample_loader_repo / "skills" / "vllm-benchmark" / "SKILL.md"
    skill_path.write_text(skill_path.read_text(encoding="utf-8") + "\nExtra source change.\n", encoding="utf-8")
    loader = SQLiteMetadataContextLoader(sample_loader_repo, index_path=index_path)

    with pytest.raises(RuntimeError, match="stale"):
        loader.load("Compare vllm throughput and latency.")


def test_sqlite_fts_loaders_reject_stale_index_before_empty_query(sample_loader_repo: Path) -> None:
    if not sqlite_fts5_available():
        pytest.skip("SQLite FTS5 is unavailable in this environment")

    index_path = sample_loader_repo / ".agentdb" / "fts.sqlite"
    build_sqlite_fts_index(sample_loader_repo, index_path)
    skill_path = sample_loader_repo / "skills" / "vllm-benchmark" / "SKILL.md"
    skill_path.write_text(skill_path.read_text(encoding="utf-8") + "\nExtra source change.\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="stale"):
        SQLiteFTSContextLoader(sample_loader_repo, index_path=index_path).load("and the or")

    with pytest.raises(RuntimeError, match="stale"):
        SQLiteFTSSectionContextLoader(sample_loader_repo, index_path=index_path).load("and the or")


def test_sqlite_fts_selects_expected_skill_or_skips_if_unavailable(sample_loader_repo: Path) -> None:
    if not sqlite_fts5_available():
        pytest.skip("SQLite FTS5 is unavailable in this environment")

    index_path = sample_loader_repo / ".agentdb" / "fts.sqlite"
    build_sqlite_fts_index(sample_loader_repo, index_path)
    loader = SQLiteFTSContextLoader(sample_loader_repo, index_path=index_path)

    context = loader.load("I need a vllm latency benchmark.")

    assert context.selected_skills == ["skill.vllm.benchmark"]
    assert context.debug["selection"][0]["score"] > 0


def test_sqlite_fts_section_loads_relevant_sections_only_and_uses_fewer_tokens(sample_loader_repo: Path) -> None:
    if not sqlite_fts5_available():
        pytest.skip("SQLite FTS5 is unavailable in this environment")

    index_path = sample_loader_repo / ".agentdb" / "fts.sqlite"
    build_sqlite_fts_index(sample_loader_repo, index_path)
    full_loader = SQLiteFTSContextLoader(sample_loader_repo, index_path=index_path)
    section_loader = SQLiteFTSSectionContextLoader(sample_loader_repo, index_path=index_path)

    full_context = full_loader.load("I need benchmark steps for vllm throughput.")
    section_context = section_loader.load("I need benchmark steps for vllm throughput.")

    assert section_context.selected_skills == ["skill.vllm.benchmark"]
    assert "Steps" in section_context.selected_sections
    assert "Inputs" not in section_context.selected_sections
    assert "## Steps" in section_context.context_text
    assert "## Inputs" not in section_context.context_text
    assert section_context.context_token_estimate < full_context.context_token_estimate


def test_sqlite_fts_section_excludes_documents_without_relevant_sections(sample_loader_repo: Path) -> None:
    if not sqlite_fts5_available():
        pytest.skip("SQLite FTS5 is unavailable in this environment")

    index_path = sample_loader_repo / ".agentdb" / "fts.sqlite"
    build_sqlite_fts_index(sample_loader_repo, index_path)
    section_loader = SQLiteFTSSectionContextLoader(sample_loader_repo, index_path=index_path)

    context = section_loader.load("Need technical writing guidance.")

    assert context.selected_skills == []
    assert context.selected_sections == []
    assert "[Skill: skill.docs.writing]" not in context.context_text
    assert "Intro paragraph for docs work." not in context.context_text
    assert "technical documentation and examples" not in context.context_text


def test_json_document_loader_selects_expected_skill(sample_loader_repo: Path) -> None:
    store_path = sample_loader_repo / ".agentdb" / "document_store.jsonl"
    build_json_document_store(sample_loader_repo, store_path)
    loader = JSONDocumentContextLoader(sample_loader_repo, store_path=store_path)

    context = loader.load("Please update docs and writing examples.")

    assert context.selected_skills == ["skill.docs.writing"]
    assert context.debug["index_path"].endswith(".agentdb/document_store.jsonl")


def test_json_document_loader_rejects_stale_store(sample_loader_repo: Path) -> None:
    store_path = sample_loader_repo / ".agentdb" / "document_store.jsonl"
    build_json_document_store(sample_loader_repo, store_path)
    skill_path = sample_loader_repo / "skills" / "docs-writing" / "SKILL.md"
    skill_path.write_text(skill_path.read_text(encoding="utf-8") + "\nExtra source change.\n", encoding="utf-8")
    loader = JSONDocumentContextLoader(sample_loader_repo, store_path=store_path)

    with pytest.raises(RuntimeError, match="stale"):
        loader.load("Please update docs and writing examples.")


def test_vector_loader_is_deterministic_and_selects_expected_skill(sample_loader_repo: Path) -> None:
    index_path = sample_loader_repo / ".agentdb" / "vector" / "index.json"
    build_deterministic_vector_index(sample_loader_repo, index_path)
    loader = VectorContextLoader(sample_loader_repo, index_path=index_path)

    first = loader.load("Benchmark throughput and latency for vllm.")
    second = loader.load("Benchmark throughput and latency for vllm.")

    assert first.selected_skills == ["skill.vllm.benchmark"]
    assert first == second
    assert first.debug["selection"][0]["similarity"] > 0


def test_vector_loader_rejects_stale_index(sample_loader_repo: Path) -> None:
    index_path = sample_loader_repo / ".agentdb" / "vector" / "index.json"
    build_deterministic_vector_index(sample_loader_repo, index_path)
    skill_path = sample_loader_repo / "skills" / "vllm-benchmark" / "SKILL.md"
    skill_path.write_text(skill_path.read_text(encoding="utf-8") + "\nExtra source change.\n", encoding="utf-8")
    loader = VectorContextLoader(sample_loader_repo, index_path=index_path)

    with pytest.raises(RuntimeError, match="stale"):
        loader.load("Benchmark throughput and latency for vllm.")


def test_hybrid_loader_reports_explicit_fallbacks(sample_loader_repo: Path) -> None:
    manifest_path = sample_loader_repo / ".agentdb" / "manifest.json"
    build_manifest_index(sample_loader_repo, manifest_path)
    loader = HybridContextLoader(
        sample_loader_repo,
        manifest_path=manifest_path,
        metadata_index_path=sample_loader_repo / ".agentdb" / "missing.sqlite",
        fts_index_path=sample_loader_repo / ".agentdb" / "missing-fts.sqlite",
        vector_index_path=sample_loader_repo / ".agentdb" / "vector" / "missing.json",
    )

    context = loader.load("Need docs writing help.")

    assert context.selected_skills == ["skill.docs.writing"]
    assert any("sqlite_metadata" in entry for entry in context.debug["fallbacks"])
    assert any("sqlite_fts" in entry for entry in context.debug["fallbacks"])
    assert any("vector_search" in entry for entry in context.debug["fallbacks"])
    assert context.debug["selected_via"] == "manifest_json"


def test_hybrid_loader_combines_provider_scores_and_section_context(sample_loader_repo: Path) -> None:
    if not sqlite_fts5_available():
        pytest.skip("SQLite FTS5 is unavailable in this environment")

    metadata_index_path = sample_loader_repo / ".agentdb" / "index.sqlite"
    fts_index_path = sample_loader_repo / ".agentdb" / "fts.sqlite"
    vector_index_path = sample_loader_repo / ".agentdb" / "vector" / "index.json"
    build_sqlite_metadata_index(sample_loader_repo, metadata_index_path)
    build_sqlite_fts_index(sample_loader_repo, fts_index_path)
    build_deterministic_vector_index(sample_loader_repo, vector_index_path)
    loader = HybridContextLoader(
        sample_loader_repo,
        manifest_path=sample_loader_repo / ".agentdb" / "missing-manifest.json",
        metadata_index_path=metadata_index_path,
        fts_index_path=fts_index_path,
        vector_index_path=vector_index_path,
    )

    context = loader.load("Need vllm benchmark steps for throughput and latency.")

    assert context.selected_skills == ["skill.vllm.benchmark"]
    assert "## Steps" in context.context_text
    assert "## Inputs" not in context.context_text
    assert len(context.debug["providers_used"]) >= 2
    assert "skill.vllm.benchmark" in context.debug["combined_scores"]
    assert len(context.debug["combined_scores"]["skill.vllm.benchmark"]["providers"]) >= 2
    assert any("manifest_json" in entry for entry in context.debug["fallbacks"])


def test_manifest_builder_writes_reproducible_json(sample_loader_repo: Path) -> None:
    manifest_path = sample_loader_repo / ".agentdb" / "manifest.json"

    build_manifest_index(sample_loader_repo, manifest_path)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [entry["id"] for entry in payload["documents"]] == [
        "skill.docs.writing",
        "skill.vllm.benchmark",
        "wiki:wiki/concepts/latency.md",
    ]

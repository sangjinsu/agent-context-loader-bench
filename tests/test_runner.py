from __future__ import annotations

import json
from dataclasses import dataclass
import sys
from types import SimpleNamespace
from pathlib import Path
from textwrap import dedent

import pytest

from agent_loader_bench.cli import main
from agent_loader_bench.loaders import sqlite_fts5_available
from agent_loader_bench.config import load_settings
from agent_loader_bench.llm.openai_client import OpenAIResponsesClient
from agent_loader_bench.runner import (
    DatasetItem,
    evaluate_loaded_context,
    load_dataset,
    run_dataset,
)


def write_file(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")
    return path


@pytest.fixture
def sample_runner_repo(tmp_path: Path) -> Path:
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

        ## When to use

        Use this skill when the user asks to benchmark vLLM performance.

        ## Inputs

        Collect hardware and model details.

        ## Steps

        1. Prepare prompt fixtures.
        2. Run repeated measurements.
        """,
    )
    write_file(
        tmp_path / "wiki" / "concepts" / "latency.md",
        """
        # Latency Notes

        ## Definitions

        Latency is end-to-end delay.
        """,
    )
    write_file(
        tmp_path / "datasets" / "requests.yml",
        """
        - id: req.vllm.benchmark.basic
          user_request: "Add a vLLM benchmark that compares throughput and latency."
          expected_skills:
            - skill.vllm.benchmark
          expected_sections:
            - When to use
            - Steps
          task_type: benchmark
          difficulty: easy
          notes: Exact keyword match.
        - id: req.docs.writing.basic
          user_request: "Update the docs examples and guide."
          expected_skills:
            - skill.docs.writing
          expected_sections:
            - When to use
            - Steps
          task_type: docs
          difficulty: easy
          notes: Exact keyword match.
        """,
    )
    return tmp_path


def test_load_dataset_parses_request_items(sample_runner_repo: Path) -> None:
    items = load_dataset(sample_runner_repo / "datasets" / "requests.yml")

    assert items[0] == DatasetItem(
        id="req.vllm.benchmark.basic",
        user_request="Add a vLLM benchmark that compares throughput and latency.",
        expected_skills=["skill.vllm.benchmark"],
        expected_sections=["When to use", "Steps"],
        task_type="benchmark",
        difficulty="easy",
        notes="Exact keyword match.",
    )


def test_evaluate_loaded_context_reports_skill_and_section_matches() -> None:
    item = DatasetItem(
        id="req.vllm.benchmark.basic",
        user_request="Add a vLLM benchmark that compares throughput and latency.",
        expected_skills=["skill.vllm.benchmark"],
        expected_sections=["When to use", "Steps"],
        task_type="benchmark",
        difficulty="easy",
        notes="Exact keyword match.",
    )

    @dataclass
    class LoadedContextStub:
        selected_skills: list[str]
        selected_sections: list[str]

    evaluation = evaluate_loaded_context(
        item,
        LoadedContextStub(
            selected_skills=["skill.vllm.benchmark"],
            selected_sections=["When to use", "Steps"],
        ),
    )

    assert evaluation["matched_expected_skills"] == ["skill.vllm.benchmark"]
    assert evaluation["missing_expected_skills"] == []
    assert evaluation["matched_expected_sections"] == ["Steps", "When to use"]
    assert evaluation["task_success"] is True

    evaluation = evaluate_loaded_context(
        item,
        LoadedContextStub(
            selected_skills=["skill.vllm.benchmark", "skill.docs.writing"],
            selected_sections=["When to use", "Steps"],
        ),
    )

    assert evaluation["unexpected_skills"] == ["skill.docs.writing"]
    assert evaluation["task_success"] is False

    evaluation = evaluate_loaded_context(
        item,
        LoadedContextStub(
            selected_skills=["skill.vllm.benchmark"],
            selected_sections=["When to use"],
        ),
    )

    assert evaluation["missing_expected_sections"] == ["Steps"]
    assert evaluation["task_success"] is False


def test_load_settings_reads_dotenv_without_overriding_environment(
    sample_runner_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_file(
        sample_runner_repo / ".env",
        """
        OPENAI_API_KEY=from-env-file-key
        LLM_MODEL=from-env-file
        LLM_TEMPERATURE=0.25
        TRACE_PATH=.agentdb/custom-traces.jsonl
        """,
    )
    monkeypatch.setenv("LLM_MODEL", "from-process-env")

    settings = load_settings(sample_runner_repo)

    assert settings.llm_model == "from-process-env"
    assert settings.openai_api_key == "from-env-file-key"
    assert settings.llm_temperature == 0.25
    assert settings.trace_path == sample_runner_repo / ".agentdb" / "custom-traces.jsonl"


def test_openai_client_accepts_api_key_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    class FakeOpenAI:
        def __init__(self, *, api_key: str) -> None:
            captured["api_key"] = api_key
            self.responses = SimpleNamespace(
                create=lambda **_kwargs: SimpleNamespace(output_text="ok"),
            )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    client = OpenAIResponsesClient(api_key="from-dotenv")

    assert captured["api_key"] == "from-dotenv"
    assert client.generate(instructions="rules", user_input="request") == "ok"


def test_cli_build_manifest_and_all_indexes_create_expected_files(
    sample_runner_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Keep the CLI build offline: the vector backend defaults to OpenAI embeddings.
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hashing")
    manifest_path = sample_runner_repo / ".agentdb" / "manifest.json"

    exit_code = main(
        [
            "build-manifest",
            "--output",
            str(manifest_path),
        ],
        repo_root=sample_runner_repo,
    )

    assert exit_code == 0
    assert manifest_path.exists()

    if not sqlite_fts5_available():
        pytest.skip("SQLite FTS5 is unavailable in this environment")

    exit_code = main(
        [
            "build-index",
            "--backend",
            "all",
        ],
        repo_root=sample_runner_repo,
    )

    assert exit_code == 0
    assert (sample_runner_repo / ".agentdb" / "manifest.json").exists()
    assert (sample_runner_repo / ".agentdb" / "index.sqlite").exists()
    assert (sample_runner_repo / ".agentdb" / "fts.sqlite").exists()
    assert (sample_runner_repo / ".agentdb" / "document_store.jsonl").exists()
    assert (sample_runner_repo / ".agentdb" / "vector" / "index.json").exists()


def test_cli_inspect_prints_selected_expected_skill(
    sample_runner_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["build-manifest"], repo_root=sample_runner_repo) == 0

    exit_code = main(
        [
            "inspect",
            "--loader",
            "manifest_json",
            "--request-id",
            "req.vllm.benchmark.basic",
        ],
        repo_root=sample_runner_repo,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "skill.vllm.benchmark" in captured.out
    assert "manifest_json" in captured.out


def test_run_without_live_llm_writes_trace_and_does_not_call_llm(sample_runner_repo: Path) -> None:
    trace_path = sample_runner_repo / ".agentdb" / "traces.jsonl"

    class NoCallClient:
        def generate(self, *, instructions: str, user_input: str) -> str:
            raise AssertionError("LLM client should not be called without --live-llm")

    assert main(["build-manifest"], repo_root=sample_runner_repo) == 0

    result = run_dataset(
        repo_root=sample_runner_repo,
        loader_name="manifest_json",
        dataset_path=sample_runner_repo / "datasets" / "requests.yml",
        live_llm=False,
        trace_path=trace_path,
        llm_client=NoCallClient(),
    )

    assert result["total_requests"] == 2
    assert result["live_llm"] is False
    trace_lines = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert len(trace_lines) == 2
    assert all("llm_output" not in line for line in trace_lines)
    assert all(line["llm_completed"] is False for line in trace_lines)


def test_run_with_live_llm_uses_fake_client_and_writes_trace(sample_runner_repo: Path) -> None:
    trace_path = sample_runner_repo / ".agentdb" / "live-traces.jsonl"

    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def generate(self, *, instructions: str, user_input: str) -> str:
            self.calls.append((instructions, user_input))
            return "implemented"

    client = FakeClient()
    assert main(["build-manifest"], repo_root=sample_runner_repo) == 0

    result = run_dataset(
        repo_root=sample_runner_repo,
        loader_name="manifest_json",
        dataset_path=sample_runner_repo / "datasets" / "requests.yml",
        live_llm=True,
        trace_path=trace_path,
        llm_client=client,
    )

    assert result["live_llm"] is True
    assert len(client.calls) == 2
    trace_lines = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert trace_lines[0]["llm_completed"] is True
    assert trace_lines[0]["task_success"] is True
    assert trace_lines[0]["test_passed"] is True
    assert set(trace_lines[0]) == {
        "context_token_estimate",
        "llm_completed",
        "loader",
        "model",
        "request_id",
        "run_id",
        "selected_sections",
        "selected_skills",
        "task_success",
        "test_passed",
    }

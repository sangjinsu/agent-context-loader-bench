from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_loader_bench.report import aggregate_traces, format_table


def write_trace(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def make_record(loader: str, request_id: str, *, success: bool, tokens: int, llm: bool = True) -> dict:
    return {
        "run_id": f"{loader}:{request_id}",
        "request_id": request_id,
        "loader": loader,
        "model": "test-model" if llm else None,
        "selected_skills": [],
        "selected_sections": [],
        "context_token_estimate": tokens,
        "llm_completed": llm,
        "task_success": success,
        "test_passed": success,
    }


def test_aggregate_counts_success_tokens_and_llm_completed(tmp_path: Path) -> None:
    trace = tmp_path / "traces.jsonl"
    write_trace(
        trace,
        [
            make_record("fs_direct", "r1", success=True, tokens=1000),
            make_record("fs_direct", "r2", success=False, tokens=2000),
            make_record("hybrid", "r1", success=True, tokens=400, llm=False),
        ],
    )

    stats = {row.loader: row for row in aggregate_traces(trace)}

    fs = stats["fs_direct"]
    assert fs.total_requests == 2
    assert fs.task_successes == 1
    assert fs.success_rate == pytest.approx(0.5)
    assert fs.avg_context_tokens == pytest.approx(1500.0)
    assert fs.llm_completed == 2

    hybrid = stats["hybrid"]
    assert hybrid.success_rate == pytest.approx(1.0)
    assert hybrid.llm_completed == 0


def test_aggregate_dedupes_by_run_id_keeping_latest(tmp_path: Path) -> None:
    trace = tmp_path / "traces.jsonl"
    write_trace(
        trace,
        [
            make_record("fs_direct", "r1", success=False, tokens=9999),
            # Re-run of the same run_id: the later record must win.
            make_record("fs_direct", "r1", success=True, tokens=1000),
        ],
    )

    [row] = aggregate_traces(trace)

    assert row.total_requests == 1
    assert row.task_successes == 1
    assert row.avg_context_tokens == pytest.approx(1000.0)


def test_aggregate_sorts_by_success_rate_then_tokens(tmp_path: Path) -> None:
    trace = tmp_path / "traces.jsonl"
    write_trace(
        trace,
        [
            # Equal 100% success; section loader uses fewer tokens, so it ranks first.
            make_record("fs_direct", "r1", success=True, tokens=1400),
            make_record("sqlite_fts_section", "r1", success=True, tokens=800),
            # Lower success rate ranks last regardless of tokens.
            make_record("vector_search", "r1", success=False, tokens=10),
        ],
    )

    order = [row.loader for row in aggregate_traces(trace)]

    assert order == ["sqlite_fts_section", "fs_direct", "vector_search"]


def test_aggregate_missing_trace_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        aggregate_traces(tmp_path / "missing.jsonl")


def test_format_table_contains_header_and_rows(tmp_path: Path) -> None:
    trace = tmp_path / "traces.jsonl"
    write_trace(trace, [make_record("fs_direct", "r1", success=True, tokens=1000)])

    table = format_table(aggregate_traces(trace))

    assert "loader" in table
    assert "fs_direct" in table
    assert "1/1" in table

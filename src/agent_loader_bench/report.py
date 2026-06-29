from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LoaderStats:
    loader: str
    total_requests: int
    task_successes: int
    success_rate: float
    avg_context_tokens: float
    llm_completed: int


def _load_records(trace_path: Path) -> list[dict[str, Any]]:
    """Read a JSONL trace file into a list of records.

    Blank lines are skipped. A malformed line raises so the caller fails loudly
    rather than silently dropping data.
    """
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(trace_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on line {line_number} of {trace_path}: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Trace line {line_number} is not an object: {trace_path}")
        records.append(record)
    return records


def _dedupe_latest(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the most recent record per run_id.

    Traces are appended in execution order, so a later record for the same
    run_id reflects a re-run and should win. Records without a run_id are kept
    as-is (keyed by position) so nothing is silently discarded.
    """
    latest: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        key = str(record.get("run_id") or f"__no_run_id__:{index}")
        latest[key] = record
    return list(latest.values())


def _summarize_loader(loader: str, records: list[dict[str, Any]]) -> LoaderStats:
    """Reduce one loader's deduped records into a single stats row."""
    total = len(records)
    successes = sum(1 for record in records if record.get("task_success"))
    llm_completed = sum(1 for record in records if record.get("llm_completed"))
    token_values = [
        int(record.get("context_token_estimate") or 0)
        for record in records
        if record.get("context_token_estimate") is not None
    ]
    avg_tokens = (sum(token_values) / len(token_values)) if token_values else 0.0
    success_rate = (successes / total) if total else 0.0
    return LoaderStats(
        loader=loader,
        total_requests=total,
        task_successes=successes,
        success_rate=success_rate,
        avg_context_tokens=avg_tokens,
        llm_completed=llm_completed,
    )


def aggregate_traces(trace_path: Path) -> list[LoaderStats]:
    """Aggregate a trace file into per-loader stats.

    Deduplicates by run_id (latest wins), groups by loader, and sorts by
    success rate (desc) then average context tokens (asc) so the most accurate
    and most token-efficient loaders rise to the top.
    """
    records = _dedupe_latest(_load_records(Path(trace_path)))

    by_loader: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        loader = str(record.get("loader") or "<unknown>")
        by_loader.setdefault(loader, []).append(record)

    stats = [_summarize_loader(loader, loader_records) for loader, loader_records in by_loader.items()]
    stats.sort(key=lambda row: (-row.success_rate, row.avg_context_tokens, row.loader))
    return stats


def format_table(stats: list[LoaderStats]) -> str:
    """Render loader stats as a fixed-width text table for local comparison."""
    header = f"{'loader':<20} {'success':>9} {'rate':>7} {'avg_tokens':>11} {'llm_done':>9}"
    separator = "-" * len(header)
    lines = [header, separator]
    for row in stats:
        success = f"{row.task_successes}/{row.total_requests}"
        rate = f"{row.success_rate * 100:.0f}%"
        avg = f"{row.avg_context_tokens:.0f}"
        lines.append(f"{row.loader:<20} {success:>9} {rate:>7} {avg:>11} {row.llm_completed:>9}")
    return "\n".join(lines)

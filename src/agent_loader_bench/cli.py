from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from agent_loader_bench.config import load_settings
from agent_loader_bench.loaders import (
    build_deterministic_vector_index,
    build_json_document_store,
    build_manifest_index,
    build_sqlite_fts_index,
    build_sqlite_metadata_index,
)
from agent_loader_bench.runner import inspect_request, run_dataset


def main(argv: list[str] | None = None, *, repo_root: Path | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolved_root = Path(repo_root or Path.cwd()).resolve()
    settings = load_settings(resolved_root)

    try:
        if args.command == "build-manifest":
            output = Path(args.output) if args.output else resolved_root / ".agentdb" / "manifest.json"
            build_manifest_index(resolved_root, output)
            print(output)
            return 0

        if args.command == "build-index":
            created_paths = _build_indexes(resolved_root, args.backend)
            for path in created_paths:
                print(path)
            return 0

        if args.command == "inspect":
            result = inspect_request(
                repo_root=resolved_root,
                loader_name=args.loader,
                dataset_path=_resolve_repo_path(resolved_root, args.dataset),
                request_id=args.request_id,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0

        if args.command == "run":
            result = run_dataset(
                repo_root=resolved_root,
                loader_name=args.loader,
                dataset_path=_resolve_repo_path(resolved_root, args.dataset),
                live_llm=args.live_llm,
                trace_path=settings.trace_path,
                settings=settings,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0

        raise ValueError(f"Unknown command: {args.command}")
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m agent_loader_bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_manifest_parser = subparsers.add_parser("build-manifest")
    build_manifest_parser.add_argument("--output", default=None)

    build_index_parser = subparsers.add_parser("build-index")
    build_index_parser.add_argument(
        "--backend",
        required=True,
        choices=["sqlite", "sqlite_fts", "json_document", "vector", "all"],
    )

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--loader", required=True)
    inspect_parser.add_argument("--request-id", required=True)
    inspect_parser.add_argument("--dataset", default="datasets/requests.yml")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--loader", required=True)
    run_parser.add_argument("--dataset", default="datasets/requests.yml")
    run_parser.add_argument("--live-llm", action="store_true")

    return parser


def _build_indexes(repo_root: Path, backend: str) -> list[Path]:
    agentdb_root = repo_root / ".agentdb"
    builders = {
        "manifest_json": lambda: build_manifest_index(repo_root, agentdb_root / "manifest.json"),
        "sqlite": lambda: build_sqlite_metadata_index(repo_root, agentdb_root / "index.sqlite"),
        "sqlite_fts": lambda: build_sqlite_fts_index(repo_root, agentdb_root / "fts.sqlite"),
        "json_document": lambda: build_json_document_store(repo_root, agentdb_root / "document_store.jsonl"),
        "vector": lambda: build_deterministic_vector_index(repo_root, agentdb_root / "vector" / "index.json"),
    }

    if backend == "all":
        ordered = ["manifest_json", "sqlite", "sqlite_fts", "json_document", "vector"]
    else:
        ordered = [backend]

    return [builders[name]() for name in ordered]


def _resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path

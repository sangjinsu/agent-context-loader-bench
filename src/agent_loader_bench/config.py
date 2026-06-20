from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    openai_api_key: str | None
    llm_model: str
    llm_temperature: float
    agentdb_path: Path
    trace_path: Path


def load_settings(repo_root: Path | None = None) -> Settings:
    root = Path(repo_root or Path.cwd()).resolve()
    dotenv = _read_dotenv(root / ".env")
    return Settings(
        repo_root=root,
        openai_api_key=_optional_setting("OPENAI_API_KEY", dotenv),
        llm_model=_setting("LLM_MODEL", dotenv, "gpt-4.1-mini"),
        llm_temperature=float(_setting("LLM_TEMPERATURE", dotenv, "0")),
        agentdb_path=_resolve_path(root, _setting("AGENTDB_PATH", dotenv, ".agentdb/index.sqlite")),
        trace_path=_resolve_path(root, _setting("TRACE_PATH", dotenv, ".agentdb/traces.jsonl")),
    )


def _setting(name: str, dotenv: dict[str, str], default: str) -> str:
    return os.environ.get(name, dotenv.get(name, default))


def _optional_setting(name: str, dotenv: dict[str, str]) -> str | None:
    value = os.environ.get(name, dotenv.get(name))
    return value or None


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if not name:
            continue
        values[name] = value.strip().strip('"').strip("'")
    return values


def _resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path

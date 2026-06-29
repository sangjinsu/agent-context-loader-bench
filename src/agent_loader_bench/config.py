from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    openai_api_key: str | None
    anthropic_api_key: str | None
    llm_provider: str
    llm_model: str
    llm_temperature: float
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int | None
    embedding_min_score: float | None
    agentdb_path: Path
    trace_path: Path


def load_settings(repo_root: Path | None = None) -> Settings:
    root = Path(repo_root or Path.cwd()).resolve()
    dotenv = _read_dotenv(root / ".env")
    llm_model = _setting("LLM_MODEL", dotenv, "gpt-4.1-mini")
    return Settings(
        repo_root=root,
        openai_api_key=_optional_setting("OPENAI_API_KEY", dotenv),
        anthropic_api_key=_optional_setting("ANTHROPIC_API_KEY", dotenv),
        llm_provider=_resolve_provider(_optional_setting("LLM_PROVIDER", dotenv), llm_model),
        llm_model=llm_model,
        llm_temperature=float(_setting("LLM_TEMPERATURE", dotenv, "0")),
        embedding_provider=_setting("EMBEDDING_PROVIDER", dotenv, "openai").strip().lower(),
        embedding_model=_setting("EMBEDDING_MODEL", dotenv, "text-embedding-3-small"),
        embedding_dimensions=_optional_int("EMBEDDING_DIMENSIONS", dotenv),
        embedding_min_score=_optional_float("EMBEDDING_MIN_SCORE", dotenv),
        agentdb_path=_resolve_path(root, _setting("AGENTDB_PATH", dotenv, ".agentdb/index.sqlite")),
        trace_path=_resolve_path(root, _setting("TRACE_PATH", dotenv, ".agentdb/traces.jsonl")),
    )


def _optional_int(name: str, dotenv: dict[str, str]) -> int | None:
    value = _optional_setting(name, dotenv)
    return int(value) if value else None


def _optional_float(name: str, dotenv: dict[str, str]) -> float | None:
    value = _optional_setting(name, dotenv)
    return float(value) if value else None


def _resolve_provider(explicit: str | None, model: str) -> str:
    """Pick the LLM provider.

    An explicit LLM_PROVIDER wins; otherwise infer from the model id so a
    `claude-*` model routes to Anthropic and everything else to OpenAI.
    """
    if explicit:
        provider = explicit.strip().lower()
        if provider not in {"openai", "anthropic"}:
            raise ValueError(f"Unsupported LLM_PROVIDER '{explicit}'. Choose 'openai' or 'anthropic'.")
        return provider
    return "anthropic" if model.lower().startswith("claude") else "openai"


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

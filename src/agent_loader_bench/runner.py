from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import inspect
from typing import Any

import yaml

from agent_loader_bench.config import Settings, load_settings
from agent_loader_bench.llm import LLMClient, OpenAIResponsesClient
from agent_loader_bench.loaders import (
    FSDirectContextLoader,
    HybridContextLoader,
    JSONDocumentContextLoader,
    ManifestContextLoader,
    SQLiteFTSContextLoader,
    SQLiteFTSSectionContextLoader,
    SQLiteMetadataContextLoader,
    VectorContextLoader,
)
from agent_loader_bench.trace import append_trace


@dataclass(frozen=True)
class DatasetItem:
    id: str
    user_request: str
    expected_skills: list[str]
    expected_sections: list[str]
    task_type: str | None
    difficulty: str | None
    notes: str | None


def load_dataset(path: Path) -> list[DatasetItem]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(payload, list):
        raise ValueError(f"Dataset must be a list: {path}")

    items: list[DatasetItem] = []
    for raw_item in payload:
        if not isinstance(raw_item, dict):
            raise ValueError(f"Invalid dataset item in {path}: expected mapping")
        items.append(
            DatasetItem(
                id=str(raw_item["id"]),
                user_request=str(raw_item["user_request"]),
                expected_skills=[str(item) for item in raw_item.get("expected_skills", [])],
                expected_sections=[str(item) for item in raw_item.get("expected_sections", [])],
                task_type=_optional_string(raw_item.get("task_type")),
                difficulty=_optional_string(raw_item.get("difficulty")),
                notes=_optional_string(raw_item.get("notes")),
            )
        )
    return items


def create_loader(loader_name: str, repo_root: Path):
    root = Path(repo_root)
    loaders = {
        "fs_direct": FSDirectContextLoader(root),
        "manifest_json": ManifestContextLoader(root),
        "sqlite_metadata": SQLiteMetadataContextLoader(root),
        "sqlite_fts": SQLiteFTSContextLoader(root),
        "sqlite_fts_section": SQLiteFTSSectionContextLoader(root),
        "json_document": JSONDocumentContextLoader(root),
        "vector_search": VectorContextLoader(root),
        "hybrid": HybridContextLoader(root),
    }
    try:
        return loaders[loader_name]
    except KeyError as error:
        supported = ", ".join(sorted(loaders))
        raise ValueError(f"Unsupported loader '{loader_name}'. Choose one of: {supported}") from error


def get_dataset_item(dataset: list[DatasetItem], request_id: str) -> DatasetItem:
    for item in dataset:
        if item.id == request_id:
            return item
    raise ValueError(f"Dataset request id not found: {request_id}")


def evaluate_loaded_context(item: DatasetItem, loaded_context: Any) -> dict[str, Any]:
    matched_skills = sorted(set(item.expected_skills) & set(loaded_context.selected_skills))
    missing_skills = sorted(set(item.expected_skills) - set(loaded_context.selected_skills))
    unexpected_skills = sorted(set(loaded_context.selected_skills) - set(item.expected_skills))
    received_sections = set(loaded_context.selected_sections)
    context_text = getattr(loaded_context, "context_text", "")
    for section in item.expected_sections:
        if f"## {section}" in context_text:
            received_sections.add(section)
    matched_sections = sorted(set(item.expected_sections) & received_sections)
    missing_sections = sorted(set(item.expected_sections) - received_sections)
    task_success = not missing_skills and not unexpected_skills and not missing_sections
    return {
        "matched_expected_skills": matched_skills,
        "missing_expected_skills": missing_skills,
        "unexpected_skills": unexpected_skills,
        "matched_expected_sections": matched_sections,
        "missing_expected_sections": missing_sections,
        "task_success": task_success,
    }


def inspect_request(
    *,
    repo_root: Path,
    loader_name: str,
    dataset_path: Path,
    request_id: str,
) -> dict[str, Any]:
    dataset = load_dataset(dataset_path)
    item = get_dataset_item(dataset, request_id)
    loader = create_loader(loader_name, repo_root)
    loaded_context = loader.load(item.user_request, task_type=item.task_type)
    evaluation = evaluate_loaded_context(item, loaded_context)
    return {
        "request": asdict(item),
        "loader": loader_name,
        "selected_skills": loaded_context.selected_skills,
        "selected_sections": loaded_context.selected_sections,
        "context_token_estimate": loaded_context.context_token_estimate,
        "evaluation": evaluation,
    }


def run_dataset(
    *,
    repo_root: Path,
    loader_name: str,
    dataset_path: Path,
    live_llm: bool = False,
    trace_path: Path | None = None,
    llm_client: LLMClient | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    resolved_settings = settings or load_settings(repo_root)
    dataset = load_dataset(dataset_path)
    loader = create_loader(loader_name, repo_root)
    resolved_trace_path = trace_path or resolved_settings.trace_path
    client = _resolve_llm_client(
        live_llm=live_llm,
        llm_client=llm_client,
        settings=resolved_settings,
    )

    completed = 0
    passed = 0

    for item in dataset:
        loaded_context = loader.load(item.user_request, task_type=item.task_type)
        evaluation = evaluate_loaded_context(item, loaded_context)
        llm_completed = False

        if live_llm and client is not None:
            _generate_with_client(
                client,
                instructions=loaded_context.context_text,
                user_input=item.user_request,
                model=resolved_settings.llm_model,
                temperature=resolved_settings.llm_temperature,
            )
            llm_completed = True

        trace_record = {
            "run_id": f"{loader_name}:{item.id}",
            "request_id": item.id,
            "loader": loader_name,
            "model": resolved_settings.llm_model if live_llm else None,
            "selected_skills": loaded_context.selected_skills,
            "selected_sections": loaded_context.selected_sections,
            "context_token_estimate": loaded_context.context_token_estimate,
            "llm_completed": llm_completed,
            "task_success": evaluation["task_success"],
            "test_passed": evaluation["task_success"],
        }
        append_trace(resolved_trace_path, trace_record)
        completed += 1
        if evaluation["task_success"]:
            passed += 1

    return {
        "loader": loader_name,
        "dataset_path": str(dataset_path),
        "live_llm": live_llm,
        "trace_path": str(resolved_trace_path),
        "total_requests": completed,
        "task_successes": passed,
    }


def _resolve_llm_client(
    *,
    live_llm: bool,
    llm_client: LLMClient | None,
    settings: Settings,
) -> LLMClient | None:
    if not live_llm:
        return None
    if llm_client is not None:
        return llm_client
    return OpenAIResponsesClient(
        api_key=settings.openai_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )


def _generate_with_client(
    client: LLMClient,
    *,
    instructions: str,
    user_input: str,
    model: str,
    temperature: float,
) -> str:
    parameters = inspect.signature(client.generate).parameters
    kwargs: dict[str, Any] = {
        "instructions": instructions,
        "user_input": user_input,
    }
    if "model" in parameters:
        kwargs["model"] = model
    if "temperature" in parameters:
        kwargs["temperature"] = temperature
    return client.generate(**kwargs)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

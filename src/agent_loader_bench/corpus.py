from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import yaml

from .tokens import estimate_tokens


_FRONTMATTER_BOUNDARY = re.compile(r"^---\s*$", re.MULTILINE)
_HEADING_PATTERN = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$", re.MULTILINE)
_REQUIRED_SKILL_FIELDS = (
    "id",
    "type",
    "title",
    "description",
    "version",
    "status",
    "tags",
    "activation.keywords",
    "priority",
)


@dataclass(frozen=True)
class DocumentSection:
    title: str
    level: int
    content: str


@dataclass(frozen=True)
class CorpusDocument:
    path: Path
    title: str | None
    frontmatter: dict[str, Any]
    body: str
    sections: list[DocumentSection]
    raw_text: str
    token_estimate: int


@dataclass(frozen=True)
class SkillDocument(CorpusDocument):
    id: str
    type: str
    description: str
    version: str
    status: str
    tags: list[str]
    activation_keywords: list[str]
    priority: int


def load_documents(paths: list[Path] | tuple[Path, ...]) -> list[CorpusDocument]:
    ordered_paths = sorted((Path(path) for path in paths), key=lambda path: path.as_posix())
    return [load_skill_document(path) if _is_skill_path(path) else load_corpus_document(path) for path in ordered_paths]


def load_corpus_document(path: Path) -> CorpusDocument:
    document_path = Path(path)
    raw_text = document_path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(raw_text, document_path)
    return CorpusDocument(
        path=document_path,
        title=_extract_title(body),
        frontmatter=frontmatter,
        body=body,
        sections=extract_sections(body),
        raw_text=raw_text,
        token_estimate=estimate_tokens(body),
    )


def load_skill_document(path: Path) -> SkillDocument:
    base_document = load_corpus_document(path)
    metadata = base_document.frontmatter
    _validate_skill_frontmatter(metadata, Path(path))

    activation = metadata["activation"]
    return SkillDocument(
        path=base_document.path,
        title=_coerce_string(metadata["title"], "title", Path(path)),
        frontmatter=base_document.frontmatter,
        body=base_document.body,
        sections=base_document.sections,
        raw_text=base_document.raw_text,
        token_estimate=base_document.token_estimate,
        id=_coerce_string(metadata["id"], "id", Path(path)),
        type=_coerce_string(metadata["type"], "type", Path(path)),
        description=_coerce_string(metadata["description"], "description", Path(path)),
        version=_coerce_string(metadata["version"], "version", Path(path)),
        status=_coerce_string(metadata["status"], "status", Path(path)),
        tags=_coerce_string_list(metadata["tags"], "tags", Path(path)),
        activation_keywords=_coerce_string_list(activation["keywords"], "activation.keywords", Path(path)),
        priority=_coerce_int(metadata["priority"], "priority", Path(path)),
    )


def extract_sections(markdown_text: str) -> list[DocumentSection]:
    matches = [match for match in _HEADING_PATTERN.finditer(markdown_text) if len(match.group(1)) >= 2]
    sections: list[DocumentSection] = []

    for index, match in enumerate(matches):
        content_start = match.end()
        content_end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown_text)
        content = markdown_text[content_start:content_end].strip()
        sections.append(
            DocumentSection(
                title=_clean_heading_title(match.group(2)),
                level=len(match.group(1)),
                content=content,
            )
        )

    return sections


def _split_frontmatter(raw_text: str, path: Path) -> tuple[dict[str, Any], str]:
    if not raw_text.startswith("---"):
        return {}, raw_text

    matches = list(_FRONTMATTER_BOUNDARY.finditer(raw_text))
    if len(matches) < 2 or matches[0].start() != 0:
        raise ValueError(f"Invalid YAML frontmatter in {path}: missing closing '---' boundary")

    frontmatter_text = raw_text[matches[0].end() : matches[1].start()]
    body = raw_text[matches[1].end() :].lstrip("\n")

    loaded = yaml.safe_load(frontmatter_text) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid YAML frontmatter in {path}: expected mapping")

    return loaded, body


def _extract_title(markdown_text: str) -> str | None:
    for match in _HEADING_PATTERN.finditer(markdown_text):
        if len(match.group(1)) == 1:
            return _clean_heading_title(match.group(2))
    return None


def _validate_skill_frontmatter(metadata: dict[str, Any], path: Path) -> None:
    missing: list[str] = []
    for field_name in _REQUIRED_SKILL_FIELDS:
        if field_name == "activation.keywords":
            activation = metadata.get("activation")
            if not isinstance(activation, dict) or "keywords" not in activation:
                missing.append(field_name)
        elif field_name not in metadata:
            missing.append(field_name)

    if missing:
        raise ValueError(f"Missing required skill frontmatter fields in {path}: {', '.join(missing)}")

    _coerce_string(metadata["id"], "id", path)
    _coerce_string(metadata["type"], "type", path)
    _coerce_string(metadata["title"], "title", path)
    _coerce_string(metadata["description"], "description", path)
    _coerce_string(metadata["version"], "version", path)
    _coerce_string(metadata["status"], "status", path)
    _coerce_string_list(metadata["tags"], "tags", path)
    _coerce_string_list(metadata["activation"]["keywords"], "activation.keywords", path)
    _coerce_int(metadata["priority"], "priority", path)


def _coerce_string(value: Any, field_name: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid skill frontmatter field '{field_name}' in {path}: expected non-empty string")
    return value.strip()


def _coerce_string_list(value: Any, field_name: str, path: Path) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Invalid skill frontmatter field '{field_name}' in {path}: expected non-empty list")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Invalid skill frontmatter field '{field_name}' in {path}: expected list[str]")
        normalized.append(item.strip())
    return normalized


def _coerce_int(value: Any, field_name: str, path: Path) -> int:
    if not isinstance(value, int):
        raise ValueError(f"Invalid skill frontmatter field '{field_name}' in {path}: expected integer")
    return value


def _clean_heading_title(title: str) -> str:
    return title.strip().rstrip("#").strip()


def _is_skill_path(path: Path) -> bool:
    return path.name == "SKILL.md" and "skills" in path.parts

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from agent_loader_bench.corpus import (
    CorpusDocument,
    DocumentSection,
    SkillDocument,
    load_corpus_document,
    load_skill_document,
)
from agent_loader_bench.tokens import estimate_tokens


_TERM_PATTERN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "help",
    "i",
    "in",
    "is",
    "it",
    "need",
    "of",
    "on",
    "or",
    "please",
    "that",
    "the",
    "this",
    "to",
    "update",
    "use",
    "user",
    "when",
    "with",
}
_DOCS_ONLY_TERMS = {"docs", "documentation", "readme", "guide", "examples", "explain", "explaining"}
_DOCS_SKILL_TERMS = {"docs", "writing", "readme", "guide", "examples"}
_IMPLEMENTATION_SKILL_TERMS = {"benchmark", "vllm", "performance", "throughput", "latency"}
_NEGATION_TERMS = {"without", "not", "never"}


@dataclass(frozen=True)
class LoadedContext:
    loader_name: str
    selected_skills: list[str]
    selected_sections: list[str]
    context_text: str
    context_token_estimate: int
    debug: dict[str, Any]


@dataclass(frozen=True)
class ScoredDocument:
    document_id: str
    path: Path
    doc_type: str
    title: str
    score: float
    reason: str
    priority: int = 0


class ContextLoader(ABC):
    loader_name = "base"

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = Path(repo_root)

    @abstractmethod
    def load(self, request: str, task_type: str | None = None, top_k: int = 3) -> LoadedContext:
        raise NotImplementedError

    def build_loaded_context(
        self,
        *,
        selected_skills: Sequence[SkillDocument],
        selected_wiki: Sequence[CorpusDocument],
        request: str,
        debug: dict[str, Any],
        selected_skill_sections: dict[str, Sequence[DocumentSection]] | None = None,
        selected_wiki_sections: dict[str, Sequence[DocumentSection]] | None = None,
    ) -> LoadedContext:
        skill_sections_map = selected_skill_sections or {}
        wiki_sections_map = selected_wiki_sections or {}

        if not selected_skills and not selected_wiki:
            return LoadedContext(
                loader_name=self.loader_name,
                selected_skills=[],
                selected_sections=[],
                context_text="",
                context_token_estimate=0,
                debug=debug,
            )

        blocks = [
            "AGENTS.md excerpt:",
            self._read_agents_excerpt(),
            "",
            "User request:",
            request.strip(),
        ]

        selected_sections: list[str] = []

        for document in selected_skills:
            doc_sections = list(skill_sections_map.get(document.id, []))
            if doc_sections:
                selected_sections.extend(section.title for section in doc_sections)
            blocks.extend(self._format_document_block(document, document.id, doc_sections))

        for document in selected_wiki:
            document_id = wiki_document_id(self.repo_root, document.path)
            doc_sections = list(wiki_sections_map.get(document_id, []))
            if doc_sections:
                selected_sections.extend(section.title for section in doc_sections)
            blocks.extend(self._format_document_block(document, document_id, doc_sections))

        context_text = "\n".join(blocks).strip()
        return LoadedContext(
            loader_name=self.loader_name,
            selected_skills=[document.id for document in selected_skills],
            selected_sections=selected_sections,
            context_text=context_text,
            context_token_estimate=estimate_tokens(context_text),
            debug=debug,
        )

    def _format_document_block(
        self,
        document: CorpusDocument,
        document_id: str,
        sections: Sequence[DocumentSection],
    ) -> list[str]:
        header = f"[Skill: {document_id}]" if isinstance(document, SkillDocument) else f"[Wiki: {document_id}]"
        section_label = ", ".join(section.title for section in sections) if sections else "full document"
        content = render_section_content(document, sections)
        return [
            "",
            header,
            f"Path: {relative_repo_path(self.repo_root, document.path)}",
            f"Sections: {section_label}",
            "",
            content.strip(),
        ]

    def _read_agents_excerpt(self, max_non_empty_lines: int = 6) -> str:
        agents_path = self.repo_root / "AGENTS.md"
        lines = agents_path.read_text(encoding="utf-8").splitlines()
        excerpt: list[str] = []
        for line in lines:
            if line.strip():
                excerpt.append(line.rstrip())
            if len(excerpt) >= max_non_empty_lines:
                break
        return "\n".join(excerpt)


def iter_skill_documents(repo_root: Path) -> list[SkillDocument]:
    skill_paths = sorted((repo_root / "skills").glob("*/SKILL.md"), key=lambda path: path.as_posix())
    return [load_skill_document(path) for path in skill_paths]


def iter_wiki_documents(repo_root: Path) -> list[CorpusDocument]:
    wiki_root = repo_root / "wiki"
    wiki_paths = sorted(wiki_root.rglob("*.md"), key=lambda path: path.as_posix()) if wiki_root.exists() else []
    return [load_corpus_document(path) for path in wiki_paths]


def iter_source_markdown_paths(repo_root: Path) -> list[Path]:
    root = Path(repo_root)
    skill_paths = sorted((root / "skills").glob("*/SKILL.md"), key=lambda path: path.as_posix())
    wiki_root = root / "wiki"
    wiki_paths = sorted(wiki_root.rglob("*.md"), key=lambda path: path.as_posix()) if wiki_root.exists() else []
    return [*skill_paths, *wiki_paths]


def corpus_fingerprint(repo_root: Path) -> str:
    root = Path(repo_root)
    digest = hashlib.sha256()
    for path in iter_source_markdown_paths(root):
        digest.update(relative_repo_path(root, path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def assert_fresh_index(repo_root: Path, index_fingerprint: str | None, index_path: Path) -> None:
    current_fingerprint = corpus_fingerprint(repo_root)
    if not index_fingerprint or index_fingerprint != current_fingerprint:
        raise RuntimeError(f"stale index: {index_path}. Rebuild the index explicitly before loading context.")


def relative_repo_path(repo_root: Path, path: Path) -> str:
    return Path(path).resolve().relative_to(repo_root.resolve()).as_posix()


def wiki_document_id(repo_root: Path, path: Path) -> str:
    return f"wiki:{relative_repo_path(repo_root, path)}"


def normalize_terms(text: str) -> list[str]:
    return [term for term in _TERM_PATTERN.findall(text.lower()) if term not in _STOPWORDS]


def unique_terms(text: str) -> set[str]:
    return set(normalize_terms(text))


def render_section_content(document: CorpusDocument, sections: Sequence[DocumentSection]) -> str:
    if not sections:
        return document.body.strip()
    return "\n\n".join(f"## {section.title}\n\n{section.content.strip()}".strip() for section in sections).strip()


def metadata_from_document(repo_root: Path, document: CorpusDocument) -> dict[str, Any]:
    if isinstance(document, SkillDocument):
        return {
            "id": document.id,
            "doc_type": "skill",
            "path": relative_repo_path(repo_root, document.path),
            "title": document.title or document.id,
            "description": document.description,
            "priority": document.priority,
            "tags": list(document.tags),
            "activation_keywords": list(document.activation_keywords),
            "body": document.body,
            "sections": [section.title for section in document.sections],
        }

    return {
        "id": wiki_document_id(repo_root, document.path),
        "doc_type": "wiki",
        "path": relative_repo_path(repo_root, document.path),
        "title": document.title or document.path.stem,
        "description": document.title or document.path.stem,
        "priority": 0,
        "tags": [],
        "activation_keywords": [],
        "body": document.body,
        "sections": [section.title for section in document.sections],
    }


def score_skill_metadata(metadata: dict[str, Any], request: str, task_type: str | None = None) -> tuple[float, str]:
    request_terms = unique_terms(request)
    keyword_terms = set(_flatten_terms(metadata.get("activation_keywords", [])))
    title_terms = unique_terms(str(metadata.get("title", "")))
    description_terms = unique_terms(str(metadata.get("description", "")))
    tag_terms = set(_flatten_terms(metadata.get("tags", [])))

    if _should_suppress_docs_only_implementation_skill(request_terms, tag_terms, keyword_terms, task_type):
        return -1000.0, "suppressed=docs-only request excludes implementation skill"

    keyword_hits = sorted(request_terms & keyword_terms)
    title_hits = sorted(request_terms & title_terms)
    description_hits = sorted(request_terms & description_terms)
    tag_hits = sorted(request_terms & tag_terms)

    score = len(keyword_hits) * 10 + len(title_hits) * 6 + len(description_hits) * 4 + len(tag_hits) * 3
    if task_type and task_type.lower() in tag_terms:
        score += 20

    reasons: list[str] = []
    if keyword_hits:
        reasons.append(f"keyword={','.join(keyword_hits)}")
    if title_hits:
        reasons.append(f"title={','.join(title_hits)}")
    if description_hits:
        reasons.append(f"description={','.join(description_hits)}")
    if tag_hits:
        reasons.append(f"tags={','.join(tag_hits)}")
    if task_type and task_type.lower() in tag_terms:
        reasons.append(f"task_type={task_type.lower()}")
    return score, "; ".join(reasons) if reasons else "no term overlap"


def score_wiki_metadata(metadata: dict[str, Any], request: str, related_terms: Iterable[str] = ()) -> tuple[float, str]:
    request_terms = unique_terms(request)
    related_term_set = set(related_terms)
    title_terms = unique_terms(str(metadata.get("title", "")))
    body_terms = unique_terms(str(metadata.get("body", "")))
    section_terms = set(_flatten_terms(metadata.get("sections", [])))

    title_hits = sorted(request_terms & title_terms)
    body_hits = sorted(request_terms & body_terms)
    section_hits = sorted(request_terms & section_terms)
    related_hits = sorted(related_term_set & body_terms)

    score = len(title_hits) * 4 + len(body_hits) * 2 + len(section_hits) * 2 + len(related_hits)
    reasons: list[str] = []
    if title_hits:
        reasons.append(f"title={','.join(title_hits)}")
    if body_hits:
        reasons.append(f"body={','.join(body_hits)}")
    if section_hits:
        reasons.append(f"sections={','.join(section_hits)}")
    if related_hits:
        reasons.append(f"related={','.join(related_hits)}")
    return score, "; ".join(reasons) if reasons else "no term overlap"


def sort_scored_documents(documents: Sequence[ScoredDocument]) -> list[ScoredDocument]:
    return sorted(
        documents,
        key=lambda item: (-item.score, -item.priority, item.path.as_posix(), item.document_id),
    )


def load_document_by_metadata(repo_root: Path, metadata: dict[str, Any]) -> CorpusDocument:
    source_path = repo_root / metadata["path"]
    if metadata["doc_type"] == "skill":
        return load_skill_document(source_path)
    return load_corpus_document(source_path)


def select_relevant_sections(document: CorpusDocument, request: str) -> list[DocumentSection]:
    request_terms = unique_terms(request)
    scored_sections: list[tuple[int, int, DocumentSection]] = []
    for index, section in enumerate(document.sections):
        section_terms = unique_terms(section.title) | unique_terms(section.content)
        overlap = len(request_terms & section_terms)
        if overlap > 0:
            scored_sections.append((overlap, index, section))

    scored_sections.sort(key=lambda item: (-item[0], item[1], item[2].title))
    return [item[2] for item in scored_sections]


def _flatten_terms(values: Sequence[str]) -> list[str]:
    flattened: list[str] = []
    for value in values:
        flattened.extend(normalize_terms(str(value)))
    return flattened


def _should_suppress_docs_only_implementation_skill(
    request_terms: set[str],
    tag_terms: set[str],
    keyword_terms: set[str],
    task_type: str | None,
) -> bool:
    if (task_type or "").lower() != "docs":
        return False
    if not request_terms & _DOCS_ONLY_TERMS:
        return False
    if not request_terms & _NEGATION_TERMS:
        return False
    if tag_terms & _DOCS_SKILL_TERMS:
        return False
    return bool((tag_terms | keyword_terms) & _IMPLEMENTATION_SKILL_TERMS)

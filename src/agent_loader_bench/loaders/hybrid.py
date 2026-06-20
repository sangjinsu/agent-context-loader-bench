from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable

from .base import ContextLoader, LoadedContext, load_document_by_metadata
from .fs_direct import FSDirectContextLoader
from .manifest_json import ManifestContextLoader
from .sqlite_fts import SQLiteFTSSectionContextLoader
from .sqlite_metadata import SQLiteMetadataContextLoader
from .vector_search import VectorContextLoader


class HybridContextLoader(ContextLoader):
    loader_name = "hybrid"

    def __init__(
        self,
        repo_root: Path,
        *,
        manifest_path: Path | None = None,
        metadata_index_path: Path | None = None,
        fts_index_path: Path | None = None,
        vector_index_path: Path | None = None,
    ) -> None:
        super().__init__(repo_root)
        self.manifest_path = manifest_path or (self.repo_root / ".agentdb" / "manifest.json")
        self.metadata_index_path = metadata_index_path or (self.repo_root / ".agentdb" / "index.sqlite")
        self.fts_index_path = fts_index_path or (self.repo_root / ".agentdb" / "fts.sqlite")
        self.vector_index_path = vector_index_path or (self.repo_root / ".agentdb" / "vector" / "index.json")

    def load(self, request: str, task_type: str | None = None, top_k: int = 3) -> LoadedContext:
        fallbacks: list[str] = []
        provider_contexts: list[tuple[str, LoadedContext]] = []

        providers: list[tuple[str, Callable[[], LoadedContext]]] = [
            (
                "sqlite_fts_section",
                lambda: SQLiteFTSSectionContextLoader(self.repo_root, index_path=self.fts_index_path).load(
                    request,
                    task_type=task_type,
                    top_k=top_k,
                ),
            ),
            (
                "sqlite_metadata",
                lambda: SQLiteMetadataContextLoader(self.repo_root, index_path=self.metadata_index_path).load(
                    request,
                    task_type=task_type,
                    top_k=top_k,
                ),
            ),
            (
                "vector_search",
                lambda: VectorContextLoader(self.repo_root, index_path=self.vector_index_path).load(
                    request,
                    task_type=task_type,
                    top_k=top_k,
                ),
            ),
            (
                "manifest_json",
                lambda: ManifestContextLoader(self.repo_root, manifest_path=self.manifest_path).load(
                    request,
                    task_type=task_type,
                    top_k=top_k,
                ),
            ),
            (
                "fs_direct",
                lambda: FSDirectContextLoader(self.repo_root).load(
                    request,
                    task_type=task_type,
                    top_k=top_k,
                ),
            ),
        ]

        for provider_name, provider in providers:
            try:
                context = provider()
            except (FileNotFoundError, RuntimeError) as error:
                fallbacks.append(f"{provider_name}: {error}")
                continue

            if context.selected_skills or context.context_text:
                provider_contexts.append((provider_name, context))
                continue

            fallbacks.append(f"{provider_name}: no match")

        if not provider_contexts:
            return LoadedContext(
                loader_name=self.loader_name,
                selected_skills=[],
                selected_sections=[],
                context_text="",
                context_token_estimate=0,
                debug={"fallbacks": fallbacks, "selected_via": None, "providers_used": []},
            )

        indexed_provider_contexts = [
            (provider_name, context) for provider_name, context in provider_contexts if provider_name != "fs_direct"
        ]
        if indexed_provider_contexts:
            provider_contexts = indexed_provider_contexts

        combined_scores: dict[str, dict[str, object]] = {}
        section_titles_by_document: dict[str, list[str]] = defaultdict(list)

        for provider_name, context in provider_contexts:
            for document_id, titles in context.debug.get("selected_sections_by_document", {}).items():
                seen_titles = section_titles_by_document[document_id]
                for title in titles:
                    if title not in seen_titles:
                        seen_titles.append(title)

            for item in context.debug.get("selection", []):
                score = float(item.get("score", 0.0))
                if score <= 0:
                    continue

                entry = combined_scores.setdefault(
                    item["id"],
                    {
                        "doc_type": item["doc_type"],
                        "path": item["path"],
                        "combined_score": 0.0,
                        "providers": {},
                    },
                )
                entry["combined_score"] = float(entry["combined_score"]) + score
                providers_used = entry["providers"]
                assert isinstance(providers_used, dict)
                providers_used[provider_name] = {
                    "score": score,
                    "reason": item.get("reason", ""),
                }

        if not combined_scores:
            return LoadedContext(
                loader_name=self.loader_name,
                selected_skills=[],
                selected_sections=[],
                context_text="",
                context_token_estimate=0,
                debug={
                    "fallbacks": fallbacks,
                    "selected_via": None,
                    "providers_used": [provider_name for provider_name, _ in provider_contexts],
                    "combined_scores": {},
                },
            )

        ranked_items = sorted(
            combined_scores.items(),
            key=lambda item: (
                -float(item[1]["combined_score"]),
                str(item[1]["path"]),
                item[0],
            ),
        )
        skill_metadata = _filter_combined_documents(
            [
                {
                    "id": document_id,
                    "doc_type": str(payload["doc_type"]),
                    "path": str(payload["path"]),
                    "score": float(payload["combined_score"]),
                }
                for document_id, payload in ranked_items
                if payload["doc_type"] == "skill"
            ],
            top_k=top_k,
        )
        wiki_metadata = _filter_combined_documents(
            [
                {
                    "id": document_id,
                    "doc_type": str(payload["doc_type"]),
                    "path": str(payload["path"]),
                    "score": float(payload["combined_score"]),
                }
                for document_id, payload in ranked_items
                if payload["doc_type"] == "wiki"
            ],
            top_k=top_k,
        )

        selected_skills = [
            load_document_by_metadata(
                self.repo_root,
                {"id": metadata["id"], "doc_type": metadata["doc_type"], "path": metadata["path"]},
            )
            for metadata in skill_metadata
        ]
        selected_wiki = [
            load_document_by_metadata(
                self.repo_root,
                {"id": metadata["id"], "doc_type": metadata["doc_type"], "path": metadata["path"]},
            )
            for metadata in wiki_metadata
        ]
        selected_skill_sections = {
            document.id: [
                section for section in document.sections if section.title in section_titles_by_document[document.id]
            ]
            for document in selected_skills
            if section_titles_by_document.get(document.id)
        }
        selected_wiki_sections = {
            metadata["id"]: [
                section for section in document.sections if section.title in section_titles_by_document[metadata["id"]]
            ]
            for metadata, document in zip(wiki_metadata, selected_wiki, strict=True)
            if section_titles_by_document.get(metadata["id"])
        }

        debug = {
            "fallbacks": fallbacks,
            "selected_via": provider_contexts[0][0] if len(provider_contexts) == 1 else "combined",
            "providers_used": [provider_name for provider_name, _ in provider_contexts],
            "combined_scores": combined_scores,
            "selected_sections_by_document": {
                document_id: list(titles) for document_id, titles in section_titles_by_document.items()
            },
        }
        return self.build_loaded_context(
            selected_skills=selected_skills,
            selected_wiki=selected_wiki,
            request=request,
            debug=debug,
            selected_skill_sections=selected_skill_sections,
            selected_wiki_sections=selected_wiki_sections,
        )


def _filter_combined_documents(documents: list[dict[str, object]], *, top_k: int) -> list[dict[str, object]]:
    if not documents:
        return []

    top_score = float(documents[0]["score"])
    minimum_score = max(0.2, top_score * 0.8)
    return [
        {
            "id": str(document["id"]),
            "doc_type": str(document["doc_type"]),
            "path": str(document["path"]),
            "score": float(document["score"]),
        }
        for document in documents
        if float(document["score"]) >= minimum_score
    ][:top_k]

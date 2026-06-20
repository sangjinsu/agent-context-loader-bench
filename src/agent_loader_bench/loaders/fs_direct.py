from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import (
    ContextLoader,
    CorpusDocument,
    LoadedContext,
    ScoredDocument,
    SkillDocument,
    iter_skill_documents,
    iter_wiki_documents,
    metadata_from_document,
    score_skill_metadata,
    score_wiki_metadata,
    sort_scored_documents,
)


class FSDirectContextLoader(ContextLoader):
    loader_name = "fs_direct"

    def __init__(self, repo_root: Path) -> None:
        super().__init__(repo_root)
        self._skill_documents = iter_skill_documents(self.repo_root)
        self._wiki_documents = iter_wiki_documents(self.repo_root)

    def load(self, request: str, task_type: str | None = None, top_k: int = 3) -> LoadedContext:
        scored_skills: list[ScoredDocument] = []
        skill_by_id: dict[str, SkillDocument] = {}

        for document in self._skill_documents:
            metadata = metadata_from_document(self.repo_root, document)
            score, reason = score_skill_metadata(metadata, request, task_type)
            scored_skills.append(
                ScoredDocument(
                    document_id=document.id,
                    path=document.path,
                    doc_type="skill",
                    title=document.title or document.id,
                    score=score,
                    reason=reason,
                    priority=document.priority,
                )
            )
            skill_by_id[document.id] = document

        ranked_skills = sort_scored_documents(scored_skills)
        selected_skill_scores = [item for item in ranked_skills if item.score > 0][:top_k]
        selected_skills = [skill_by_id[item.document_id] for item in selected_skill_scores]

        selected_skill_terms = {
            term
            for document in selected_skills
            for term in document.activation_keywords + document.tags + [document.title or ""]
        }
        wiki_by_id: dict[str, CorpusDocument] = {}
        scored_wiki: list[ScoredDocument] = []
        for document in self._wiki_documents:
            metadata = metadata_from_document(self.repo_root, document)
            score, reason = score_wiki_metadata(metadata, request, selected_skill_terms)
            document_id = metadata["id"]
            scored_wiki.append(
                ScoredDocument(
                    document_id=document_id,
                    path=document.path,
                    doc_type="wiki",
                    title=document.title or document.path.stem,
                    score=float(score),
                    reason=reason,
                )
            )
            wiki_by_id[document_id] = document

        ranked_wiki = sort_scored_documents(scored_wiki)
        selected_wiki_scores = [item for item in ranked_wiki if item.score > 0][:top_k]
        selected_wiki = [wiki_by_id[item.document_id] for item in selected_wiki_scores]

        debug: dict[str, Any] = {
            "index_path": "skills/**/SKILL.md + wiki/**/*.md",
            "selected": [item.document_id for item in selected_skill_scores],
            "selection": [
                {
                    "id": item.document_id,
                    "doc_type": item.doc_type,
                    "score": item.score,
                    "reason": item.reason,
                    "path": item.path.as_posix(),
                }
                for item in ranked_skills + ranked_wiki
            ],
        }
        return self.build_loaded_context(
            selected_skills=selected_skills,
            selected_wiki=selected_wiki,
            request=request,
            debug=debug,
        )

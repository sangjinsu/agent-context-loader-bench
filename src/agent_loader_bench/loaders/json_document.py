from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import (
    ContextLoader,
    LoadedContext,
    ScoredDocument,
    assert_fresh_index,
    corpus_fingerprint,
    iter_skill_documents,
    iter_wiki_documents,
    load_document_by_metadata,
    metadata_from_document,
    score_skill_metadata,
    score_wiki_metadata,
    sort_scored_documents,
)


def build_json_document_store(repo_root: Path, store_path: Path) -> Path:
    root = Path(repo_root)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    documents = [metadata_from_document(root, document) for document in iter_skill_documents(root)]
    documents.extend(metadata_from_document(root, document) for document in iter_wiki_documents(root))
    lines = [json.dumps({"_meta": {"corpus_fingerprint": corpus_fingerprint(root)}}, sort_keys=True)]
    lines.extend(json.dumps(document, sort_keys=True) for document in sorted(documents, key=lambda item: item["id"]))
    store_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return store_path


class JSONDocumentContextLoader(ContextLoader):
    loader_name = "json_document"

    def __init__(self, repo_root: Path, store_path: Path | None = None) -> None:
        super().__init__(repo_root)
        self.store_path = store_path or (self.repo_root / ".agentdb" / "document_store.jsonl")

    def load(self, request: str, task_type: str | None = None, top_k: int = 3) -> LoadedContext:
        if not self.store_path.exists():
            raise FileNotFoundError(f"Missing JSON document store: {self.store_path}")

        lines = [json.loads(line) for line in self.store_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        meta = lines[0].get("_meta", {}) if lines else {}
        assert_fresh_index(self.repo_root, meta.get("corpus_fingerprint"), self.store_path)
        documents = [line for line in lines if "_meta" not in line]

        skill_records: dict[str, dict[str, Any]] = {}
        wiki_records: dict[str, dict[str, Any]] = {}
        skill_scores: list[ScoredDocument] = []
        wiki_scores: list[ScoredDocument] = []

        for metadata in documents:
            if metadata["doc_type"] == "skill":
                score, reason = score_skill_metadata(metadata, request, task_type)
                skill_records[metadata["id"]] = metadata
                skill_scores.append(
                    ScoredDocument(
                        document_id=metadata["id"],
                        path=self.repo_root / metadata["path"],
                        doc_type="skill",
                        title=metadata["title"],
                        score=score,
                        reason=reason,
                        priority=int(metadata.get("priority", 0)),
                    )
                )
            else:
                score, reason = score_wiki_metadata(metadata, request)
                wiki_records[metadata["id"]] = metadata
                wiki_scores.append(
                    ScoredDocument(
                        document_id=metadata["id"],
                        path=self.repo_root / metadata["path"],
                        doc_type="wiki",
                        title=metadata["title"],
                        score=float(score),
                        reason=reason,
                    )
                )

        ranked_skills = sort_scored_documents(skill_scores)
        ranked_wiki = sort_scored_documents(wiki_scores)
        selected_skill_meta = [skill_records[item.document_id] for item in ranked_skills if item.score > 0][:top_k]
        selected_wiki_meta = [wiki_records[item.document_id] for item in ranked_wiki if item.score > 0][:top_k]

        debug = {
            "index_path": self.store_path.as_posix(),
            "selected": [item["id"] for item in selected_skill_meta],
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
            selected_skills=[load_document_by_metadata(self.repo_root, metadata) for metadata in selected_skill_meta],
            selected_wiki=[load_document_by_metadata(self.repo_root, metadata) for metadata in selected_wiki_meta],
            request=request,
            debug=debug,
        )

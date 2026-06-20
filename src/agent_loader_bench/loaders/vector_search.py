from __future__ import annotations

from hashlib import sha256
import json
import math
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
    normalize_terms,
    sort_scored_documents,
)


_DEFAULT_VECTOR_DIMENSIONS = 24


def build_deterministic_vector_index(
    repo_root: Path, index_path: Path, dimensions: int = _DEFAULT_VECTOR_DIMENSIONS
) -> Path:
    root = Path(repo_root)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    documents = [metadata_from_document(root, document) for document in iter_skill_documents(root)]
    documents.extend(metadata_from_document(root, document) for document in iter_wiki_documents(root))

    payload = {
        "corpus_fingerprint": corpus_fingerprint(root),
        "dimensions": dimensions,
        "documents": [
            {
                **metadata,
                "vector": deterministic_text_vector(_metadata_vector_text(metadata), dimensions=dimensions),
            }
            for metadata in sorted(documents, key=lambda item: item["id"])
        ],
    }
    index_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index_path


def deterministic_text_vector(text: str, dimensions: int = _DEFAULT_VECTOR_DIMENSIONS) -> list[float]:
    values = [0.0] * dimensions
    for term in normalize_terms(text):
        digest = sha256(term.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        weight = 1.0 + (digest[5] / 255.0)
        values[bucket] += sign * weight

    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return values
    return [value / norm for value in values]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(left_item * right_item for left_item, right_item in zip(left, right, strict=True))


class VectorContextLoader(ContextLoader):
    loader_name = "vector_search"

    def __init__(self, repo_root: Path, index_path: Path | None = None) -> None:
        super().__init__(repo_root)
        self.index_path = index_path or (self.repo_root / ".agentdb" / "vector" / "index.json")

    def load(self, request: str, task_type: str | None = None, top_k: int = 3) -> LoadedContext:
        if not self.index_path.exists():
            raise FileNotFoundError(f"Missing vector index: {self.index_path}")

        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        assert_fresh_index(self.repo_root, payload.get("corpus_fingerprint"), self.index_path)
        dimensions = int(payload["dimensions"])
        query_vector = deterministic_text_vector(request, dimensions=dimensions)

        skill_records: dict[str, dict[str, Any]] = {}
        wiki_records: dict[str, dict[str, Any]] = {}
        skill_scores: list[ScoredDocument] = []
        wiki_scores: list[ScoredDocument] = []

        for metadata in payload["documents"]:
            similarity = cosine_similarity(query_vector, metadata["vector"])
            scored = ScoredDocument(
                document_id=metadata["id"],
                path=self.repo_root / metadata["path"],
                doc_type=metadata["doc_type"],
                title=metadata["title"],
                score=similarity,
                reason=f"cosine={similarity:.6f}",
                priority=int(metadata.get("priority", 0)),
            )
            if metadata["doc_type"] == "skill":
                skill_records[metadata["id"]] = metadata
                skill_scores.append(scored)
            else:
                wiki_records[metadata["id"]] = metadata
                wiki_scores.append(scored)

        ranked_skills = sort_scored_documents(skill_scores)
        ranked_wiki = sort_scored_documents(wiki_scores)
        selected_skill_meta = [
            skill_records[item.document_id] for item in _filter_similar_documents(ranked_skills, top_k=top_k)
        ]
        selected_wiki_meta = [
            wiki_records[item.document_id] for item in _filter_similar_documents(ranked_wiki, top_k=top_k)
        ]

        debug = {
            "index_path": self.index_path.as_posix(),
            "selected": [item["id"] for item in selected_skill_meta],
            "selection": [
                {
                    "id": item.document_id,
                    "doc_type": item.doc_type,
                    "score": item.score,
                    "similarity": item.score,
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


def _metadata_vector_text(metadata: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(metadata.get("title", "")),
            str(metadata.get("description", "")),
            " ".join(metadata.get("tags", [])),
            " ".join(metadata.get("activation_keywords", [])),
            str(metadata.get("body", "")),
        ]
    )


def _filter_similar_documents(documents: list[ScoredDocument], top_k: int) -> list[ScoredDocument]:
    if not documents:
        return []

    top_score = documents[0].score
    minimum_score = max(0.2, top_score * 0.8)
    return [item for item in documents if item.score >= minimum_score][:top_k]

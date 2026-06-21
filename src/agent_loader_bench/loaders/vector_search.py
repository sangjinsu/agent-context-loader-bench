from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_loader_bench.embeddings import (
    DEFAULT_HASHING_DIMENSIONS,
    Embedder,
    HashingEmbedder,
    deterministic_text_vector,  # re-exported for backward compatibility
    embedder_from_index,
)

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
    sort_scored_documents,
)

__all__ = [
    "VectorContextLoader",
    "build_vector_index",
    "build_deterministic_vector_index",
    "cosine_similarity",
    "default_min_score",
    "deterministic_text_vector",
]

# Minimum cosine to consider a document relevant, by embedding provider.
# Real embeddings (openai) give unrelated text a higher baseline cosine (~0.2),
# so they need a higher floor than the near-zero hashing space.
_DEFAULT_MIN_SCORES = {"hashing": 0.2, "openai": 0.28}


def default_min_score(provider: str | None) -> float:
    return _DEFAULT_MIN_SCORES.get((provider or "hashing").lower(), 0.2)


def build_vector_index(repo_root: Path, index_path: Path, embedder: Embedder) -> Path:
    """Build a vector index by embedding each document with `embedder`.

    The provider/model/dimensions are recorded so the loader can embed queries
    with the same model later (see `embedder_from_index`).
    """
    root = Path(repo_root)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    documents = [metadata_from_document(root, document) for document in iter_skill_documents(root)]
    documents.extend(metadata_from_document(root, document) for document in iter_wiki_documents(root))
    documents = sorted(documents, key=lambda item: item["id"])

    texts = [_metadata_vector_text(metadata) for metadata in documents]
    vectors = embedder.embed_documents(texts)

    dimensions = len(vectors[0]) if vectors else int(getattr(embedder, "dimensions", 0) or 0)
    payload = {
        "corpus_fingerprint": corpus_fingerprint(root),
        "embedding_provider": embedder.provider,
        "embedding_model": embedder.model,
        "dimensions": dimensions,
        "documents": [{**metadata, "vector": vector} for metadata, vector in zip(documents, vectors, strict=True)],
    }
    index_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index_path


def build_deterministic_vector_index(
    repo_root: Path, index_path: Path, dimensions: int = DEFAULT_HASHING_DIMENSIONS
) -> Path:
    """Offline hashing index (no network). Kept for tests and offline use."""
    return build_vector_index(repo_root, index_path, HashingEmbedder(dimensions=dimensions))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(left_item * right_item for left_item, right_item in zip(left, right, strict=True))


class VectorContextLoader(ContextLoader):
    loader_name = "vector_search"

    def __init__(
        self,
        repo_root: Path,
        index_path: Path | None = None,
        *,
        embedder: Embedder | None = None,
        settings: Any | None = None,
    ) -> None:
        super().__init__(repo_root)
        self.index_path = index_path or (self.repo_root / ".agentdb" / "vector" / "index.json")
        self._embedder = embedder
        self._settings = settings

    def load(self, request: str, task_type: str | None = None, top_k: int = 3) -> LoadedContext:
        if not self.index_path.exists():
            raise FileNotFoundError(f"Missing vector index: {self.index_path}")

        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        assert_fresh_index(self.repo_root, payload.get("corpus_fingerprint"), self.index_path)
        embedder = self._embedder or embedder_from_index(payload, self._settings)
        query_vector = embedder.embed_query(request)

        provider = str(payload.get("embedding_provider", "hashing"))
        configured_min = getattr(self._settings, "embedding_min_score", None)
        min_score = float(configured_min) if configured_min is not None else default_min_score(provider)

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
            skill_records[item.document_id]
            for item in _filter_similar_documents(ranked_skills, top_k=top_k, min_score=min_score)
        ]
        selected_wiki_meta = [
            wiki_records[item.document_id]
            for item in _filter_similar_documents(ranked_wiki, top_k=top_k, min_score=min_score)
        ]

        debug = {
            "index_path": self.index_path.as_posix(),
            "embedding_provider": payload.get("embedding_provider"),
            "embedding_model": payload.get("embedding_model"),
            "min_score": min_score,
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


def _filter_similar_documents(
    documents: list[ScoredDocument], top_k: int, min_score: float = 0.2
) -> list[ScoredDocument]:
    if not documents:
        return []

    top_score = documents[0].score
    minimum_score = max(min_score, top_score * 0.8)
    return [item for item in documents if item.score >= minimum_score][:top_k]

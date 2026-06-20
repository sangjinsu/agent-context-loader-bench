from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sqlite3
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


def build_sqlite_metadata_index(repo_root: Path, index_path: Path) -> Path:
    root = Path(repo_root)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(index_path)
    try:
        cursor = connection.cursor()
        cursor.executescript(
            """
            DROP TABLE IF EXISTS documents;
            DROP TABLE IF EXISTS document_tags;
            DROP TABLE IF EXISTS document_keywords;
            DROP TABLE IF EXISTS index_metadata;

            CREATE TABLE index_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                doc_type TEXT NOT NULL,
                path TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                priority INTEGER NOT NULL,
                body TEXT NOT NULL
            );

            CREATE TABLE document_tags (
                document_id TEXT NOT NULL,
                tag TEXT NOT NULL
            );

            CREATE TABLE document_keywords (
                document_id TEXT NOT NULL,
                keyword TEXT NOT NULL
            );
            """
        )
        cursor.execute(
            "INSERT INTO index_metadata (key, value) VALUES (?, ?)",
            ("corpus_fingerprint", corpus_fingerprint(root)),
        )

        for document in list(iter_skill_documents(root)) + list(iter_wiki_documents(root)):
            metadata = metadata_from_document(root, document)
            cursor.execute(
                """
                INSERT INTO documents (id, doc_type, path, title, description, priority, body)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata["id"],
                    metadata["doc_type"],
                    metadata["path"],
                    metadata["title"],
                    metadata["description"],
                    int(metadata.get("priority", 0)),
                    metadata["body"],
                ),
            )
            cursor.executemany(
                "INSERT INTO document_tags (document_id, tag) VALUES (?, ?)",
                [(metadata["id"], tag) for tag in metadata.get("tags", [])],
            )
            cursor.executemany(
                "INSERT INTO document_keywords (document_id, keyword) VALUES (?, ?)",
                [(metadata["id"], keyword) for keyword in metadata.get("activation_keywords", [])],
            )

        connection.commit()
    finally:
        connection.close()

    return index_path


class SQLiteMetadataContextLoader(ContextLoader):
    loader_name = "sqlite_metadata"

    def __init__(self, repo_root: Path, index_path: Path | None = None) -> None:
        super().__init__(repo_root)
        self.index_path = index_path or (self.repo_root / ".agentdb" / "index.sqlite")

    def load(self, request: str, task_type: str | None = None, top_k: int = 3) -> LoadedContext:
        if not self.index_path.exists():
            raise FileNotFoundError(f"Missing SQLite metadata index: {self.index_path}")

        connection = sqlite3.connect(self.index_path)
        connection.row_factory = sqlite3.Row
        try:
            fingerprint_row = connection.execute(
                "SELECT value FROM index_metadata WHERE key = ?",
                ("corpus_fingerprint",),
            ).fetchone()
            assert_fresh_index(
                self.repo_root,
                fingerprint_row["value"] if fingerprint_row is not None else None,
                self.index_path,
            )
            rows = connection.execute(
                "SELECT id, doc_type, path, title, description, priority, body FROM documents ORDER BY path ASC"
            ).fetchall()
            tags_map: dict[str, list[str]] = defaultdict(list)
            for row in connection.execute(
                "SELECT document_id, tag FROM document_tags ORDER BY document_id ASC, tag ASC"
            ).fetchall():
                tags_map[row["document_id"]].append(row["tag"])

            keywords_map: dict[str, list[str]] = defaultdict(list)
            for row in connection.execute(
                "SELECT document_id, keyword FROM document_keywords ORDER BY document_id ASC, keyword ASC"
            ).fetchall():
                keywords_map[row["document_id"]].append(row["keyword"])
        finally:
            connection.close()

        records: dict[str, dict[str, Any]] = {}
        skill_scores: list[ScoredDocument] = []
        wiki_scores: list[ScoredDocument] = []

        for row in rows:
            metadata = {
                "id": row["id"],
                "doc_type": row["doc_type"],
                "path": row["path"],
                "title": row["title"],
                "description": row["description"],
                "priority": row["priority"],
                "body": row["body"],
                "tags": tags_map.get(row["id"], []),
                "activation_keywords": keywords_map.get(row["id"], []),
                "sections": [],
            }
            records[row["id"]] = metadata
            if row["doc_type"] == "skill":
                score, reason = score_skill_metadata(metadata, request, task_type)
                skill_scores.append(
                    ScoredDocument(
                        document_id=row["id"],
                        path=self.repo_root / row["path"],
                        doc_type="skill",
                        title=row["title"],
                        score=score,
                        reason=reason,
                        priority=int(row["priority"]),
                    )
                )
            else:
                score, reason = score_wiki_metadata(metadata, request)
                wiki_scores.append(
                    ScoredDocument(
                        document_id=row["id"],
                        path=self.repo_root / row["path"],
                        doc_type="wiki",
                        title=row["title"],
                        score=float(score),
                        reason=reason,
                    )
                )

        ranked_skills = sort_scored_documents(skill_scores)
        ranked_wiki = sort_scored_documents(wiki_scores)
        selected_skill_meta = [records[item.document_id] for item in ranked_skills if item.score > 0][:top_k]
        selected_wiki_meta = [records[item.document_id] for item in ranked_wiki if item.score > 0][:top_k]

        debug = {
            "index_path": self.index_path.as_posix(),
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

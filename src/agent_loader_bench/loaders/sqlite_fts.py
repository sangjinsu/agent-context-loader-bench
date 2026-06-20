from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any

from .base import (
    ContextLoader,
    CorpusDocument,
    LoadedContext,
    ScoredDocument,
    SkillDocument,
    assert_fresh_index,
    corpus_fingerprint,
    iter_skill_documents,
    iter_wiki_documents,
    load_document_by_metadata,
    metadata_from_document,
    normalize_terms,
    score_skill_metadata,
    score_wiki_metadata,
    select_relevant_sections,
    sort_scored_documents,
    wiki_document_id,
)


def sqlite_fts5_available() -> bool:
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE VIRTUAL TABLE fts_probe USING fts5(content)")
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        connection.close()


def build_sqlite_fts_index(repo_root: Path, index_path: Path) -> Path:
    if not sqlite_fts5_available():
        raise RuntimeError("SQLite FTS5 is unavailable in this Python sqlite3 build")

    root = Path(repo_root)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(index_path)
    try:
        cursor = connection.cursor()
        cursor.executescript(
            """
            DROP TABLE IF EXISTS documents;
            DROP TABLE IF EXISTS document_fts;
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
                body TEXT NOT NULL,
                tags TEXT NOT NULL,
                activation_keywords TEXT NOT NULL,
                sections TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE document_fts USING fts5(
                document_id UNINDEXED,
                doc_type UNINDEXED,
                title,
                description,
                body,
                sections
            );
            """
        )
        cursor.execute(
            "INSERT INTO index_metadata (key, value) VALUES (?, ?)",
            ("corpus_fingerprint", corpus_fingerprint(root)),
        )

        documents = list(iter_skill_documents(root)) + list(iter_wiki_documents(root))
        for document in documents:
            metadata = metadata_from_document(root, document)
            cursor.execute(
                """
                INSERT INTO documents (id, doc_type, path, title, description, priority, body, tags, activation_keywords, sections)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata["id"],
                    metadata["doc_type"],
                    metadata["path"],
                    metadata["title"],
                    metadata["description"],
                    int(metadata.get("priority", 0)),
                    metadata["body"],
                    "\n".join(metadata.get("tags", [])),
                    "\n".join(metadata.get("activation_keywords", [])),
                    "\n".join(metadata.get("sections", [])),
                ),
            )
            cursor.execute(
                """
                INSERT INTO document_fts (document_id, doc_type, title, description, body, sections)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata["id"],
                    metadata["doc_type"],
                    metadata["title"],
                    metadata["description"],
                    metadata["body"],
                    "\n".join(metadata.get("sections", [])),
                ),
            )

        connection.commit()
    finally:
        connection.close()

    return index_path


class SQLiteFTSContextLoader(ContextLoader):
    loader_name = "sqlite_fts"

    def __init__(self, repo_root: Path, index_path: Path | None = None) -> None:
        super().__init__(repo_root)
        self.index_path = index_path or (self.repo_root / ".agentdb" / "fts.sqlite")

    def load(self, request: str, task_type: str | None = None, top_k: int = 3) -> LoadedContext:
        selected_skills, selected_wiki, debug = self._select_documents(request, task_type=task_type, top_k=top_k)
        return self.build_loaded_context(
            selected_skills=selected_skills,
            selected_wiki=selected_wiki,
            request=request,
            debug=debug,
        )

    def _select_documents(
        self,
        request: str,
        *,
        task_type: str | None,
        top_k: int,
    ) -> tuple[list[SkillDocument], list[CorpusDocument], dict[str, Any]]:
        if not self.index_path.exists():
            raise FileNotFoundError(f"Missing SQLite FTS index: {self.index_path}")
        if not sqlite_fts5_available():
            raise RuntimeError("SQLite FTS5 is unavailable in this Python sqlite3 build")

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
            query = _build_fts_query(request)
            if not query:
                return [], [], {"index_path": self.index_path.as_posix(), "selected": [], "selection": []}

            rows = connection.execute(
                """
                SELECT d.id, d.doc_type, d.path, d.title, d.description, d.priority, d.body, d.tags,
                       d.activation_keywords, d.sections, bm25(document_fts) AS rank
                FROM document_fts
                JOIN documents AS d ON d.id = document_fts.document_id
                WHERE document_fts MATCH ?
                ORDER BY rank ASC, d.path ASC
                """,
                (query,),
            ).fetchall()
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
                "tags": [item for item in row["tags"].splitlines() if item],
                "activation_keywords": [item for item in row["activation_keywords"].splitlines() if item],
                "sections": [item for item in row["sections"].splitlines() if item],
            }
            records[row["id"]] = metadata
            fts_score = 1.0 / (1.0 + max(float(row["rank"]), 0.0))
            if row["doc_type"] == "skill":
                metadata_score, reason = score_skill_metadata(metadata, request, task_type)
                skill_scores.append(
                    ScoredDocument(
                        document_id=row["id"],
                        path=self.repo_root / row["path"],
                        doc_type="skill",
                        title=row["title"],
                        score=metadata_score + fts_score,
                        reason=f"fts={query}; {reason}".strip(),
                        priority=int(row["priority"]),
                    )
                )
            else:
                metadata_score, reason = score_wiki_metadata(metadata, request)
                wiki_scores.append(
                    ScoredDocument(
                        document_id=row["id"],
                        path=self.repo_root / row["path"],
                        doc_type="wiki",
                        title=row["title"],
                        score=metadata_score + fts_score,
                        reason=f"fts={query}; {reason}".strip(),
                    )
                )

        ranked_skills = sort_scored_documents(skill_scores)
        ranked_wiki = sort_scored_documents(wiki_scores)
        selected_skill_meta = [
            records[item.document_id] for item in _filter_ranked_documents(ranked_skills, top_k=top_k)
        ]
        selected_wiki_meta = [records[item.document_id] for item in _filter_ranked_documents(ranked_wiki, top_k=top_k)]

        debug = {
            "index_path": self.index_path.as_posix(),
            "fts_query": query,
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
        return (
            [load_document_by_metadata(self.repo_root, metadata) for metadata in selected_skill_meta],
            [load_document_by_metadata(self.repo_root, metadata) for metadata in selected_wiki_meta],
            debug,
        )


class SQLiteFTSSectionContextLoader(SQLiteFTSContextLoader):
    loader_name = "sqlite_fts_section"

    def load(self, request: str, task_type: str | None = None, top_k: int = 3) -> LoadedContext:
        selected_skills, selected_wiki, debug = self._select_documents(request, task_type=task_type, top_k=top_k)
        skill_sections = {document.id: select_relevant_sections(document, request) for document in selected_skills}
        wiki_sections: dict[str, list[Any]] = {}
        for document in selected_wiki:
            wiki_sections[wiki_document_id(self.repo_root, document.path)] = select_relevant_sections(
                document,
                request,
            )

        filtered_skills = [document for document in selected_skills if skill_sections.get(document.id)]
        filtered_wiki = [
            document for document in selected_wiki if wiki_sections.get(wiki_document_id(self.repo_root, document.path))
        ]
        filtered_skill_sections = {document.id: skill_sections[document.id] for document in filtered_skills}
        filtered_wiki_sections = {
            wiki_document_id(self.repo_root, document.path): wiki_sections[
                wiki_document_id(self.repo_root, document.path)
            ]
            for document in filtered_wiki
        }
        filtered_out = [document.id for document in selected_skills if not skill_sections.get(document.id)]
        filtered_out.extend(
            wiki_document_id(self.repo_root, document.path)
            for document in selected_wiki
            if not wiki_sections.get(wiki_document_id(self.repo_root, document.path))
        )
        debug = dict(debug)
        debug["selected_sections_by_document"] = {
            document_id: [section.title for section in sections]
            for document_id, sections in filtered_skill_sections.items()
        }
        debug["selected_sections_by_document"].update(
            {
                document_id: [section.title for section in sections]
                for document_id, sections in filtered_wiki_sections.items()
            }
        )
        debug["filtered_out_documents"] = filtered_out

        return self.build_loaded_context(
            selected_skills=filtered_skills,
            selected_wiki=filtered_wiki,
            request=request,
            debug=debug,
            selected_skill_sections=filtered_skill_sections,
            selected_wiki_sections=filtered_wiki_sections,
        )


def _build_fts_query(request: str) -> str:
    terms = []
    seen: set[str] = set()
    for term in normalize_terms(request):
        if term not in seen:
            seen.add(term)
            terms.append(f'"{term}"')
    return " OR ".join(terms)


def _filter_ranked_documents(documents: list[ScoredDocument], top_k: int) -> list[ScoredDocument]:
    if not documents:
        return []

    top_score = documents[0].score
    threshold = max(1.0, top_score * 0.5)
    return [item for item in documents if item.score >= threshold][:top_k]

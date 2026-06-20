# Markdown Source of Truth Decision

The benchmark keeps Markdown as the canonical instruction format.

## Decision

Project instructions live in `AGENTS.md`, reusable skills live in `skills/**/SKILL.md`, and accumulated knowledge lives in `wiki/**/*.md`. SQLite, JSON, and vector indexes are derived runtime artifacts.

## Rationale

Human-readable Markdown keeps instructions reviewable, diffable, and easy to audit. Indexes are useful for retrieval experiments, but database IDs alone are not sufficient context for an LLM agent.

## Consequences

Index builders must load from Markdown sources. Loader output must include actual instruction text and source identifiers. If generated artifacts are missing or stale, they should be rebuilt explicitly rather than treated as canonical data.

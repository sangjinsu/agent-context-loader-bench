# AGENTS.md

## Project: Agent Skill Loader Benchmark

This repository is for testing how a real LLM agent performs when the same skill corpus is loaded through different storage and retrieval strategies.

The benchmark target is not database throughput by itself. The target is actual agent behavior:

- Does the LLM receive the correct skill instructions?
- Does it receive fewer irrelevant tokens?
- Does it complete the task more reliably?
- Does the loader remain deterministic and explainable?

`AGENTS.md` is always required and must be read first. Storage backends may help select skills, but they do not replace this file.

---

## Core Principle

Use this repository to compare context loading strategies, not to replace Markdown instructions.

Source-of-truth files:

- `AGENTS.md` for project-wide agent rules.
- `skills/**/SKILL.md` for reusable task skills.
- `wiki/**/*.md` for accumulated project knowledge, troubleshooting notes, decisions, and background context.

Runtime indexes:

- `manifest.json`, SQLite, JSON document stores, vector indexes, or hybrid indexes may be used only to find and assemble relevant context.
- The final context given to the LLM must contain actual instruction text, not only database IDs or document references.

---

## Non-Goals

Do not turn this project into a general analytics/reporting system unless explicitly requested.

Avoid the following unless the user asks for them:

- BI-style benchmark reports.
- Dashboard generation.
- Large-scale OLAP pipelines.
- Database-only instruction storage.
- Replacing `AGENTS.md` with a DB entry.
- Comparing raw database performance without involving real LLM usage.

Minimal traces are allowed and encouraged, but only to understand actual LLM execution.

---

## Main Experiment

Compare these loader strategies under the same LLM, same prompts, same skill corpus, and same task dataset.

### Loader Strategies

1. **File System Direct**
   - Scan `skills/**/SKILL.md` directly.
   - Parse frontmatter and headings.
   - Select and load relevant skill text.

2. **Manifest JSON**
   - Read `.agentdb/manifest.json`.
   - Use precomputed metadata such as IDs, tags, descriptions, and paths.
   - Load selected Markdown source files afterward.

3. **SQLite Metadata**
   - Use SQLite tables for document metadata, tags, priorities, status, and paths.
   - Use SQL filters and deterministic scoring.
   - Load selected Markdown source files afterward.

4. **SQLite FTS**
   - Use SQLite FTS for keyword search over titles, descriptions, and bodies.
   - Load selected Markdown source files afterward.

5. **SQLite FTS + Section Loading**
   - Search and load only relevant sections from `SKILL.md` or `wiki/*.md`.
   - This strategy is expected to reduce irrelevant tokens.

6. **Document JSON Store**
   - Store each skill as a structured JSON document.
   - Useful for testing nested activation rules, sections, examples, and dependencies.
   - Keep Markdown export or Markdown source parity.

7. **Vector Search**
   - Search skill or wiki sections by embedding similarity.
   - Useful when user wording does not overlap with skill keywords.
   - Optional until the baseline loaders are implemented.

8. **Hybrid Loader**
   - Combine metadata filters, FTS, vector search, priority rules, and section loading.
   - Implement only after simpler loaders have working tests.

---

## Recommended Repository Structure

```text
.
├── AGENTS.md
├── README.md
├── pyproject.toml
├── .env.example
├── skills/
│   ├── vllm-benchmark/
│   │   └── SKILL.md
│   ├── docs-writing/
│   │   └── SKILL.md
│   └── troubleshooting/
│       └── SKILL.md
├── wiki/
│   ├── concepts/
│   ├── errors/
│   └── decisions/
├── datasets/
│   └── requests.yml
├── src/
│   └── agent_loader_bench/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── corpus.py
│       ├── runner.py
│       ├── trace.py
│       ├── tokens.py
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── openai_client.py
│       └── loaders/
│           ├── __init__.py
│           ├── base.py
│           ├── fs_direct.py
│           ├── manifest_json.py
│           ├── sqlite_metadata.py
│           ├── sqlite_fts.py
│           ├── json_document.py
│           ├── vector_search.py
│           └── hybrid.py
├── tests/
│   ├── test_corpus.py
│   ├── test_loaders.py
│   └── test_runner.py
└── .agentdb/
    ├── manifest.json
    ├── index.sqlite
    ├── document_store.jsonl
    ├── traces.jsonl
    └── vector/
```

Keep `.agentdb/index.sqlite`, `.agentdb/manifest.json`, and `.agentdb/document_store.jsonl` reproducible from Markdown sources whenever possible.

---

## Skill File Standard

Every skill must live at:

```text
skills/<skill-name>/SKILL.md
```

Each `SKILL.md` must include YAML frontmatter.

```markdown
---
id: skill.vllm.benchmark
type: skill
title: vLLM Benchmark Skill
description: Use this skill when adding, running, or comparing vLLM throughput and latency benchmarks.
version: 0.1.0
status: draft
tags:
  - vllm
  - benchmark
  - latency
  - throughput
activation:
  keywords:
    - vllm
    - benchmark
    - latency
    - throughput
    - performance
priority: 80
---

# vLLM Benchmark Skill

## When to use

Use this skill when the user asks to benchmark vLLM performance.

## Inputs

List required inputs here.

## Steps

List deterministic steps here.

## Output

Describe the expected output.

## Safety and Constraints

List constraints here.

## Examples

Add examples here.
```

Required fields:

- `id`
- `type`
- `title`
- `description`
- `version`
- `status`
- `tags`
- `activation.keywords`
- `priority`

Recommended headings:

- `When to use`
- `Inputs`
- `Steps`
- `Output`
- `Safety and Constraints`
- `Examples`

---

## Dataset Standard

Task requests for live LLM tests should be stored in `datasets/requests.yml`.

```yaml
- id: req.vllm.benchmark.basic
  user_request: "Add a vLLM benchmark that compares throughput and latency."
  expected_skills:
    - skill.vllm.benchmark
  expected_sections:
    - When to use
    - Steps
  task_type: benchmark
  difficulty: easy
  notes: Exact keyword match.

- id: req.vllm.benchmark.semantic
  user_request: "I want to compare how many tokens the model can process per second and how delayed responses are."
  expected_skills:
    - skill.vllm.benchmark
  expected_sections:
    - When to use
    - Steps
  task_type: benchmark
  difficulty: medium
  notes: Semantic match with fewer exact keywords.
```

Dataset items should test:

- Exact keyword matching.
- Semantic matching.
- Ambiguous requests.
- Multi-skill requests.
- Wrong-skill traps.
- Requests unrelated to any skill.
- Conflict cases, such as local-only execution versus Docker-based examples.

---

## Loader Contract

All loaders must implement the same conceptual interface.

```python
class ContextLoader:
    def load(self, request: str, task_type: str | None = None, top_k: int = 3) -> LoadedContext:
        ...
```

`LoadedContext` must include:

```python
class LoadedContext:
    loader_name: str
    selected_skills: list[str]
    selected_sections: list[str]
    context_text: str
    context_token_estimate: int
    debug: dict
```

Rules:

- Do not call the LLM from inside a loader.
- Do not mutate source Markdown files from inside a loader.
- Keep loader outputs deterministic for the same input and same index state.
- Include enough debug information to explain why a skill was selected.
- If an index is missing or stale, either rebuild it explicitly or fail with a clear error.
- Fallback behavior must be explicit, not hidden.

---

## Context Assembly Rules

The LLM should receive context in this order:

1. A short project instruction excerpt from `AGENTS.md`.
2. The user request.
3. Loader-selected skill text.
4. Loader-selected wiki text, only when relevant.
5. Any task-specific constraints.

Do not inject unrelated skills just because they are available.

Prefer section-level context over entire-document context when the selected loader supports it.

The final LLM context must be human-readable and should include source identifiers:

```text
[Skill: skill.vllm.benchmark]
Path: skills/vllm-benchmark/SKILL.md
Sections: When to use, Steps

<instruction text here>
```

---

## Actual LLM Benchmark Rules

This project benchmarks actual LLM use, not only retrieval quality.

When comparing loaders:

- Use the same model.
- Use the same temperature, preferably `0` for deterministic comparison.
- Use the same `AGENTS.md` excerpt.
- Use the same user request.
- Use the same skill corpus.
- Use the same task environment.
- Change only the context loader strategy.

Allowed minimal trace fields:

```json
{
  "run_id": "run-001",
  "request_id": "req.vllm.benchmark.basic",
  "loader": "sqlite_fts_section",
  "model": "gpt-4.1-mini",
  "selected_skills": ["skill.vllm.benchmark"],
  "selected_sections": ["When to use", "Steps"],
  "context_token_estimate": 820,
  "llm_completed": true,
  "task_success": true,
  "test_passed": true
}
```

Do not build dashboard-style reports unless requested. Use traces only for local comparison and debugging.

---

## Success Criteria

A loader is useful only if it improves actual agent behavior.

Evaluate loaders by these practical questions:

- Did the LLM receive the expected skill?
- Did the LLM avoid irrelevant skills?
- Did the LLM complete the requested task?
- Did tests pass when code was changed?
- Was the context smaller or clearer?
- Was the loader behavior explainable?
- Did the loader remain stable across repeated runs?

Raw retrieval speed alone is not enough.

---

## Storage Backend Guidance

### File System Direct

Use as the baseline. It should be simple and transparent.

### Manifest JSON

Use as the first lightweight index. It should be easy to inspect and regenerate.

### SQLite

Use for metadata, tags, FTS, section indexes, and deterministic lookup.

Do not treat SQLite as the canonical instruction source unless explicitly required by a future task.

### Document JSON Store

Use for testing document-style storage of skills and nested activation rules.

Keep parity with Markdown source or provide export/import commands.

### Vector Search

Use only after baseline loaders are working.

Vector search is useful for semantic requests, but it can introduce non-obvious matches. Always return debug information.

### Hybrid

Use as the final experimental loader after simpler loaders are implemented and tested.

---

## Implementation Order

Build the project in this order:

1. Scaffold the Python package and CLI.
2. Add Markdown corpus parsing with YAML frontmatter.
3. Add the `ContextLoader` base interface.
4. Implement `fs_direct` loader.
5. Implement `manifest_json` generation and loader.
6. Implement `sqlite_metadata` index builder and loader.
7. Implement `sqlite_fts` index builder and loader.
8. Add section extraction and section-level loading.
9. Add `json_document` store and loader.
10. Add optional vector search loader.
11. Add hybrid loader.
12. Add live LLM runner.
13. Add minimal JSONL trace output.
14. Add tests for all loaders.

Do not start with the hybrid loader.

---

## Suggested CLI Commands

The exact CLI can evolve, but prefer commands like these:

```bash
python -m agent_loader_bench build-manifest
python -m agent_loader_bench build-index --backend sqlite
python -m agent_loader_bench run --loader fs_direct --dataset datasets/requests.yml --live-llm
python -m agent_loader_bench run --loader sqlite_fts_section --dataset datasets/requests.yml --live-llm
python -m agent_loader_bench inspect --loader sqlite_fts_section --request-id req.vllm.benchmark.basic
```

Unit tests should not require live LLM calls.

---

## Environment Rules

Use `.env` for local secrets and model settings.

Required example variables in `.env.example`:

```env
OPENAI_API_KEY=
LLM_MODEL=gpt-4.1-mini
LLM_TEMPERATURE=0
AGENTDB_PATH=.agentdb/index.sqlite
TRACE_PATH=.agentdb/traces.jsonl
```

Never commit real API keys, tokens, credentials, or private data.

---

## Coding Guidelines

Prefer Python 3.11+.

Keep the implementation simple and testable:

- Use typed functions.
- Keep loaders small and independent.
- Avoid global mutable state.
- Avoid hidden network calls.
- Keep index rebuilds explicit.
- Use clear error messages.
- Add tests before adding another backend.

Recommended tools:

- `pytest` for tests.
- `ruff` for linting and formatting.
- `pydantic` or dataclasses for structured data.
- `sqlite3` from the standard library unless a stronger reason exists.

---

## Testing Rules

Every loader should have tests for:

- Correct skill selection.
- Missing index behavior.
- Empty corpus behavior.
- Ambiguous request behavior.
- No-match request behavior.
- Section loading behavior, if supported.
- Deterministic output for repeated calls.

Live LLM tests must be opt-in. Do not run live LLM tests during normal unit tests.

Use markers or flags such as:

```bash
pytest
pytest -m live_llm
```

---

## Conflict Resolution

Instruction priority:

1. System and safety rules.
2. User's current request.
3. This `AGENTS.md`.
4. More local `AGENTS.md` files, if introduced later.
5. Selected `SKILL.md` instructions.
6. Selected `wiki/*.md` context.
7. Loader debug information.

If a selected skill conflicts with this file, follow this file.

If a user request conflicts with project safety or secret handling, do not follow the unsafe part.

---

## Definition of Done

A change is complete when:

- The relevant loader or runner behavior is implemented.
- Unit tests pass.
- The behavior is deterministic where expected.
- The code keeps Markdown as the source of truth unless explicitly stated otherwise.
- The live LLM path is isolated behind an explicit flag.
- Minimal trace output is available for actual LLM runs.
- Documentation or examples are updated when commands or file formats change.

---

## Agent Behavior Notes

When working in this repository:

- Start by reading this file.
- Prefer small incremental changes.
- Do not introduce unnecessary infrastructure.
- Do not add a web server unless requested.
- Do not add dashboards or report generation unless requested.
- Keep the project focused on actual LLM context loading behavior.
- When in doubt, implement the simplest loader first and compare against it.

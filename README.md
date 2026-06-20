# Agent Context Loader Bench

Agent Context Loader Bench compares how different context loading strategies affect a real LLM agent working from the same instruction corpus.

The benchmark is about agent behavior, not raw database throughput. A useful loader should select the right skill text, avoid irrelevant tokens, keep behavior explainable, and help the agent complete the requested task reliably.

## Source of Truth

Markdown remains the canonical instruction format:

- `AGENTS.md` contains project-wide agent rules and must be read first.
- `skills/**/SKILL.md` contains reusable task skills with YAML frontmatter.
- `wiki/**/*.md` contains project knowledge, troubleshooting notes, decisions, and background context.

Runtime indexes under `.agentdb/` are derived artifacts. They may help select context, but the final LLM context must contain human-readable instruction text from the Markdown sources.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
cp .env.example .env
```

Leave `OPENAI_API_KEY` empty unless you are explicitly running live LLM comparisons.

## CLI Examples

File-system loading does not require an index:

```bash
python -m agent_loader_bench inspect \
  --loader fs_direct \
  --request-id req.vllm.benchmark.exact
```

Build derived indexes before using index-backed loaders:

```bash
python -m agent_loader_bench build-manifest
python -m agent_loader_bench build-index --backend all
```

Inspect or run a dataset with a specific loader:

```bash
python -m agent_loader_bench inspect \
  --loader manifest_json \
  --request-id req.docs.troubleshooting.multi_skill

python -m agent_loader_bench run \
  --loader sqlite_fts_section \
  --dataset datasets/requests.yml
```

## Loader Strategies

The project compares these strategies against the same model, task dataset, and Markdown corpus:

- `fs_direct`: scan `skills/**/SKILL.md` and `wiki/**/*.md` directly.
- `manifest_json`: read `.agentdb/manifest.json`, then load Markdown sources.
- `sqlite_metadata`: use SQLite metadata filters, then load Markdown sources.
- `sqlite_fts`: use SQLite full-text search over titles, descriptions, and bodies.
- `sqlite_fts_section`: use full-text search and assemble only relevant sections.
- `json_document`: use structured JSON documents derived from Markdown.
- `vector_search`: use deterministic vector-like matching for semantic requests.
- `hybrid`: combine simpler strategies after their behavior is tested.

Indexes must be explicitly rebuilt when the Markdown corpus changes.

## Live LLM Opt-In

Normal test and run commands do not call an LLM. Live comparison is opt-in:

```bash
OPENAI_API_KEY=... python -m agent_loader_bench run \
  --loader manifest_json \
  --dataset datasets/requests.yml \
  --live-llm
```

Use the same model, temperature, task dataset, and corpus when comparing loaders. Change only the loader strategy.

## Testing

Run unit tests without live LLM calls:

```bash
python3 -m pytest
```

Run tests marked for the opt-in live LLM path only when that path is intentionally under review:

```bash
python3 -m pytest -m live_llm
```

Sample corpus checks live in `tests/test_project_samples.py`.

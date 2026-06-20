---
id: skill.troubleshooting
type: skill
title: Troubleshooting Skill
description: Use this skill when diagnosing loader failures, missing indexes, test failures, stale generated artifacts, and context assembly problems.
version: 0.1.0
status: draft
tags:
  - troubleshooting
  - debugging
  - errors
  - tests
  - index
activation:
  keywords:
    - troubleshoot
    - troubleshooting
    - debug
    - failure
    - failing
    - error
    - traceback
    - index
    - manifest
    - sqlite
    - local-only
    - docker
priority: 70
---

# Troubleshooting Skill

Use this skill to diagnose context loader benchmark failures with a small, local, reproducible path.

## When to use

Use this skill when a command fails, an index is missing or stale, selected skills are unexpected, section loading returns too much or too little context, or tests fail after corpus changes.

Use it for local-only conflict requests that explicitly reject Docker-based setup or external services.

## Inputs

Collect these inputs before proposing a fix:

- Exact command and working directory.
- Full error output or traceback.
- Loader strategy under test.
- Dataset request ID, if applicable.
- Whether `.agentdb/` indexes were rebuilt after Markdown changes.
- Whether the task is unit-test-only or an explicit live LLM run.

## Steps

1. Reproduce the failure locally with the smallest command that shows it.
2. Identify whether the failing path uses direct Markdown loading or a generated index.
3. If an index-backed loader fails, rebuild the relevant index explicitly.
4. Compare expected skills and sections from `datasets/requests.yml` with the loaded context.
5. Check whether the Markdown source, generated index, or request wording caused the mismatch.
6. Keep the fix local unless the user explicitly asks for an external service setup.
7. Re-run the targeted command, then run the relevant unit tests.

## Output

Return a compact diagnosis that includes:

- Symptom.
- Likely root cause.
- Files or commands involved.
- Minimal fix.
- Verification command and result.

## Safety and Constraints

- Do not introduce Docker setup when a request asks for a local-only reproduction.
- Do not hide fallback behavior; state which loader or index was used.
- Do not mutate Markdown source files from inside a loader.
- Do not treat generated `.agentdb/` artifacts as source files.
- Do not run live LLM calls unless the user explicitly opts in.

## Examples

Example request:

```text
Troubleshoot a missing manifest error when I inspect a request with manifest_json.
```

Expected response shape:

```text
The manifest-backed loader requires .agentdb/manifest.json. Rebuild it with python -m agent_loader_bench build-manifest, then rerun the inspect command. If the selected skill is still wrong, compare the request wording with the skill activation keywords and dataset expected_skills.
```

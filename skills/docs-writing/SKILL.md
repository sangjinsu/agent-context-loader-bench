---
id: skill.docs.writing
type: skill
title: Docs Writing Skill
description: Use this skill when writing or maintaining technical documentation, README content, guides, examples, and task instructions.
version: 0.1.0
status: draft
tags:
  - docs
  - writing
  - readme
  - guide
  - examples
activation:
  keywords:
    - docs
    - documentation
    - readme
    - guide
    - tutorial
    - examples
    - instructions
    - explain
priority: 60
---

# Docs Writing Skill

Use this skill to produce clear technical documentation for the context loader benchmark without expanding the project scope.

## When to use

Use this skill when the user asks to write, improve, reorganize, or clarify README content, guides, sample requests, examples, or public-facing project documentation.

Prefer this skill over benchmark implementation skills when the user explicitly asks for explanation, documentation, or examples only.

## Inputs

Collect these inputs before editing documentation:

- Target reader and expected skill level.
- The command, file format, or workflow being documented.
- Source-of-truth files that the documentation must reflect.
- Verification command the reader can run locally.
- Any explicit exclusions, such as avoiding new infrastructure or generated artifacts.

## Steps

1. Read the current project instructions before writing public documentation.
2. Keep public repository documentation in English.
3. Explain the source-of-truth rule before describing runtime indexes.
4. Provide commands that can run from a fresh checkout.
5. Distinguish normal unit-test paths from opt-in live LLM paths.
6. Avoid adding broader product scope that is not part of context loader comparison.
7. Update examples when file formats, CLI commands, or sample datasets change.

## Output

Return documentation that is concise, task-oriented, and easy to scan. Include:

- Purpose and scope.
- Setup steps.
- Runnable command examples.
- Important constraints.
- Verification commands.

## Safety and Constraints

- Do not include real API keys, tokens, credentials, or private data.
- Do not describe generated indexes as canonical instruction storage.
- Do not add unrelated analytics, deployment, or service operation content.
- Do not leave commands without enough context for a new contributor to run them.

## Examples

Example request:

```text
Update the README so contributors understand how to run loader comparisons.
```

Expected response shape:

```text
Document the benchmark purpose, source-of-truth files, setup, index rebuild commands, inspect/run examples, live LLM opt-in behavior, and test commands.
```

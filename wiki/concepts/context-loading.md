# Context Loading Model

Context loading is the process of selecting the smallest useful instruction set for a user request.

## Source-of-Truth Documents

`AGENTS.md`, `skills/**/SKILL.md`, and `wiki/**/*.md` are the human-readable source documents. Runtime indexes may point to these documents, but they do not replace them.

## Context Assembly Order

The LLM should receive a short `AGENTS.md` excerpt, the user request, selected skill text, relevant wiki text, and task-specific constraints. This order keeps project rules ahead of narrower context.

## Section-Level Loading

Section-level loaders should include only the relevant headings and content from a skill or wiki document. This reduces irrelevant tokens while preserving readable source identifiers.

## Evaluation Questions

Useful loader comparisons ask whether the expected skill was selected, irrelevant skills were avoided, context size was reduced, behavior was explainable, and the task could be completed.

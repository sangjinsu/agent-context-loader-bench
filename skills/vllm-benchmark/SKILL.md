---
id: skill.vllm.benchmark
type: skill
title: vLLM Benchmark Skill
description: Use this skill when adding, running, or comparing vLLM throughput, latency, and performance benchmarks.
version: 0.1.0
status: draft
tags:
  - vllm
  - benchmark
  - latency
  - throughput
  - performance
activation:
  keywords:
    - vllm
    - benchmark
    - latency
    - throughput
    - performance
    - tokens per second
    - requests per second
    - response delay
priority: 80
---

# vLLM Benchmark Skill

Use this skill to prepare benchmark tasks that compare vLLM throughput and latency under controlled conditions.

## When to use

Use this skill when the user asks to add, run, inspect, or compare vLLM benchmark behavior. It also applies when the user describes performance in semantic terms such as tokens per second, request rate, response delay, or end-to-end latency.

Do not use this skill for documentation-only tasks that mention benchmark results but do not require benchmark implementation or execution.

## Inputs

Collect these inputs before changing benchmark behavior:

- Target model name and serving configuration.
- Hardware profile, including GPU type, CPU type, memory, and local runtime constraints.
- Prompt dataset or synthetic prompt fixture.
- Concurrency level, request count, warmup count, and timeout.
- Metrics to compare, especially throughput, first-token latency, total latency, and error rate.
- Loader strategy under test, if the benchmark is part of a context loading comparison.

## Steps

1. Confirm that the task is about benchmark execution or benchmark code, not only explanatory documentation.
2. Keep Markdown instructions as the source of truth and treat generated indexes as rebuildable runtime artifacts.
3. Use the same model, prompts, temperature, and task environment across loader strategies.
4. Separate warmup runs from measured runs.
5. Capture throughput and latency metrics with deterministic field names.
6. Record enough local trace data to explain what context the LLM received.
7. Verify that unit tests do not require live LLM calls or external vLLM services.

## Output

Return a compact benchmark plan or implementation summary that includes:

- Benchmark scope and excluded work.
- Required local setup.
- Metrics collected.
- Commands to run.
- How to compare loader behavior from the resulting traces.

## Safety and Constraints

- Do not require live LLM calls for normal unit tests.
- Do not assume cloud hardware or paid services.
- Do not compare results from different hardware profiles as if they were equivalent.
- Do not store benchmark instructions only in SQLite, JSON, or vector indexes.
- Do not commit secrets, API keys, generated traces, or runtime indexes.

## Examples

Example request:

```text
Add a vLLM benchmark that compares throughput and latency for two loader strategies.
```

Expected response shape:

```text
Use the same prompt fixture, model, temperature, and request count for both loaders. Record selected skills, selected sections, context token estimate, throughput, p50 latency, p95 latency, and task success. Keep live LLM execution behind an explicit flag.
```

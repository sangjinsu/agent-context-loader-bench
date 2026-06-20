# Missing Index Troubleshooting

Index-backed loaders fail clearly when a required `.agentdb/` artifact is missing.

## Symptoms

Common symptoms include errors mentioning `manifest.json`, `index.sqlite`, `fts.sqlite`, `document_store.jsonl`, or vector index files. The failure usually happens when using an index-backed loader before rebuilding derived artifacts.

## Local Fix

Use the direct file-system loader when no index should be required:

```bash
python -m agent_loader_bench inspect --loader fs_direct --request-id req.vllm.benchmark.exact
```

Rebuild derived artifacts before using index-backed loaders:

```bash
python -m agent_loader_bench build-manifest
python -m agent_loader_bench build-index --backend all
```

## Local-Only Constraint

Do not add Docker or external services just to reproduce missing index behavior. The expected fix is to rebuild the local `.agentdb/` artifact or choose a loader that does not require it.

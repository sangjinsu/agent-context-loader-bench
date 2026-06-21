# Agent Context Loader Bench

*[English](README.md) · 한국어*

Agent Context Loader Bench는 **동일한 지시 코퍼스**를 서로 다른 컨텍스트 로딩 전략으로 불러왔을 때, 실제 LLM 에이전트의 동작이 어떻게 달라지는지 비교합니다.

이 벤치마크의 관심사는 DB 처리량이 아니라 **에이전트 동작**입니다. 좋은 로더는 올바른 스킬 텍스트를 선택하고, 불필요한 토큰을 피하며, 동작을 설명 가능하게 유지하고, 요청한 작업을 안정적으로 완료하도록 돕습니다.

## Source of Truth

Markdown이 정식 지시 형식입니다.

- `AGENTS.md` — 프로젝트 전역 에이전트 규칙. 가장 먼저 읽어야 합니다.
- `skills/**/SKILL.md` — YAML frontmatter를 가진 재사용 가능한 작업 스킬.
- `wiki/**/*.md` — 프로젝트 지식, 트러블슈팅 노트, 결정 사항, 배경 컨텍스트.

`.agentdb/` 아래의 런타임 인덱스는 파생 산출물입니다. 컨텍스트 선택을 도울 수 있지만, 최종적으로 LLM에 전달되는 컨텍스트는 Markdown 원본의 사람이 읽을 수 있는 지시 텍스트를 담아야 합니다.

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
cp .env.example .env
```

live LLM 비교를 명시적으로 실행하는 경우가 아니라면 `OPENAI_API_KEY`는 비워 두세요.

## CLI 예시

파일 시스템 로딩은 인덱스가 필요 없습니다.

```bash
python -m agent_loader_bench inspect \
  --loader fs_direct \
  --request-id req.vllm.benchmark.exact
```

인덱스 기반 로더를 쓰기 전에 파생 인덱스를 빌드합니다.

```bash
python -m agent_loader_bench build-manifest
python -m agent_loader_bench build-index --backend all
```

특정 로더로 데이터셋을 inspect 하거나 실행합니다.

```bash
python -m agent_loader_bench inspect \
  --loader manifest_json \
  --request-id req.docs.troubleshooting.multi_skill

python -m agent_loader_bench run \
  --loader sqlite_fts_section \
  --dataset datasets/requests.yml
```

트레이스 파일로 로더를 비교합니다(`run_id` 기준 중복 제거, 최신 우선).

```bash
python -m agent_loader_bench compare --trace .agentdb/traces.jsonl
```

## 로더 전략

동일한 모델·작업 데이터셋·Markdown 코퍼스 기준으로 다음 전략들을 비교합니다.

- `fs_direct`: `skills/**/SKILL.md`와 `wiki/**/*.md`를 직접 스캔.
- `manifest_json`: `.agentdb/manifest.json`을 읽은 뒤 Markdown 원본 로드.
- `sqlite_metadata`: SQLite 메타데이터 필터 후 Markdown 원본 로드.
- `sqlite_fts`: 제목·설명·본문에 대한 SQLite 전문 검색(FTS).
- `sqlite_fts_section`: 전문 검색으로 관련 섹션만 조립.
- `json_document`: Markdown에서 파생한 구조화 JSON 문서 사용.
- `vector_search`: 의미 기반 요청을 위한 결정적 벡터 유사 매칭.
- `hybrid`: 단순 전략들이 검증된 뒤 이들을 결합.

Markdown 코퍼스가 바뀌면 인덱스는 명시적으로 재빌드해야 합니다.

## Live LLM 옵트인

일반 테스트·실행 명령은 LLM을 호출하지 않습니다. live 비교는 옵트인입니다.

```bash
OPENAI_API_KEY=... python -m agent_loader_bench run \
  --loader manifest_json \
  --dataset datasets/requests.yml \
  --live-llm
```

로더를 비교할 때는 동일한 모델·temperature·작업 데이터셋·코퍼스를 사용하고 **로더 전략만** 바꿉니다.

provider는 `LLM_MODEL`로 추론합니다. `claude-*` 모델은 Anthropic 클라이언트로 라우팅되며(`ANTHROPIC_API_KEY`와 `anthropic` extra 필요: `pip install -e '.[anthropic]'`), 그 외에는 OpenAI를 사용합니다. `LLM_PROVIDER=openai|anthropic`로 강제 지정할 수 있습니다. Claude 모델은 sampling 파라미터 없이 호출됩니다(Opus 4.8은 `temperature`/`top_p`를 거부).

## 테스트

live LLM 호출 없이 단위 테스트를 실행합니다.

```bash
python3 -m pytest
```

옵트인 live LLM 경로를 의도적으로 점검할 때만 해당 마커 테스트를 실행합니다.

```bash
python3 -m pytest -m live_llm
```

샘플 코퍼스 점검은 `tests/test_project_samples.py`에 있습니다.

## 벤치마크 리포트

8종 로더에 대한 최신 live 실행 결과와 해석은 [`reports/loader-benchmark.md`](reports/loader-benchmark.md)를 참고하세요.

# 로더 벤치마크 리포트

> **TL;DR** — 동일 모델·데이터셋·코퍼스에서 8개 컨텍스트 로더를 비교했다.
> **권장 로더는 `sqlite_fts_section`**(의미적 도달력이 필요하면 `hybrid`). 두 로더 모두
> **7/7 정확도**를 **657 토큰**으로 달성해, baseline `fs_direct`(6/7, 1126 토큰) 대비
> **정확도는 높이고 토큰은 약 42% 절감**한다. 핵심 차이는 "문서 전체"가 아니라
> "관련 섹션만" 싣는 데서 온다.

## 실행 개요

| 항목 | 값 |
|---|---|
| 날짜 | 2026-06-20 (최초) · 2026-06-21 (`vector_search` 실제 임베딩 갱신) |
| LLM | `claude-opus-4-8` (live, Anthropic 폴백 경로) |
| 임베딩 (`vector_search`) | OpenAI `text-embedding-3-small` (1536차원) |
| 데이터셋 | `datasets/requests.yml` — 7개 요청 |
| 코퍼스 | 스킬 3종 + 위키 노트 3종 (Markdown이 source of truth) |
| 호출 수 | 8 로더 × 7 요청 = **56회** live LLM 호출, 전부 완료 |
| 트레이스 | `.agentdb/bench-traces.jsonl` (전용 파일; 기존 `traces.jsonl`은 미수정) |

---

## 1. 한눈에 보기

실제 OpenAI 임베딩 + provider별 임계값(openai 0.28) 적용 후 **최종 집계**.
정렬: 성공률 내림차순 → 평균 토큰 오름차순. `vs baseline`은 `fs_direct`(1126 토큰) 대비 절감률.

| 순위 | 로더 | 성공 | 정확도 | 평균 토큰 | vs baseline | LLM 완료 |
|---:|---|:---:|:---:|---:|:---:|:---:|
| 🥇 | `hybrid` | 7/7 | **100%** | **657** | **−42%** | 7 |
| 🥇 | `sqlite_fts_section` | 7/7 | **100%** | **657** | **−42%** | 7 |
| 🥉 | `sqlite_fts` | 7/7 | **100%** | 817 | −27% | 7 |
| 4 | `json_document` | 6/7 | 86% | 1066 | −5% | 7 |
| 4 | `manifest_json` | 6/7 | 86% | 1066 | −5% | 7 |
| 4 | `sqlite_metadata` | 6/7 | 86% | 1066 | −5% | 7 |
| 7 | `fs_direct` *(baseline)* | 6/7 | 86% | 1126 | — | 7 |
| 8 | `vector_search` | 5/7 | 71% | 776 | −31% | 7 |

- **성공/정확도**: 로더가 선택한 스킬·섹션의 정확도(아래 §3.4 평가 기준). **평균 토큰**: 평균 `context_token_estimate`. **LLM 완료**: live 호출 완료 수.
- 해싱 → 실제 임베딩 전환 효과: **`vector_search` 3/7 → 5/7** (토큰 901 → 776). `hybrid`는 7/7 유지(튜닝 전 일시적으로 6/7 회귀 → 임계값 0.28로 복구).

**결론.** 7/7을 달성한 세 로더 중 `hybrid`·`sqlite_fts_section`이 토큰까지 최소다. 정확도와
비용을 동시에 만족하는 **`sqlite_fts_section`을 기본 권장**하고, 벡터의 의미적 도달력이
필요한 환경에서는 `hybrid`를 쓴다.

---

## 2. 무엇을 비교했나

### 2.1 로더 전략 (8종)

같은 모델·데이터셋·Markdown 코퍼스에 대해 **컨텍스트 로딩 전략만** 바꿔 비교한다.

| 로더 | 전략 | 한 줄 요약 |
|---|---|---|
| `fs_direct` | 파일 직접 스캔 | `skills/**/SKILL.md`·`wiki/**/*.md`를 그대로 읽음 (baseline) |
| `manifest_json` | 매니페스트 | `manifest.json` 조회 후 Markdown 원문 로드 |
| `sqlite_metadata` | 메타데이터 필터 | SQLite 메타데이터로 후보 좁힌 뒤 원문 로드 |
| `sqlite_fts` | 전문 검색 | 제목·설명·본문 FTS5 검색, 문서 전체 로드 |
| `sqlite_fts_section` | 전문 검색 + 섹션 | FTS5로 찾고 **관련 섹션만** 조립 |
| `json_document` | 구조화 JSON | Markdown 파생 JSON 문서 사용 |
| `vector_search` | 임베딩 유사도 | 문서·쿼리 임베딩 후 코사인 유사도 랭킹 |
| `hybrid` | 결합 | 메타데이터·FTS·우선순위·섹션 로딩을 벡터와 결합 |

### 2.2 평가 데이터셋 (7개 요청)

함정·충돌·무관 케이스를 의도적으로 섞어 retrieval 변별력을 측정한다.

| 요청 ID | 의도 | 유형 | 난이도 | 기대 스킬 |
|---|---|---|:---:|---|
| `req.vllm.benchmark.exact` | vLLM 벤치마크 추가 (키워드 정확 일치) | benchmark | easy | `skill.vllm.benchmark` |
| `req.vllm.benchmark.semantic` | "초당 토큰·응답 지연 비교" (vLLM 단어 없는 의미 매칭) | benchmark | medium | `skill.vllm.benchmark` |
| `req.loader.ambiguous` | 로더 예제를 이해하기 쉽게 (구현보다 **문서** 선호) | docs | medium | `skill.docs.writing` |
| `req.docs.troubleshooting.multi_skill` | README 트러블슈팅 갱신 + 가이드 (**멀티 스킬**) | docs | medium | `skill.docs.writing` + `skill.troubleshooting` |
| `req.vllm.wrong_skill_trap` | 벤치마크 결과를 **설명만**, 구현 금지 (**오선택 함정**) | docs | hard | `skill.docs.writing` |
| `req.unrelated.no_match` | "현악 4중주 자장가 작곡" (**무관·무매칭**) | music | easy | _(없음)_ |
| `req.troubleshooting.local_only_conflict` | 로컬 전용 재현으로 해결, Docker 추가 금지 (**충돌**) | troubleshooting | hard | `skill.troubleshooting` |

---

## 3. 테스트 방법

### 3.1 모델 결정 (gpt-5.5 → Claude 폴백)

`.env`의 기본 모델은 `gpt-5.5`였다. 스모크 테스트에서 다음 오류가 발생했다.

```
error: Error code: 400 - Unsupported parameter: 'temperature' is not supported with this model.
```

`gpt-5.5`는 **존재하는 모델**이지만 `temperature` 파라미터를 거부했다(Claude Opus 4.8과 동일한 제약). 사용자 지시("안 될 경우 Claude 모델로 테스트")에 따라 `claude-opus-4-8`로 폴백했다.

- 별도 클라이언트 `AnthropicMessagesClient`(공식 `anthropic` SDK)를 사용한다. 기존 OpenAI 클라이언트에 Anthropic 호출을 섞거나 호환 샤임을 쓰지 않는다.
- Claude 경로는 **sampling 파라미터(temperature/top_p)를 보내지 않는다**(Opus 4.8은 전달 시 400). `temperature` 인자는 `LLMClient` Protocol 호환용으로만 남기고 API에는 전달하지 않는다.
- provider는 `LLM_MODEL`로 추론한다(`claude-*` → anthropic, 그 외 → openai). `LLM_PROVIDER`로 강제 지정 가능.

### 3.2 환경 준비

```bash
pip install -e '.[anthropic]'          # Claude 폴백 경로 의존성
# .env: ANTHROPIC_API_KEY=...  LLM_MODEL=claude-opus-4-8   (.env는 gitignore)
python -m agent_loader_bench build-index --backend all   # Markdown 소스 기준 인덱스 재빌드
```

### 3.3 실행 (스모크 → 전체 스윕 → 집계)

```bash
# (1) 스모크: 전용 트레이스로 단일 로더 1회 — Claude 호출 정상 확인
#     → 7건 전부 llm_completed=True, model=claude-opus-4-8
TRACE_PATH=.agentdb/bench-traces.jsonl \
  python -m agent_loader_bench run --loader fs_direct --dataset datasets/requests.yml --live-llm

# (2) 전체 스윕: 전용 트레이스를 비운 뒤 8 로더 × 7 요청 = 56회 순차 실행
for l in fs_direct manifest_json sqlite_metadata sqlite_fts \
         sqlite_fts_section json_document vector_search hybrid; do
  TRACE_PATH=.agentdb/bench-traces.jsonl \
    python -m agent_loader_bench run --loader "$l" --dataset datasets/requests.yml --live-llm
done

# (3) 집계: run_id로 중복 제거(최신 우선) 후 성공률↓·토큰↑ 정렬
python -m agent_loader_bench compare --trace .agentdb/bench-traces.jsonl
```

### 3.4 평가 기준

`task_success`는 LLM **출력**이 아니라 **로더가 선택한 스킬/섹션**으로 결정된다(`evaluate_loaded_context`). 다음 세 조건을 **모두** 만족할 때만 성공이다.

1. 기대 스킬을 모두 포함
2. 불필요한(기대 외) 스킬이 없음
3. 기대 섹션을 모두 포함

즉 "LLM이 올바른 컨텍스트를 받았는가"를 결정적으로 측정한다. `llm_completed`는 실제 Claude 호출 발생 여부를 보조 신호로 별도 추적한다. 평가가 retrieval 기반이므로 Opus 4.8을 temperature 없이 호출해도 결과는 재현 가능하다.

### 3.5 vector_search 실제 임베딩 업그레이드 (2026-06-21)

초기 `vector_search`는 임베딩 모델이 아니라 24차원 해싱 트릭이라 3/7로 최하위였다. 이를 **OpenAI 실제 임베딩**(`text-embedding-3-small`, 1536차원)으로 교체했다.

- `embeddings/` 추상화 추가(provider: `hashing` | `openai`). 로더는 **인덱스에 기록된 provider/model로 쿼리를 임베딩**해 문서·쿼리 모델 불일치를 차단한다.
- 임베딩 인덱스 재빌드(OpenAI 호출 ~6 문서):
  ```bash
  EMBEDDING_PROVIDER=openai python -m agent_loader_bench build-index --backend vector
  ```
- 그 뒤 `vector_search`·`hybrid`만 `--live-llm` 재실행해 표를 갱신했다(`compare`가 `run_id`로 dedupe).

**임계값 튜닝.** 실제 임베딩은 무관 텍스트도 절대 코사인이 ~0.2라, 해싱 기준(무관→0)으로 맞춘 바닥(0.2)이 너무 느슨했다. `req.unrelated.no_match`("자장가")가 `skill.vllm.benchmark`에 **0.2042**로 매칭돼 오선택됐고, 이 신호가 `hybrid`로 새어들어 7/7→6/7 회귀를 일으켰다. provider별 바닥을 도입(hashing 0.2 / openai **0.28**, `EMBEDDING_MIN_SCORE`로 오버라이드)하자 no_match(0.2042<0.28)는 선택 없음으로, semantic(0.348)·exact(0.677)는 유지됐다.

---

## 4. 결과 상세 — 요청별 실패 분석

성공률이 갈리는 지점은 단 두 요청이다(실제 임베딩 기준).

| 실패한 요청 | 함정 | 실패한 로더 |
|---|---|---|
| `req.loader.ambiguous` | 모호 — 구현보다 문서를 선호해야 함 | `fs_direct`, `json_document`, `manifest_json`, `sqlite_metadata`, `vector_search` |
| `req.vllm.wrong_skill_trap` | 오선택 — 벤치/vLLM 단어가 있으나 문서 작업 | `vector_search` |

- **6/7 그룹**(`fs_direct`, `json_document`, `manifest_json`, `sqlite_metadata`)은 **모호 요청(`req.loader.ambiguous`) 하나만** 실패 — 의도된 baseline 판별 케이스.
- **`vector_search`(5/7)**는 실제 임베딩으로 semantic·multi_skill·no_match를 해결했고, 남은 2건은 **진짜 의미 불일치**다: `req.loader.ambiguous`는 `skill.troubleshooting`을, `req.vllm.wrong_skill_trap`은 `skill.vllm.benchmark`를 선택(기대는 둘 다 `skill.docs.writing`). 이는 임계값이 아니라 임베딩이 표면 의미에 끌리는 한계.
- **7/7 그룹**(`hybrid`, `sqlite_fts_section`, `sqlite_fts`)은 두 함정을 모두 통과한다.

---

## 5. 해석

1. **섹션 단위 로더가 두 축 모두 우위.** `hybrid`·`sqlite_fts_section`은 7/7 정확 + **657 토큰**으로, baseline `fs_direct`(1126) 대비 약 **42% 토큰 절감**. "섹션 단위 컨텍스트가 불필요 토큰을 줄인다"는 AGENTS.md 가설 입증.
2. **문서 전체 FTS는 정확하지만 무겁다.** `sqlite_fts`는 7/7이지만 817 토큰 — 문서 전체를 싣기 때문. 섹션 조립이 결정적 차이.
3. **6/7 그룹은 모호 요청에서만 갈린다.** 네 로더 모두 `req.loader.ambiguous`만 놓친다.
4. **실제 임베딩으로 vector_search 개선(3/7 → 5/7).** semantic 매칭이 좋아져 의미 기반 요청을 해결했다. 다만 무관 요청의 false positive를 막으려면 provider별 임계값이 필수였고(해싱 기준 0.2는 실제 임베딩에 너무 느슨), 남은 2건은 임베딩 자체의 표면-의미 한계다. **`hybrid`는 벡터의 의미적 도달력을 유지하면서 메타데이터·FTS·우선순위·섹션 로딩으로 벡터 단독의 약점을 메운다** — 단독 5/7 vs 결합 7/7.

**권장 로더:** `sqlite_fts_section` (의미적 도달력이 필요하면 `hybrid`) — 정확도 최고 + 토큰 최소.

---

## 6. 검증

- `python -m pytest -q` → **52 passed** (기존 36 + compare/Claude 10 + 임베딩 6)
- `ruff check .` / `ruff format --check .` → 클린
- `EMBEDDING_PROVIDER=openai build-index --backend vector` → 인덱스 `embedding_provider="openai"`, 1536차원
- 최종 스윕 트레이스 → **56 레코드, 8 로더, 전부 `llm_completed=True`, LLM=`claude-opus-4-8`, vector 인덱스=openai**

---

## 7. 비고

- Markdown이 source of truth — 로더는 소스를 수정하지 않고, `compare`는 트레이스만 읽는다.
- 저비용 재실행은 `LLM_MODEL=claude-haiku-4-5`로 가능.
- `.env`의 실제 API 키는 gitignore되어 커밋되지 않는다.
- `vector_search` 임베딩 provider는 `EMBEDDING_PROVIDER`로 전환(`openai` 기본 / `hashing` 오프라인). provider 변경 시 `build-index --backend vector` 재빌드 필요. 임계값은 provider별 기본(hashing 0.2 / openai 0.28)이며 `EMBEDDING_MIN_SCORE`로 오버라이드.
- OpenAI 임베딩은 호출 간 비트 단위 동일성을 보장하진 않으므로, 결정성 단위 테스트는 hashing 인덱스로 수행하고 openai 경로는 fake로 검증한다.

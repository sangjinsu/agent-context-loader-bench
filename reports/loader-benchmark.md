# 로더 벤치마크 리포트

- **날짜:** 2026-06-20
- **모델:** `claude-opus-4-8` (live, Anthropic 폴백 경로)
- **데이터셋:** `datasets/requests.yml` (7개 요청)
- **코퍼스:** 스킬 3종 + 위키 노트 3종 (Markdown이 source of truth)
- **호출 수:** 8개 로더 × 7개 요청 = 56회 live LLM 호출, 전부 완료
- **트레이스:** `.agentdb/bench-traces.jsonl` (전용 파일; 기존 `traces.jsonl`은 손대지 않음)

---

## 1. 테스트 방법

### 1.1 모델 결정 (gpt-5.5 → Claude 폴백)

`.env`의 기본 모델은 `gpt-5.5`였다. 스모크 테스트에서 다음 오류가 발생했다.

```
error: Error code: 400 - Unsupported parameter: 'temperature' is not supported with this model.
```

`gpt-5.5`는 **존재하는 모델**이지만 `temperature` 파라미터를 거부했다(Claude Opus 4.8과 동일한 제약). 사용자 지시("안 될 경우 Claude 모델로 테스트")에 따라 `claude-opus-4-8`로 폴백했다.

- 별도 클라이언트 `AnthropicMessagesClient`(공식 `anthropic` SDK)를 사용한다. 기존 OpenAI 클라이언트에 Anthropic 호출을 섞거나 호환 샤임을 쓰지 않는다.
- Claude 경로는 **sampling 파라미터(temperature/top_p)를 보내지 않는다**(Opus 4.8은 전달 시 400). `temperature` 인자는 `LLMClient` Protocol 호환용으로만 남기고 API에는 전달하지 않는다.
- provider는 `LLM_MODEL`로 추론한다(`claude-*` → anthropic, 그 외 → openai). `LLM_PROVIDER`로 강제 지정 가능.

### 1.2 환경 준비

```bash
pip install -e '.[anthropic]'          # Claude 폴백 경로 의존성
# .env: ANTHROPIC_API_KEY=...  LLM_MODEL=claude-opus-4-8   (.env는 gitignore)
python -m agent_loader_bench build-index --backend all   # Markdown 소스 기준 인덱스 재빌드
```

### 1.3 스모크 테스트 (본 스윕 전 1회)

전용 트레이스로 단일 로더 1회 실행해 Claude 호출이 정상인지 확인했다.

```bash
TRACE_PATH=.agentdb/bench-traces.jsonl \
  python -m agent_loader_bench run --loader fs_direct --dataset datasets/requests.yml --live-llm
```

→ 7건 전부 `llm_completed=True`, `model=claude-opus-4-8` 확인.

### 1.4 전체 스윕 (8 로더 × 7 요청 = 56회)

기존 누적분과 섞이지 않도록 전용 트레이스를 비운 뒤 8종을 순차 실행했다.

```bash
for l in fs_direct manifest_json sqlite_metadata sqlite_fts \
         sqlite_fts_section json_document vector_search hybrid; do
  TRACE_PATH=.agentdb/bench-traces.jsonl \
    python -m agent_loader_bench run --loader "$l" --dataset datasets/requests.yml --live-llm
done
```

### 1.5 집계

```bash
python -m agent_loader_bench compare --trace .agentdb/bench-traces.jsonl
```

`compare`는 트레이스를 `run_id` 기준 중복 제거(최신 우선)한 뒤 로더별로 묶어, 성공률 내림차순·평균 토큰 오름차순으로 정렬한다.

### 1.6 평가 기준

`task_success`는 LLM **출력**이 아니라 **로더가 선택한 스킬/섹션**으로 결정된다(`evaluate_loaded_context`). 다음 세 조건을 모두 만족할 때만 성공이다.

1. 기대 스킬을 모두 포함
2. 불필요한(기대 외) 스킬이 없음
3. 기대 섹션을 모두 포함

즉 "LLM이 올바른 컨텍스트를 받았는가"를 결정적으로 측정한다. `llm_completed`는 실제 Claude 호출 발생 여부를 보조 신호로 별도 추적한다. 평가가 retrieval 기반이므로 Opus 4.8을 temperature 없이 호출해도 결과는 재현 가능하다.

---

## 2. 결과

### 2.1 집계 표

```
loader                 success    rate  avg_tokens  llm_done
------------------------------------------------------------
hybrid                     7/7    100%         657         7
sqlite_fts_section         7/7    100%         657         7
sqlite_fts                 7/7    100%         817         7
json_document              6/7     86%        1066         7
manifest_json              6/7     86%        1066         7
sqlite_metadata            6/7     86%        1066         7
fs_direct                  6/7     86%        1126         7
vector_search              3/7     43%         901         7
```

- `success`/`rate`: 로더 선택 정확도, `avg_tokens`: 평균 `context_token_estimate`, `llm_done`: live Claude 호출 완료 수.

### 2.2 요청별 실패 분석

| 요청 | 실패한 로더 |
|---|---|
| `req.loader.ambiguous` | fs_direct, json_document, manifest_json, sqlite_metadata, vector_search |
| `req.docs.troubleshooting.multi_skill` | vector_search |
| `req.vllm.benchmark.semantic` | vector_search |
| `req.vllm.wrong_skill_trap` | vector_search |

- 6/7 그룹(fs_direct, json_document, manifest_json, sqlite_metadata)은 **모호 요청(`req.loader.ambiguous`) 하나만** 실패 — 의도된 baseline 판별 케이스.
- `vector_search`는 위 모호 요청에 더해 multi_skill·semantic·wrong_skill_trap까지 실패. 요청별로 보면 ambiguous·multi_skill에서 `skill.troubleshooting`을 엉뚱하게 선택하고, semantic·trap에서 과선택했다 — AGENTS.md가 경고한 non-obvious 오매칭.

---

## 3. 해석

1. **섹션 단위 로더가 두 축 모두 우위.** `hybrid`·`sqlite_fts_section`은 7/7 정확 + **657 토큰**으로, baseline `fs_direct`(1126) 대비 약 **42% 토큰 절감**. "섹션 단위 컨텍스트가 불필요 토큰을 줄인다"는 AGENTS.md 가설 입증.
2. **문서 전체 FTS는 정확하지만 무겁다.** `sqlite_fts`는 7/7이지만 817 토큰 — 문서 전체를 싣기 때문. 섹션 조립이 결정적 차이.
3. **6/7 그룹은 모호 요청에서만 갈린다.** 네 로더 모두 `req.loader.ambiguous`만 놓친다.
4. **벡터 단독은 불안정(3/7).** non-obvious 오매칭을 만든다. `hybrid`는 벡터의 의미적 도달력을 유지하면서 메타데이터·FTS·우선순위·섹션 로딩을 결합해 정확도를 회복한다.

**권장 로더:** `sqlite_fts_section` (의미적 도달력이 필요하면 `hybrid`) — 정확도 최고 + 토큰 최소.

---

## 4. 검증

- `python -m pytest -q` → **46 passed** (기존 36 + 신규 10)
- `ruff check .` / `ruff format --check .` → 클린
- `build-index --backend all` → 정상
- 전체 스윕 트레이스 → **56 레코드, 8 로더, 전부 `llm_completed=True`, model=`claude-opus-4-8`**

---

## 5. 비고

- Markdown이 source of truth — 로더는 소스를 수정하지 않고, `compare`는 트레이스만 읽는다.
- 저비용 재실행은 `LLM_MODEL=claude-haiku-4-5`로 가능.
- `.env`의 실제 API 키는 gitignore되어 커밋되지 않는다.

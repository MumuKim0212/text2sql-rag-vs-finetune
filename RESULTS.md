# 실험 결과 — RAG vs 파인튜닝 비교 (Spider dev, n=1034)

평가 방법론은 [project-text2sql-brief.md](project-text2sql-brief.md) 참고. 조건별 원본 파일(프롬프트 `prompts.jsonl`, 예측 `*_predictions.jsonl`, 점수 `summary.json`)은 `data/results/<condition>/`에 함께 추적된다. 재다운로드 가능한 Spider 원본(`dev.json`, `train_spider.json`, 4.9GB `test_suite_database` 등)은 비추적.

## 비교표

채점은 공식 Spider 설정(`plug_value=False`)을 기본값으로 쓴다 — 리더보드 수치와 같은 기준. 각 `summary.json`에 이 설정이 함께 기록된다.

| 조건 | 모델 | Test-Suite Accuracy | Exact Match | 측정일 |
|---|---|---|---|---|
| 1. 클라우드 API (베이스라인) | Gemini 3.6 Flash | 83.1% | 80.5% | 2026-08-11 재채점 |
| 2. 로컬 베이스 모델 | Qwen2.5-Coder-7B (vLLM) | 72.7% | 56.5% | 2026-08-11 |
| 3a. 로컬 베이스 + RAG (few-shot) | Qwen2.5-Coder-7B + RAG | TBD | TBD | - |
| 3b. 로컬 베이스 + RAG + schema linking | Qwen2.5-Coder-7B + RAG | TBD | TBD | - |
| 4. 로컬 + LoRA/QLoRA | Qwen2.5-Coder-7B + FT | TBD | TBD | - |
| 5. 로컬 + FT + RAG | Qwen2.5-Coder-7B + FT + RAG | TBD | TBD | - |

## 조건 1: 클라우드 API 베이스라인 (Gemini 3.6 Flash)

- 원본 예측: `data/results/cloud_baseline/gemini_dev_predictions.jsonl`
- 요약: `data/results/cloud_baseline/summary.json`
- 실행: `uv run python scripts/run_cloud_baseline.py`

2026-08-11에 채점 기본값을 공식 Spider 설정(`plug_value=False`)으로 바꾼 뒤 API 재호출 없이 예측 파일 그대로 재채점함
(`uv run python scripts/score_predictions.py --predictions data/results/cloud_baseline/gemini_dev_predictions.jsonl --condition cloud_baseline`).
Test-Suite Accuracy는 85.6% → 83.1%로 소폭 하락(리더보드와 같은 기준이 됨). Exact Match(80.5%)는 실행이 아니라
쿼리 구조로 계산되므로 `plug_value`와 무관해 그대로다.

**난이도별 Test-Suite Accuracy** (plug_value=False, 공식 기준)

| easy | medium | hard | extra |
|---|---|---|---|
| 93.5% | 85.4% | 81.6% | 62.7% |

가장 복잡한 쿼리(서브쿼리 다중 중첩, 집합 연산 등)가 몰린 `extra` 구간에서만 큰 폭으로 떨어지는 패턴 — Spider 리더보드에서 흔히 보이는 경향과 일치. 이후 로컬 조건들의 성능 상한선 참고용.

## 조건 2: 로컬 베이스 모델 (Qwen2.5-Coder-7B-Instruct-AWQ, vLLM)

- 원본 예측: `data/results/local_base/qwen_dev_predictions.jsonl`
- 요약: `data/results/local_base/summary.json`
- 생성: Colab(T4)에서 `notebooks/condition2_local_qwen_colab.ipynb` 실행
- 채점: `uv run python scripts/score_predictions.py --predictions data/results/local_base/qwen_dev_predictions.jsonl --condition local_base`

RAG도 파인튜닝도 없는 순수 베이스 모델 성능. 클라우드 API(83.1%) 대비 Test-Suite Accuracy가 10.4%p 낮고,
Exact Match 격차(80.5% → 56.5%)는 더 크다 — 쿼리 구조 자체를 덜 정확히 맞추지만, 실행 결과가 맞아떨어지는
경우(예: 동치인 다른 형태의 쿼리)가 그보다는 많다는 뜻. RAG(3단계)·파인튜닝(4단계)이 이 격차를 얼마나
좁히는지가 다음 측정 포인트.

## 조건 3: 로컬 베이스 + RAG (measurement pending)

조건 2와 모델·샘플링 설정(`temperature=0.0`, `max_tokens=256`, AWQ, `max_model_len=4096`)이 동일하고 **프롬프트만 다르다**.
프롬프트의 `Schema:` 이후 부분은 조건 2와 바이트 단위로 같고, 앞에 검색된 예제 블록만 붙는다.

- 프롬프트 생성: `uv run python scripts/build_rag_prompts.py` (로컬 CPU, 두 변형을 한 번에 생성)
- 생성: Colab(T4)에서 `notebooks/condition3_rag_colab.ipynb`를 변형별로 1회씩
- 채점: `uv run python scripts/score_predictions.py --predictions ... --condition local_rag[_linked]`

**설정**

| 항목 | 값 |
|---|---|
| 임베딩 모델 | `all-MiniLM-L6-v2` (CPU, 정규화 후 cosine) |
| 검색 풀 | Spider train 질문 7,000개 (dev와 DB 분리 → 누수 없음) |
| few-shot 예제 수 | 5 |
| schema linking 유지 테이블 수 | 5 (3b만) |

두 변형은 한 번의 실행으로 함께 생성되어 **검색된 예제가 완전히 동일**하다. 따라서 3a↔3b 차이는 스키마 축소에서만 온다.

**측정 전에 확인된 제약 (결과 해석에 필요)**

Spider dev의 스키마는 작다 — 테이블 수 중앙값 3개, 스키마 프롬프트 중앙값 784자. 그래서 schema linking이
실제로 프롬프트를 바꾸는 건 1,034개 중 **252개(24.4%)** 뿐이고, 나머지 782개는 3a와 프롬프트가 동일하다.
즉 3b가 3a와 다를 수 있는 최대 폭이 24.4%로 이미 묶여 있다.

유지 테이블 수는 4가 아니라 5로 잡았다. 두 값의 pruning 적용 범위는 사실상 같은데(256개 vs 252개),
정답 쿼리가 필요로 하는 테이블을 잘라버리는 사례가 k=4에서 22개, k=5에서 6개로 차이가 크기 때문이다.

| 유지 테이블 수 k | 프롬프트가 바뀐 예제 | 정답에 필요한 테이블을 잘라먹은 예제 |
|---|---|---|
| 3 | 583 | 40 (3.9%) |
| 4 | 256 | 22 (2.1%) |
| **5** | **252** | **6 (0.6%)** |
| 6 | 160 | 5 (0.5%) |

k=5에서도 6개는 스키마 축소 때문에 확정적으로 틀리게 된다. 3b가 3a를 이기려면 이 손실을 넘는 이득이 있어야 한다.

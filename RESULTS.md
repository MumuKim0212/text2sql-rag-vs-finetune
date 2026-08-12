# 실험 결과 — RAG vs 파인튜닝 비교 (Spider dev, n=1034)

평가 방법론은 [project-text2sql-brief.md](project-text2sql-brief.md) 참고. 조건별 원본 파일(프롬프트 `prompts.jsonl`, 예측 `*_predictions.jsonl`, 점수 `summary.json`)은 `data/results/<condition>/`에 함께 추적된다. 재다운로드 가능한 Spider 원본(`dev.json`, `train_spider.json`, 4.9GB `test_suite_database` 등)은 비추적.

## 비교표

채점은 공식 Spider 설정(`plug_value=False`)을 기본값으로 쓴다 — 리더보드 수치와 같은 기준. 각 `summary.json`에 이 설정이 함께 기록된다.

| 조건 | 모델 | Test-Suite Accuracy | Exact Match | 측정일 |
|---|---|---|---|---|
| 1. 클라우드 API (베이스라인) | Gemini 3.6 Flash | 83.1% | 80.5% | 2026-08-11 재채점 |
| 2. 로컬 베이스 모델 | Qwen2.5-Coder-7B (vLLM) | 72.7% | 56.5% | 2026-08-11 |
| 3a. 로컬 베이스 + RAG (few-shot) | Qwen2.5-Coder-7B + RAG | 72.5% | 68.3% | 2026-08-12 |
| 3b. 로컬 베이스 + RAG + schema linking | Qwen2.5-Coder-7B + RAG | 72.6% | 68.0% | 2026-08-12 |
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

## 조건 3: 로컬 베이스 + RAG

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

### 결과: RAG는 정확도가 아니라 문체를 샀다

정답 개수로 보면 차이가 더 분명하다.

| 조건 | Test-Suite Accuracy | Exact Match |
|---|---|---|
| 2. 베이스 (RAG 없음) | **752**/1034 (72.7%) | 584/1034 (56.5%) |
| 3a. + few-shot | 750/1034 (72.5%) | **706**/1034 (68.3%) |
| 3b. + few-shot + schema linking | 751/1034 (72.6%) | 703/1034 (68.0%) |

**실행 정확도는 전혀 오르지 않았다** (752 → 750). 반면 **Exact Match는 +11.8%p 뛰었다** (584 → 706).
예측 자체는 1,034개 중 794개(76.8%)가 바뀌었는데도 실행 결과 기준 정답 수는 2개 줄었을 뿐이다.

Exact Match를 예제별로 다시 계산해(파싱만 하므로 DB 실행 불필요, 기록된 584/706과 정확히 일치 확인)
어떤 게 바뀌었는지 보면 원인이 드러난다 — RAG가 새로 맞힌 153개, 놓친 31개는 대부분 **실행 결과가
같은 표현 차이**다:

| 질문 | 베이스 | RAG |
|---|---|---|
| 2014 또는 2015년 콘서트 수 | `WHERE YEAR IN ('2014','2015')` | `WHERE YEAR = 2014 OR YEAR = 2015` |
| 2014년 콘서트가 없는 경기장 | `WHERE Stadium_ID NOT IN (SELECT ...)` | `EXCEPT SELECT ...` |
| 경기장별 콘서트 수 | `GROUP BY T2.Name` | `GROUP BY T2.Stadium_ID` |

즉 검색된 예제는 Spider의 **작성 관습**(`NOT IN`보다 `EXCEPT`, `IN`보다 `OR`, 이름 대신 키로 GROUP BY)을
가르쳤을 뿐, 모델이 원래 틀리던 문제를 풀게 해주지는 못했다.

**이유는 구조적이다.** 누수를 막으려고 train/dev의 DB를 분리해 놨기 때문에, 검색된 예제는 **평가 대상 DB의
스키마에 대해 아무것도 알려줄 수 없다.** 전달 가능한 건 표면 형식뿐이다. 질문 유사도 기반 few-shot 검색이
Spider에서 실행 정확도를 못 올리는 건 검색 품질 문제가 아니라 이 설계에서 나오는 필연에 가깝다.

부수적으로, Exact Match가 **정확성만큼이나 문체 일치를 재는 지표**라는 게 드러났다 — 실행 결과가 하나도
나아지지 않았는데 11.8%p가 올랐다. 조건 4·5에서 두 지표가 갈릴 때 이 점을 기억할 것.

### schema linking: 측정 가능한 효과 없음

3a와 3b 차이는 실행 정확도 1개(750 vs 751), Exact Match 3개(706 vs 703)로 **노이즈 수준이다.**

노이즈 바닥을 따로 쟀다: 프롬프트가 완전히 동일한 782개 중 **4개(0.5%)** 가 두 실행에서 다른 예측을 냈다.
`temperature=0.0`인데도 그런 이유는 32개씩 배치로 묶어 처리하는데 pruning으로 시퀀스 길이가 달라지면
같은 배치 안의 모든 시퀀스에서 부동소수점 누적 순서가 바뀌기 때문이다. 3a↔3b 차이(1~3개)는 이 바닥보다 작다.

프롬프트가 실제로 바뀐 252개 중 73개(29%)는 출력이 달라졌지만, 좋아진 만큼 나빠져 상쇄됐다.
Spider dev 스키마가 애초에 작아 잘라낼 게 없었다는 사전 진단과 일치한다 — **이 벤치마크에서 table-level
schema linking은 값어치가 없다.** 스키마가 크고 노이즈가 있는 BIRD(9단계)에서 다시 볼 문제.

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
| **3c. 로컬 베이스 + RAG + value linking** | Qwen2.5-Coder-7B + RAG | **73.8%** | **69.0%** | 2026-08-12 |
| 3d. 로컬 베이스 + value linking만 (few-shot 없음) | Qwen2.5-Coder-7B + RAG | TBD | TBD | - |
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

### 결과: 예제 검색은 정확도가 아니라 문체를 샀다

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

이 설명은 **예제 검색에만** 적용된다. 대상 DB를 직접 읽는 검색(조건 3c)은 이 논리에서 벗어나므로,
"RAG가 정확도를 못 올린다"로 일반화하지 않는다.

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

### 조건 3c: value linking

3a·3b가 놓친 축을 채우기 위해 추가했다. 예제 검색은 train DB에서 오므로 평가 대상 DB에 대해 아무것도
말해주지 못하지만, **대상 DB를 직접 읽는 검색은 그 제약을 받지 않는다.**

**측정된 헤드룸** (조건 3a 예측 기준)

| 항목 | 값 |
|---|---|
| gold 쿼리에 문자열 리터럴이 있는 예제 | 336/1034 (32.5%) |
| 그중 예측 리터럴이 gold와 불일치 | 77 (dev의 7.4%) |

전형적인 실패는 대소문자다. 모델은 `WHERE PetType = 'Cat'`이라고 쓰는데 DB는 `'cat'`으로 저장한다:

```
DB의 실제 PetType 값: [('cat',), ('dog',)]
  WHERE PetType = 'cat'  ->  1 rows
  WHERE PetType = 'Cat'  ->  0 rows
```

SQLite의 `=`는 대소문자를 구분하므로 조용히 0행을 반환한다. 채점은 `plug_value=False`(리더보드 기준)라
리터럴 오류가 그대로 감점된다 — `plug_value=True`였다면 가려졌을 실패다.

**방식**: 대상 DB의 TEXT 컬럼에서 distinct 값을 읽어(컬럼당 최대 200개), 질문에 **온전한 단어**로
등장하는 값을 프롬프트의 질문 바로 앞에 `테이블.컬럼 = "값"` 형태로 붙인다. 대소문자 무시 매칭이 핵심이다 —
질문의 표기와 다른 값이 걸리는 경우가 정확히 모델이 틀리던 지점이다.

값을 읽는 DB는 `data/spider/database/`(Spider 원본, 실제 배포 환경이 질의할 대상)다. **이건 누수가 아니다** —
브리프의 제약은 dev 질문과 gold SQL에 대한 것이고, 실제 시스템은 질의 대상 DB를 항상 손에 쥐고 있다.

1,034개 중 **391개(37.8%)** 에 값이 하나 이상 붙었고, 프롬프트당 평균 1.0개다.

**이 조건이 비교에서 갖는 의미**: 값 그라운딩은 **파인튜닝으로 대체할 수 없다.** dev DB의 셀 값은 어느
학습 스플릿에도 없기 때문이다. 즉 RAG가 파인튜닝 대비 구조적 우위를 갖는 축이고, 이걸 빼놓고 조건 4·5와
비교하면 RAG 쪽을 과소평가하게 된다.

#### 결과: 실행 정확도를 움직인 유일한 요소

| 조건 | Test-Suite Accuracy | Exact Match |
|---|---|---|
| 2. 베이스 | 752/1034 (72.7%) | 584/1034 (56.5%) |
| 3a. + few-shot | 750/1034 (72.5%) | 706/1034 (68.3%) |
| 3b. + schema linking | 751/1034 (72.6%) | 703/1034 (68.0%) |
| **3c. + value linking** | **763/1034 (73.8%)** | **713/1034 (69.0%)** |

3a 대비 **+13개**로, 노이즈 바닥(약 5개)을 넘는 유일한 상승이다. 예상 헤드룸 7.4%p 중 실현된 건 1.1%p다.

**주입한 값을 실제로 썼는지 확인**했다. 3a→3c에서 예측이 바뀐 189개 중 **186개가 값이 붙은 391개 안**에
있고, 값이 없는 643개 중에서는 3개(0.47%)만 바뀌었다 — 노이즈 바닥과 같다. 효과가 의도한 지점에만 발생했다.

리터럴 정확도(gold에 문자열 리터럴이 있는 336개 기준, 예측 리터럴 집합이 gold와 일치하는 수):

| 조건 | 리터럴 일치 |
|---|---|
| 2. 베이스 | 271/336 |
| 3a. few-shot | 259/336 |
| 3c. + values | 285/336 |

**few-shot은 리터럴을 오히려 12개 망가뜨렸다**(271 → 259). 문체를 배우면서 값 표기까지 예제를 따라간 것으로
보인다. 값 주입이 그걸 되돌리고 그 위에 +14를 얹었다.

**예상보다 넓은 효과 — 값이 컬럼까지 짚어준다.** 대소문자 교정만 예상했는데, `테이블.컬럼 = "값"` 형식이
**값이 어느 컬럼에 사는지**를 알려주기 때문에 컬럼·조인 선택까지 고쳤다:

| 질문 | 3a | 3c |
|---|---|---|
| American Motor Company의 차종 수 | `WHERE T1.Maker = '...'` | `WHERE T1.FullName = '...'` |
| volvo 모델의 실린더 수 | `WHERE T2.Make = 'Volvo'` | `WHERE T2.Model = "volvo"` |
| United Airlines의 ASY 도착편 수 | `JOIN airports ...` | `JOIN airlines ... WHERE T2.Airline = 'United Airlines'` |

즉 value linking은 리터럴 그라운딩이자 **데이터 기반 schema linking**으로도 작동한다 — 3b의 임베딩 기반
table pruning이 못 해낸 일을 값이 해냈다.

### 조건 3 종합

| 추가한 것 | 실행 정확도 변화 |
|---|---|
| few-shot 예제 검색 (train에서) | 0 (−2, 노이즈) |
| + table pruning | 0 (+1, 노이즈) |
| + 대상 DB 값 주입 | **+13** |

**Spider에서 RAG의 이득은 전부 대상 DB를 직접 읽는 데서 나왔고, 예제 검색에서는 하나도 나오지 않았다.**
train/dev DB가 분리돼 있어 검색된 예제는 평가 대상 스키마에 대해 말해줄 게 없고, 남는 건 표면 형식뿐이다.
Exact Match만 +11.8%p 오른 게 그 증거다.

이 결론이 조건 5의 구성을 정한다: 파인튜닝과 결합할 RAG는 3c 구성이어야 한다(브리프 진행 순서 7번).

### 조건 3d: value linking만 (측정 대기)

3c는 few-shot + values라서 값의 순수 기여가 예제 검색과 섞여 있다. few-shot이 리터럴 정확도를 12개
**떨어뜨렸으므로**(271 → 259) 값만 쓰는 편이 3c보다 나을 가능성도 있어, 추정 대신 측정한다.

3d를 넣으면 네 조건이 (few-shot × values)의 2×2가 된다:

| | few-shot 없음 | few-shot 있음 |
|---|---|---|
| **값 없음** | 조건 2 (752) | 3a (750) |
| **값 있음** | **3d (TBD)** | 3c (763) |

프롬프트 조립기는 두 블록을 각각 독립적으로 생략하므로, 둘 다 없으면 조건 2의 프롬프트가 그대로 나온다.
실제로 3d 프롬프트 1,034개 중 **값이 안 걸린 643개는 조건 2의 프롬프트와 바이트 단위로 동일**하다 —
그 구간이 내장된 대조군 역할을 한다. 프롬프트 길이 중앙값은 968자로 3c(1,844자)의 절반 남짓이다.

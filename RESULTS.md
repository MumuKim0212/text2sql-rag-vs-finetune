# 실험 결과 — RAG vs 파인튜닝 비교 (Spider dev, n=1034)

평가 방법론은 [project-text2sql-brief.md](project-text2sql-brief.md) 참고. 원본 예측/스코어 파일은 `data/results/<condition>/`에 있음(용량 문제로 git 비추적 — 이 문서가 결과의 기록판).

## 비교표

채점은 공식 Spider 설정(`plug_value=False`)을 기본값으로 쓴다 — 리더보드 수치와 같은 기준. 각 `summary.json`에 이 설정이 함께 기록된다.

| 조건 | 모델 | Test-Suite Accuracy | Exact Match | 측정일 |
|---|---|---|---|---|
| 1. 클라우드 API (베이스라인) | Gemini 3.6 Flash | 83.1% | 80.5% | 2026-08-11 재채점 |
| 2. 로컬 베이스 모델 | Qwen2.5-Coder-7B (vLLM) | 72.7% | 56.5% | 2026-08-11 |
| 3. 로컬 베이스 + RAG | Qwen2.5-Coder-7B + RAG | TBD | TBD | - |
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

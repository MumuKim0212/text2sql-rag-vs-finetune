# 실험 결과 — RAG vs 파인튜닝 비교 (Spider dev, n=1034)

평가 방법론은 [project-text2sql-brief.md](project-text2sql-brief.md) 참고. 원본 예측/스코어 파일은 `data/results/<condition>/`에 있음(용량 문제로 git 비추적 — 이 문서가 결과의 기록판).

## 비교표

채점은 공식 Spider 설정(`plug_value=False`)을 기본값으로 쓴다 — 리더보드 수치와 같은 기준. 각 `summary.json`에 이 설정이 함께 기록된다.

| 조건 | 모델 | Test-Suite Accuracy | Exact Match | 측정일 |
|---|---|---|---|---|
| 1. 클라우드 API (베이스라인) | Gemini 3.6 Flash | 85.6% ⚠️ | 80.5% | 2026-08-10 |
| 2. 로컬 베이스 모델 | Qwen2.5-Coder-7B (vLLM) | TBD | TBD | - |
| 3. 로컬 베이스 + RAG | Qwen2.5-Coder-7B + RAG | TBD | TBD | - |
| 4. 로컬 + LoRA/QLoRA | Qwen2.5-Coder-7B + FT | TBD | TBD | - |
| 5. 로컬 + FT + RAG | Qwen2.5-Coder-7B + FT + RAG | TBD | TBD | - |

## 조건 1: 클라우드 API 베이스라인 (Gemini 3.6 Flash)

- 원본 예측: `data/results/cloud_baseline/gemini_dev_predictions.jsonl`
- 요약: `data/results/cloud_baseline/summary.json`
- 실행: `uv run python scripts/run_cloud_baseline.py`

> ⚠️ **재채점 필요.** 아래 Test-Suite Accuracy 수치(85.6% 및 난이도별 값)는 `plug_value=True`로 측정됐다.
> 2026-08-11에 채점 기본값을 공식 Spider 설정(`plug_value=False`)으로 바꿨으므로, 조건 2~5와 같은
> 기준이 아니고 리더보드와도 비교할 수 없다. 예측 파일이 남아 있어 API 재호출 없이 채점만 다시 하면 된다:
>
> ```bash
> uv run python scripts/score_predictions.py --predictions data/results/cloud_baseline/gemini_dev_predictions.jsonl --condition cloud_baseline
> ```
>
> Exact Match(80.5%)는 실행이 아니라 쿼리 구조로 계산되므로 `plug_value`와 무관하며 그대로 유효하다.

**난이도별 Test-Suite Accuracy** (plug_value=True 기준 — 위 경고 참고)

| easy | medium | hard | extra |
|---|---|---|---|
| 96.0% | 87.4% | 86.2% | 64.5% |

가장 복잡한 쿼리(서브쿼리 다중 중첩, 집합 연산 등)가 몰린 `extra` 구간에서만 큰 폭으로 떨어지는 패턴 — Spider 리더보드에서 흔히 보이는 경향과 일치. 이후 로컬 조건들의 성능 상한선 참고용.

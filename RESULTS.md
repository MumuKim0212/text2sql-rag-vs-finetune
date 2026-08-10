# 실험 결과 — RAG vs 파인튜닝 비교 (Spider dev, n=1034)

평가 방법론은 [project-text2sql-brief.md](project-text2sql-brief.md) 참고. 원본 예측/스코어 파일은 `data/results/<condition>/`에 있음(용량 문제로 git 비추적 — 이 문서가 결과의 기록판).

## 비교표

| 조건 | 모델 | Test-Suite Accuracy | Exact Match | 측정일 |
|---|---|---|---|---|
| 1. 클라우드 API (베이스라인) | Gemini 3.6 Flash | **85.6%** | 80.5% | 2026-08-10 |
| 2. 로컬 베이스 모델 | Qwen2.5-Coder-7B (vLLM) | TBD | TBD | - |
| 3. 로컬 베이스 + RAG | Qwen2.5-Coder-7B + RAG | TBD | TBD | - |
| 4. 로컬 + LoRA/QLoRA | Qwen2.5-Coder-7B + FT | TBD | TBD | - |
| 5. 로컬 + FT + RAG | Qwen2.5-Coder-7B + FT + RAG | TBD | TBD | - |

## 조건 1: 클라우드 API 베이스라인 (Gemini 3.6 Flash)

- 원본 예측: `data/results/cloud_baseline/gemini_dev_predictions.jsonl`
- 요약: `data/results/cloud_baseline/summary.json`
- 실행: `uv run python scripts/run_cloud_baseline.py`

**난이도별 Test-Suite Accuracy**

| easy | medium | hard | extra |
|---|---|---|---|
| 96.0% | 87.4% | 86.2% | 64.5% |

가장 복잡한 쿼리(서브쿼리 다중 중첩, 집합 연산 등)가 몰린 `extra` 구간에서만 큰 폭으로 떨어지는 패턴 — Spider 리더보드에서 흔히 보이는 경향과 일치. 이후 로컬 조건들의 성능 상한선 참고용.

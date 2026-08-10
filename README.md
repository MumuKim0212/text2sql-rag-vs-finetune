# Text-to-SQL: RAG vs 파인튜닝 비교 연구

프로젝트 배경/목적/진행 순서는 [project-text2sql-brief.md](project-text2sql-brief.md) 참고. 실험 결과 비교표는 [RESULTS.md](RESULTS.md) 참고.

## 개발 환경

```bash
uv sync
```

## 프로젝트 구조

```
src/rag_text2sql/
  data/       # 데이터셋 로드 (WikiSQL, Spider)
  eval/       # 평가 스크립트 (test-suite accuracy, exact match)
  models/     # 로컬/클라우드 모델 서빙 클라이언트
  rag/        # 스키마/예제 검색 (RAG)
  finetune/   # LoRA/QLoRA 파인튜닝
  api/        # FastAPI 엔드포인트
scripts/      # 실행 스크립트 (평가 파이프라인 등)
data/         # 데이터셋 저장 위치 (git 추적 제외)
third_party/  # 외부 평가 코드 vendoring (taoyds/test-suite-sql-eval, Apache-2.0)
tests/
```

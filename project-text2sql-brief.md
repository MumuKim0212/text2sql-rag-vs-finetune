# 프로젝트: Text-to-SQL — RAG vs 파인튜닝 비교 연구

## 목적

동일한 평가셋(held-out)을 기준으로, 다음 5개 조건의 Text-to-SQL 성능을 정량 비교한다.

1. 클라우드 API (베이스라인, 상한선 참고용)
2. 로컬 베이스 모델 (파인튜닝/RAG 없음)
3. 로컬 베이스 모델 + RAG (스키마/예제 검색 증강)
4. 로컬 모델 + LoRA/QLoRA 파인튜닝 (RAG 없음)
5. 로컬 모델 + 파인튜닝 + RAG (결합)

결론으로 "RAG와 파인튜닝을 각각 언제 쓰는 것이 유리한가"를 데이터로 답하는 것이 이 프로젝트의 핵심 산출물이다.

### 조건 3의 분화 (2026-08-12)

"RAG"는 단일 기법이 아니라서 조건 3을 세 변형으로 나눠 각각 측정했다. 셋은 한 번의 실행으로 함께 생성되어
**검색된 few-shot 예제가 완전히 동일**하고, 따라서 변형 간 차이는 각자 추가한 요소에서만 온다.

| 변형 | 추가한 것 | 결과 (Test-Suite Accuracy) |
|---|---|---|
| 3a | train 질문 유사도 기반 few-shot 예제 5개 | 72.5% — 베이스(72.7%)와 차이 없음 |
| 3b | 3a + 임베딩 기반 table pruning (상위 5개) | 72.6% — 노이즈 수준 |
| 3c | 3a + **대상 DB에서 읽은 실제 값** 주입 | **73.8%** — 유일하게 유의미한 상승 |
| 3d | few-shot 없이 값 주입만 | 73.4% — 3c와 노이즈 범위 내 동률 |

조건 2/3a/3d/3c가 (few-shot × 값)의 2×2를 이루며, 두 요소가 **서로 다른 지표만** 움직인다는 것이 드러났다.

| 요소 | Test-Suite Accuracy | Exact Match |
|---|---|---|
| few-shot 예제 검색 | −2 / +4 (노이즈) | **+122 / +130** |
| 대상 DB 값 주입 | **+7 / +13** | −1 / +7 (노이즈) |

few-shot은 문체(Exact Match)를, 값 주입은 정확성(실행)을 산다. 3d의 Exact Match(583)가 조건 2(584)와
사실상 같다는 게 결정적 증거다.

핵심은 **RAG의 이득이 예제 검색이 아니라 대상 DB를 직접 읽는 데서만 나왔다**는 점이다. train/dev의 DB가
분리돼 있어(누수 방지 설계) 검색된 예제는 평가 대상 스키마에 대해 아무것도 말해줄 수 없고, 실제로 실행
정확도를 못 올렸다. 자세한 근거는 [RESULTS.md](RESULTS.md) 참고.

이 결과는 조건 5(파인튜닝 + RAG)의 구성도 결정한다 — 아래 진행 순서 7번 참고.

## 배경 / 동기

- 로컬 LLM을 실제로 써본 결과 속도와 품질이 클라우드 API 대비 떨어짐을 체감함
- 하드웨어 업그레이드보다 "어떤 조건에서 로컬 모델이 실용적인가"를 검증하는 것이 더 유의미하다고 판단
- 이 프로젝트는 방법론(RAG vs 파인튜닝 비교 프레임워크) 자체를 표준 벤치마크로 검증하는 역할이며, 프로젝트 2(전기공사 도메인 QA)에서 같은 프레임워크를 실전 도메인에 적용할 예정

## 데이터셋

- **1차 파이프라인 검증용**: WikiSQL (단일 테이블, 구조 단순 — 파이프라인 버그를 빠르게 잡기 위함)
- **본 실험용**: Spider (멀티 테이블, 표준 벤치마크, 리더보드 비교 가능)
- **확장 단계(8단계 완료 후)**: BIRD — 노이즈 있는 DB 값, 외부 지식 그라운딩이 포함되어 RAG vs 파인튜닝 비교에 더 현실적. Spider로 얻은 결론이 일반화되는지 재검증 용도. Spider와 마찬가지로 dev split만 평가셋으로 사용하고 train/파인튜닝/RAG 인덱스에 섞지 않음
- 세 데이터셋 모두 Hugging Face Hub에서 로드 (`datasets` 라이브러리)
- Spider 2.0(엔터프라이즈급, o1-preview도 21.3%만 해결)은 이 프로젝트 스코프에 비해 과도하게 복잡해 제외

### Spider 데이터셋 구성 (중요 — 스플릿별 용도 구분)

Spider는 train(공개, 정답 포함) / dev(공개, 정답 포함) / **test(비공개 — 공식 배포되지 않으며 리더보드 제출을 통해서만 접근 가능)** 로 나뉜다. 이 프로젝트는 test split을 사용하지 않는다.

| 스플릿 | 공개 여부 | 이 프로젝트에서의 용도 |
|---|---|---|
| train (7,000개, 140 DB) | 공개, 정답 포함 | 파인튜닝 학습 데이터 + RAG용 스키마/예제 검색 소스 |
| dev (1,034개, 20 DB) | 공개, 정답 포함 | **평가셋(고정, held-out)** — 5개 조건 전부 이걸로만 측정 |
| test (2,147개, 40 DB) | 비공개 | 사용하지 않음 |

train은 `xlangai/spider`의 train 스플릿(= 공식 `train_spider.json`, 7,000개)을 쓴다. 공식 배포본에는 다른 데이터셋(GeoQuery, Scholar, IMDB 등 6개 DB)에서 가져온 `train_others.json` 1,659개가 따로 있지만, 쿼리 분포가 Spider 본체와 달라 제외한다 — RAG 검색 풀과 파인튜닝 학습셋을 같은 7,000개로 맞춰야 조건 3과 조건 4가 같은 지식원을 쓴 것이 된다.

**데이터 누수 방지 (필수 준수 사항)**
- dev split은 오직 최종 평가에만 사용한다. RAG의 few-shot 예제 검색 풀이나 파인튜닝 학습 데이터에 dev 데이터가 섞이면 안 된다.
- train/dev/test는 스키마(DB)가 서로 겹치지 않게 설계되어 있으므로, RAG 검색 인덱스를 train 스키마로만 구축하면 자연스럽게 누수가 방지된다.
- 평가셋(dev)은 **처음에 한 번 확정 후 절대 변경하지 않음** — 5개 조건 모두 동일 평가셋 사용이 비교의 전제조건

## 평가 지표

- **Test-Suite Accuracy** (주 지표): 단일 DB 인스턴스 실행 결과 비교가 아니라, 정답 쿼리 기준으로 생성된 여러 DB 인스턴스 전체에서 실행 결과가 일치하는지 검증 — 단순 execution accuracy 대비 우연한 일치(false positive)를 줄임. Spider 공식 지표(2020~)이며 참조 구현 [taoyds/test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval) 재사용
- **Exact Match**: SQL 쿼리 문자열/구조 일치 여부 (참고 지표)
- 5개 조건 전체를 하나의 비교표로 정리 (조건 × 지표)
- 부가 지표: 응답 지연시간(latency), 로컬 모델의 경우 VRAM 사용량

## 기술 스택

- Python, FastAPI (평가 파이프라인을 API로 감싸 재사용 가능하게)
- 로컬 모델 서빙: vLLM (재시작 없이 LoRA 어댑터 교체 가능 — 5개 조건을 전환하며 측정하는 실험 구조에 필요. Ollama는 이 기능이 없어 배제)
- 로컬 베이스 모델: Qwen2.5-Coder-7B (코드/SQL 특화 베이스, 이미 SQL 전용으로 튜닝된 모델(SQLCoder 등)은 파인튜닝 전/후 비교 취지에 안 맞아 제외. VRAM 여유가 있으면 착수 시점 Qwen3-Coder 계열 재검토)
- 파인튜닝: Hugging Face `transformers`, `peft`(LoRA/QLoRA), `trl`
- RAG: 벡터 DB(pgvector 또는 로컬 파일 기반 cosine similarity — 기존 DBot 프로젝트 방식 재사용 가능), 스키마 설명 + few-shot 예제 검색. 검색 전 질문에서 관련 테이블/컬럼을 먼저 추리는 schema linking 단계 추가 고려
- 클라우드 API 베이스라인: Google Gemini (`gemini-3.6-flash`, `google-genai` SDK) — 보유한 API 키 기준으로 확정. Anthropic API로 전환할 경우를 대비해 `src/rag_text2sql/models/cloud.py`(Claude 클라이언트)는 유지, `--provider anthropic`으로 전환 가능

## 진행 순서

1. 평가셋 확정 (Spider **dev** split을 held-out 평가셋으로 고정 — test split은 비공개이므로 사용 불가)
2. 평가 스크립트 작성 (test-suite accuracy 자동 계산, 5개 조건 공통으로 재사용)
3. 클라우드 API 베이스라인 측정
4. 로컬 베이스 모델(Qwen2.5-Coder-7B) 서빙 환경 구축 (vLLM) 후 베이스라인 측정
5. RAG 파이프라인 구축 후 측정 — 세 변형(3a few-shot / 3b + schema linking / 3c + value linking)을 각각 측정 (완료)
6. LoRA/QLoRA 파인튜닝 (Spider train split 사용) 후 측정
7. 파인튜닝 모델 + RAG 결합 측정 — **RAG는 3d 구성(value linking만, few-shot 없음)을 쓴다.** 3c와 실행
   정확도가 동률(763 vs 759, 노이즈 범위)인데, few-shot이 사는 건 Spider 문체이고 조건 4가 바로 그
   Spider train으로 파인튜닝하므로 둘이 하는 일이 겹친다. 값 주입은 파인튜닝이 원리적으로 줄 수 없는
   것(dev DB의 실제 셀 값)만 공급해 겹치지 않는다. 프롬프트도 절반 길이라 지연시간에 유리하다.
   이로써 조건 2/3d/4/5가 "RAG × 파인튜닝"의 2×2가 된다. schema linking은 효과가 없어 탈락
8. 5개 조건 비교표 작성, 결론 도출
9. (확장) 동일 프레임워크를 BIRD 데이터셋에 재적용해 결론의 일반화 여부 검증

## 포트폴리오 방향

- 기존 DBot(경량 RAG 기반 Text-to-SQL) 프로젝트의 후속작으로 포지셔닝 — "RAG의 한계를 파인튜닝으로 어떻게 보완하는가"
- AI Backend/Platform 관점 강조: 파인튜닝 자체보다 서빙 구조(비동기 처리, 어댑터 병합, 양자화/GGUF 변환, API 엔드포인트화)와 운영 관점(지연시간 프로파일링, 리소스 사용량)을 함께 다룰 것
- 정량적 비교표가 핵심 자산 — 블로그 포스트/기술 문서의 중심 그래픽으로 사용

## 미확정 사항 (착수 시 확인 필요)

- **로컬 GPU 자원 → 일단 대기, Colab으로 진행 (2026-08-10 결정)**: 개인 GPU가 RTX 5060 8GB인데, (1) VRAM이 vLLM+QLoRA 동시 운용에 여유가 부족하고 (2) Blackwell(sm_120) 아키텍처가 아직 stable PyTorch/vLLM에서 정식 지원되지 않아 로컬 세팅 리스크가 큼. 대신 Colab(T4 16GB)으로 우선 진행. 단, Colab 무료 티어의 세션 12시간 제한/주간 GPU 할당량/배정 불확실성은 그대로 남아있어, 본 실험(5개 조건 정식 측정) 단계에서 끊김이 반복되면 Colab Pro나 RunPod 재검토 필요
- 베이스 모델은 Qwen2.5-Coder-7B로 잠정 확정(2026-08 기준 조사) — 착수 시점에 Qwen3-Coder 등 신규 모델 재조사 후 최종 확정
- 파인튜닝 시 사용할 정확한 하이퍼파라미터 (rank, alpha, learning rate 등)는 착수 후 실험적으로 결정 — 참고: 최근 QLoRA 사례는 r=64, alpha=16, lr 2e-4~2e-5 부근에서 시작
- **로컬 베이스 모델 생성/채점 분리 (2026-08-10)**: GPU가 필요한 예측 생성은 Colab에서, 채점은 로컬에서 진행. `notebooks/condition2_local_qwen_colab.ipynb`로 Colab에서 vLLM + Qwen2.5-Coder-7B-Instruct-AWQ(T4 16GB에서 fp16 풀모델은 KV 캐시 여유가 부족해 AWQ 양자화 사용)를 서빙해 예측 jsonl을 생성하고, `scripts/score_predictions.py`로 로컬에서 test-suite accuracy를 계산(4.9GB test_suite_database가 로컬에만 있어서). 예측 결과(jsonl)와 채점 결과(summary.json)를 분리해두면 재채점/재현이 쉬워짐 — 조건 3~5도 동일 패턴 재사용 예정

---
*2026-08-10: 최신 동향 조사 반영 (벤치마크 확장 계획, 로컬 모델/서빙 스택, 평가지표, RAG 기법 업데이트)*
*2026-08-10: 클라우드 API 베이스라인을 Anthropic API에서 Google Gemini로 확정 (보유 API 키 기준). 20개 파일럿에서 test-suite accuracy 0.9 확인 후 전환*
*2026-08-10: 조건 2 진행 방식을 노트북(Colab, 생성) + 스크립트(로컬, 채점) 분리로 정리*
*2026-08-12: 채점 속도 문제 해결 — 벤더링한 평가 코드가 쿼리마다 `asyncio.run()`으로 이벤트 루프를 만들었고(Spider dev 1회 채점에 약 8만 개), Windows에서는 이벤트 루프마다 `socket.socketpair()`가 실제 TCP 루프백 연결로 대체돼 채점이 수 시간 걸리고 간헐적으로 교착됐다. 동기 호출로 바꿔 **1,034개 채점이 37초**로 줄었다. 타임아웃은 원래 도달 불가능한 코드였으므로(코루틴에 `await`가 없어 `wait_for`가 선점 불가) 의미론은 동일하며, 기록된 5개 조건 전부 점수가 동일하게 재현되는 것을 확인했다 — `third_party/test_suite_sql_eval/NOTICE.md` 참고*
*2026-08-12: 조건 3을 schema linking 없는 변형(`local_rag`)과 포함 변형(`local_rag_linked`) 둘로 나눠 측정하기로 결정. 검색·프롬프트 조립은 로컬 CPU에서 미리 끝내고(`scripts/build_rag_prompts.py`) Colab은 완성된 프롬프트를 재생만 함 — 두 변형이 동일한 few-shot 예제를 쓰게 되어 차이가 스키마 축소에서만 나온다*
*2026-08-11: 채점 기본값을 `plug_value=False`(공식 Spider 기본값)로 확정 — 리더보드 비교 가능성을 유지하기 위함. 조건 1은 `plug_value=True`로 측정됐으므로 재채점 필요(RESULTS.md 참고)*

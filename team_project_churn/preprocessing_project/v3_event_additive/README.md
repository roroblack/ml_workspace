# v3_event_additive — 이벤트-단위 시퀀스 전처리 (추가형, 5m 호환)

목적: 실시간 세션 이탈/몰입을 위한 **이벤트-단위 시퀀스**(행동 간 시간간격 포함)를 **추가형**으로 얹는다. v1_daily·v2_5m·배포본을 **건드리지 않고** 신규 디렉토리·신규 파일·스키마 버전·DB 판별컬럼·신규 model_type로만 추가.

근거 연구(요약): TiSASRec(시간간격 임베딩)·Wide&Deep/DeepFM(하이브리드)·MTL/MMoE(멀티태스크) — DL이 GBM/Last-cat을 넘으려면 **시간간격 + Wide 통계피처 + 멀티태스크**가 핵심. 상세: [reports/17-7-x 전처리 최적화 계획서] 참조.

## 호환 원칙 (변경 없음 / 추가만)
| 영역 | 기존(유지) | 신규(추가) |
| --- | --- | --- |
| 디렉토리 | processed/, processed_5m/ | `processed_eventseq/` |
| artifact | sequences.npz, seq_scaler.npz | `event_sequences.npz`, `event_cat_vocab.json`, `event_scaler.joblib` |
| 스키마 | feature_schema.yaml 기존 | `event_sequence: {version: 2}` 섹션 추가 |
| DB | sequence_snapshot 기존 | **+`seq_type` 컬럼(DEFAULT 'daily')**, 신규는 'event' |
| 모델 | model_type='sequence' | `model_type='event_sequence'` |

## 계획 X/Y (확정 예정)
- X: 유저×세션 이벤트 시퀀스 `[N, L, F]`. F = {event_type 임베딩, category embedding, price(log1p), **Δt(다음행동까지 초)**, dwell-time bucket}.
- Y: 세션 이탈(이 클릭 후 30분 무이벤트=1) + 보조 헤드(다음 카테고리, 장바구니 의도) — 멀티태스크.

## 재현 소스 (`src/`)
| 파일 | 역할 | 실행 |
| --- | --- | --- |
| `migrate_seq_type.py` | `sequence_snapshot`에 `seq_type` 컬럼 추가(DEFAULT 'daily', 멱등) | `python .../migrate_seq_type.py` |
| `build_event_sequence.py` | **전처리만** — 이벤트 시퀀스(Δt·price·hour·event_type·category) 생성·DB 적재(seq_type='event') | `python .../build_event_sequence.py [zip]` |
| `smoke_regression.py` | 일별 배포본 무영향 회귀 검증(파일·DB·실제예측) | `python .../smoke_regression.py` |

## 실측 결과 (2026-06-20, 2019-Nov 표본 USER_CAP=4000)
- X: `X_cont[4000,50,4]`(price_log·dt_next_log·dt_prev_log·hour_norm, **train fit StandardScaler**) + `evt_idx[4000,50]`(event_type 4종) + `cat_idx[4000,50]`(category vocab 200+PAD+UNK) + `mask[4000,50]`.
- Y: `churn`(결과 7일 무활동=1), 이탈률 **87.8%**(관찰 14일 활동유저 코호트).
- DB: `sequence_snapshot` seq_type 분포 **{daily: 3911, event: 4000}** — **기존 daily 행 보존**.
- **회귀 스모크 PASS ✅** — 일별 산출물 존재·daily 행 보존·배포본 LSTM 예측 정상(prob=0.539).
- 산출 위치: `output/processed_eventseq/` (event_sequences.npz, event_cat_vocab.json, event_scaler.joblib, event_schema.json).

상태: **구현·검증 완료**. 다음(모델 담당): 이 입력으로 TiSASRec/Wide&Deep 학습 → DL 임베딩을 LightGBM 스태킹(상세 [17-7-1](../../reports/17-7-1_전처리최적화_논문리서치_계획서.md)).

# Project Rules

이 문서는 `team_project_churn`의 현재 기준 규칙이다. 과거 실험과 문서가 남아 있어도, 새 코드·모델·리포트·대시보드 기능은 아래 규칙을 우선한다.

## 1. 최종 데이터 소스

- **확정(2026-06-19 점검 완료): 데이터셋은 REES46로 최종 고정. 다른 데이터셋은 최종 모델/리포트/데모에 쓰지 않는다.**
- 최종 프로젝트 데이터 소스는 **REES46 eCommerce Events History in Cosmetics Shop**만 사용한다.
- REES46 산출물은 둘이며 리포트에서 혼동 금지:
  - **정식 학습 산출물(§3)**: `retail_rees46_multi/data/{tabular.csv, seq.npz, split.npz}` — 5개월·80,000명·18주 시퀀스. 최종 ML/DL 이탈 모델은 이걸 쓴다.
  - **실시간/세분 데모 파생본**: `sample_project/` — REES46 **2019-Nov만·8,000명 표본·초단위 원시 이벤트**(정식 주별 집계로는 초단위/세션 분석 불가하여 사용). 리포트 17-x가 사용. REES46가 맞으나 "REES46 2019-Nov 표본"으로 표기하고 정식 학습셋으로 오인 표기 금지.
- `sample_project/src/run_pipeline.py`는 기본값이 REES46이며 합성으로 자동 폴백하지 않는다(합성은 `synthetic` 인자 명시 시 데모 전용).
- 최종 모델 학습, 검증, 실시간 예측, 대시보드, 추천 기능은 모두 REES46 기반이어야 한다.
- 주요 원본 파일은 다음 5개월 zip이다.
  - `src/2019-Oct.csv.zip`
  - `src/2019-Nov.csv.zip`
  - `src/2019-Dec.csv.zip`
  - `src/2020-Jan.csv.zip`
  - `src/2020-Feb.csv.zip`
- 5개월 전처리 결과의 기준 위치는 다음이다.
  - `retail_rees46_multi/data/tabular.csv`
  - `retail_rees46_multi/data/seq.npz`
  - `retail_rees46_multi/data/split.npz`
  - `retail_rees46_multi/data/rees46_multi_preprocessed_5months.zip`

## 2. 사용하지 않는 데이터

- **Bank Customer Churn / Churn_Modelling.csv는 최종 모델에 사용하지 않는다.**
- Telco Churn, Mobile Game User Loss, Online Retail 등은 최종 모델 학습 데이터가 아니다.
- 위 데이터들은 과거 실험, 비교 설명, 아카이브 문서에서만 언급할 수 있다.
- 새 코드나 리포트에서 Bank/Telco/Game 데이터를 기본 데이터처럼 쓰면 안 된다.

## 3. 모델링 범위

- 최종 모델링 주제는 **REES46 행동 로그 기반 이탈 예측**이다.
- 정형 ML은 `retail_rees46_multi/data/tabular.csv`를 사용한다.
- 시퀀스 DL은 `retail_rees46_multi/data/seq.npz`와 `split.npz`를 사용한다.
- 학습/평가 분할은 가능한 한 `split.npz`를 공유해 ML/DL 비교 조건을 맞춘다.
- 라벨과 피처는 REES46의 관찰기간/결과기간 분리 원칙을 따라 미래 누수를 막는다.

## 4. 추천 기능 규칙

추천 기능은 허용한다. 단, **추천도 REES46만 가지고 만든다.**

허용되는 추천 입력:
- `user_id`
- `event_time`
- `event_type`
- `product_id`
- `category_id`
- `category_code`
- `brand`
- `price`
- `user_session`

추천 기능의 1차 목표:
- 유저의 최근 관심 카테고리, 브랜드, 상품을 파악한다.
- view/cart/purchase 행동을 가중치로 반영해 관심 점수를 만든다.
- 이탈 위험이 높은 유저에게 재방문/구매 전환을 유도할 상품, 카테고리, 브랜드를 추천한다.

권장 기본 가중치:
- `view = 1`
- `cart = 3`
- `purchase = 5`
- 최근 이벤트일수록 더 높은 가중치를 둔다.

추천 후보:
- REES46 원본 로그에 존재하는 `product_id`, `category_code`, `brand`만 사용한다.
- 외부 상품 메타데이터, 외부 리뷰, 외부 이미지, 다른 쇼핑몰 데이터는 사용하지 않는다.

대시보드 출력 예:
- 최근 관심 카테고리
- 최근 관심 브랜드
- 추천 상품 후보
- 장바구니 리마인드 대상
- 고위험 유저용 리텐션 액션

## 5. 실시간/대시보드 규칙

- 실시간 이벤트, 예측 로그, 추천 액션은 REES46 이벤트 스키마를 기준으로 한다.
- Vercel 시뮬레이션 사이트가 생성하는 이벤트도 REES46와 같은 형태여야 한다.
- Streamlit 대시보드는 REES46 기반 이탈 확률, 행동 그래프, 추천/개선안을 보여준다.
- 얼굴 로그인은 인증/권한 기능이며, 모델 학습 데이터로 사용하지 않는다.

## 6. 문서 작성 규칙

- 새 리포트에서 "사용 데이터"를 쓸 때는 기본적으로 **REES46**로 적는다.
- Bank Churn 결과는 "초기 베이스라인/과거 실험"으로만 표시한다.
- 최종 발표 문장에서는 Bank/Telco/Game 데이터를 본 프로젝트의 핵심 데이터처럼 말하지 않는다.

## 7. 충돌 시 우선순위

문서나 코드에 과거 Bank 기준 설명이 남아 있더라도, 현재 작업의 우선순위는 다음과 같다.

1. 이 파일 `PROJECT_RULES.md`
2. 최신 비전 명세서 `reports/05-5_프로그램_구조도_및_함수_변수_비전_명세서.md`
3. REES46 5개월 산출물 `retail_rees46_multi/`
4. 과거 Bank/Telco/Game 리포트와 코드

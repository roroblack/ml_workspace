# preprocessing_project 데이터셋 인덱스 (전 버전)

| 버전 | 종류 | 원본크기 | 백업 |
| --- | --- | --- | --- |
| v1_daily | 배포본 일별14 | (sample_project) | 정본 위치 |
| v2_5m | 5개월 10피처·7모델 | 96MB | output/OUTPUT_INDEX.md |
| v3_event_additive | 이벤트시퀀스 추가형 | (processed_eventseq) | output/ |
| v5_item_recommendation | 다음 상품(product_id) 예측 fast benchmark | sample raw 97,659 rows | output/benchmark_* |
| legacy_onlineretail | 온라인리테일(UCI) 이탈 | 0.8MB | 복사 |
| legacy_rees46_cosmetics | REES46 화장품 단월 이탈 | 122.5MB | 복사 |
| legacy_rees46_multi | REES46 다개월 시퀀스 이탈 | 27.1MB | 복사 |
| legacy_rees46_labels | REES46 라벨변형(base/A/B/C) | 66.6MB | 복사 |
| legacy_mobile_game | 모바일게임 레벨시퀀스 이탈 | 123.6MB | 복사 |
| aux_fullsession | 풀 세션-레벨 데이터셋 | 15.1MB | 복사 |
| aux_nextcat | 다음카테고리 예측 데이터셋 | 16.5MB | 복사 |
| aux_rec | 추천 전용(user×item) 데이터셋 | 31.8MB | 복사 |

> CAP=40MB 초과는 기본 포인터. 전체 물리 백업: `python preprocessing_project/backup_datasets.py --all`

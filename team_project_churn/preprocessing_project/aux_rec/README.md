# aux_rec — 추천 전용(user×item) 데이터셋

전처리 버전 관리 항목(과거 작업 백업).

## 구성
- **종류**: 추천 전용(user×item) 데이터셋
- **X 구성**: rec_user_interest_5m / rec_user_top_categories / rec_item_popularity .parquet
- **Y 구성**: 추천(무라벨)
- **재현 소스**: `preprocessing_project/v2_5m/src/rec_extract_5m.py` (원본 위치 유지)
- **원본 데이터**: `sample_project/data/processed_rec` (31.8 MB)
- **백업**: `output/` (복사됨) — 상세 `output/DATASET_BACKUP.md`

> 대용량은 git 비대화 방지를 위해 기본 포인터. `--all`로 물리 복사 가능.

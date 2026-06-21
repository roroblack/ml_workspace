# aux_fullsession — 풀 세션-레벨 데이터셋

전처리 버전 관리 항목(과거 작업 백업).

## 구성
- **종류**: 풀 세션-레벨 데이터셋
- **X 구성**: session_level_full.npz([N,T,F]) + recommend_user_interest.parquet
- **Y 구성**: session_level_full_meta.json
- **재현 소스**: `scripts/save_full_windows.py, session_full_1734.py` (원본 위치 유지)
- **원본 데이터**: `sample_project/data/processed_full` (15.1 MB)
- **백업**: `output/` (복사됨) — 상세 `output/DATASET_BACKUP.md`

> 대용량은 git 비대화 방지를 위해 기본 포인터. `--all`로 물리 복사 가능.

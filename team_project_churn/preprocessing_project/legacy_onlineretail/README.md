# legacy_onlineretail — 온라인리테일(UCI) 이탈

전처리 버전 관리 항목(과거 작업 백업).

## 구성
- **종류**: 온라인리테일(UCI) 이탈
- **X 구성**: tabular.csv(RFM) + seq.npz([N,T,F])
- **Y 구성**: split.npz(train/test) 라벨
- **재현 소스**: `retail/*.py (analyze/label_features/train_*)` (원본 위치 유지)
- **원본 데이터**: `retail/data` (0.8 MB)
- **백업**: `output/` (복사됨) — 상세 `output/DATASET_BACKUP.md`

> 대용량은 git 비대화 방지를 위해 기본 포인터. `--all`로 물리 복사 가능.

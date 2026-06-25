# REES46 화장품 이벤트 데이터 — 이탈 예측 (1개월 테스트)

## 데이터 준비 (사용자 작업)
- 데이터셋: Kaggle **"eCommerce Events History in Cosmetics Shop"** (by mkechinov)
  - 파일: `2019-Oct.csv`, `2019-Nov.csv`, `2019-Dec.csv`, `2020-Jan.csv`, `2020-Feb.csv`
- 전부 다운로드하되, **테스트용 1개 파일**(권장: `2019-Nov.csv`)을 아래 경로에 복사:
  ```
  team_project_churn/retail_rees46/data/raw/2019-Nov.csv
  ```
  (raw/ 안의 첫 CSV를 자동으로 사용)

## 실행 (파일 배치 후)
```powershell
$py="C:\Users\playdata2\anaconda3\python.exe"
$d="team_project_churn\retail_rees46"
& $py "$d\prep_rees46.py"   # 라벨링+피처+시퀀스 생성
& $py "$d\train_ml.py"      # LogReg/RF/GBM
& $py "$d\train_dl.py"      # MLP + RNN/LSTM/Attention/Transformer
& $py "$d\analyze.py"       # 이탈 영향요인
```

## 설계 요약
- **churn 정의**: 활동 기반(접속형) — 관찰기간(월 앞 70%) 피처 → 결과기간(뒤 30%)에 **어떤 이벤트도 없음 = 이탈**.
- **정형 피처**: recency·active_days·sessions·event수(view/cart/purchase)·고유상품/브랜드·평균가격·view→purchase 전환·tenure.
- **일별 시퀀스** `[N, 일수, 3]` (view/cart/purchase 일별 횟수) → 시퀀스 DL의 본 무대.
- **사용자 표본 상한** `USER_CAP=60000` (CPU 고려; prep_rees46.py에서 조정).

> 1개월만 쓰면 관찰/결과 윈도우가 짧아 라벨이 거칠 수 있습니다(프로토타입). 정식 버전은 여러 달을 이어 관찰/결과 기간을 길게 잡는 것을 권장합니다.

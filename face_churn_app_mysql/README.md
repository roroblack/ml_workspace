# InsightFace 얼굴 로그인 + 고객 이탈 예측 Streamlit 앱

이 프로젝트는 `Streamlit + OpenCV + InsightFace(ArcFace 계열 임베딩) + scikit-learn`을 사용하여 만든 교육용·실습용 웹 애플리케이션입니다.

앱은 먼저 얼굴을 등록하고, 등록된 얼굴과 로그인 얼굴을 비교하여 인증합니다. 얼굴 로그인에 성공한 사용자만 고객 이탈 예측 서비스를 사용할 수 있습니다.

---

## 1. 프로젝트 구조

```text
face_churn_app/
├─ app.py                         # Streamlit 메인 실행 파일
├─ train_churn_model.py            # 고객 이탈 예측 모델 학습 및 저장 스크립트
├─ requirements.txt                # 필요한 Python 패키지 목록
├─ README.md                       # 프로젝트 설명 문서
├─ app/
│  ├─ __init__.py                  # Python 패키지 인식 파일
│  ├─ face_auth.py                 # InsightFace 얼굴 등록/로그인 기능
│  ├─ churn_service.py             # 고객 이탈 예측 모델 로딩/예측 기능
│  └─ ui.py                        # Streamlit 세션/로그아웃 보조 기능
├─ data/
│  └─ telco_churn_sample.csv       # 실습용 고객 이탈 데이터셋
├─ models/
│  └─ churn_model.joblib           # 사전 학습된 고객 이탈 예측 모델
└─ registered_faces/
   └─ .gitkeep                     # 등록 얼굴 이미지 저장 폴더 유지용 파일
```

---

## 2. 실행 방법

### 2.1 가상환경 생성

```bash
python -m venv .venv
```

### 2.2 가상환경 활성화

Windows PowerShell:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2.3 패키지 설치

```bash
pip install -r requirements.txt
```

### 2.4 고객 이탈 예측 모델 재학습이 필요한 경우

프로젝트에는 `models/churn_model.joblib` 파일이 포함되어 있으므로 일반 실행 시에는 재학습이 필요 없습니다.

다만 모델 파일을 삭제했거나 데이터를 바꾸어 다시 학습하려면 다음 명령을 실행합니다.

```bash
python train_churn_model.py
```

### 2.5 앱 실행

```bash
streamlit run app.py
```

---

## 3. 얼굴 로그인 모델 설명

이 프로젝트는 InsightFace의 `FaceAnalysis(name="buffalo_l")` 모델팩을 사용합니다.

InsightFace는 얼굴 검출, 얼굴 정렬, 얼굴 특징 추출을 제공하는 얼굴 분석 라이브러리입니다. `buffalo_l` 모델팩은 RetinaFace 기반 얼굴 검출 모델과 ArcFace 계열 얼굴 인식 임베딩 모델을 포함합니다.

앱의 얼굴 로그인 절차는 다음과 같습니다.

```text
1. 사용자가 얼굴 이미지를 등록한다.
2. InsightFace가 얼굴을 검출한다.
3. 가장 큰 얼굴 하나를 선택한다.
4. 얼굴 이미지를 512차원 특징 벡터로 변환한다.
5. 특징 벡터를 L2 정규화한다.
6. 사용자 ID와 함께 face_db.npy 파일에 저장한다.
7. 로그인 시 촬영한 얼굴도 같은 방식으로 특징 벡터로 변환한다.
8. 등록된 벡터와 로그인 벡터의 코사인 유사도를 계산한다.
9. 유사도가 임계값 이상이면 로그인 성공으로 판단한다.
```

### 코사인 유사도 기준

두 얼굴 임베딩 벡터가 비슷할수록 코사인 유사도는 1에 가까워집니다.

```text
유사도 높음  → 같은 사람일 가능성 높음
유사도 낮음  → 다른 사람일 가능성 높음
```

앱의 기본 임계값은 `0.45`입니다. 수업 실습에서는 사이드바 슬라이더로 임계값을 조정하면서 보안성과 편의성의 차이를 확인할 수 있습니다.

---

## 4. InsightFace 모델 다운로드 경로

InsightFace Python 패키지는 `insightface>=0.3.3`부터 `FaceAnalysis()` 초기화 시 필요한 모델을 자동 다운로드할 수 있습니다.

자동 다운로드 방식:

```python
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1, det_size=(640, 640))
```

수동 다운로드 명령 예시:

```bash
insightface-cli model.download buffalo_l
```

모델팩은 일반적으로 다음 경로에 저장됩니다.

Windows:

```text
C:\Users\사용자명\.insightface\models\buffalo_l
```

macOS/Linux:

```text
~/.insightface/models/buffalo_l
```

공식 참고 경로:

- InsightFace GitHub: https://github.com/deepinsight/insightface
- Python package README: https://github.com/deepinsight/insightface/blob/master/python-package/README.md
- Model Zoo README: https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md
- SourceForge mirror 예시: https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download

### 라이선스 주의

InsightFace 코드 자체는 MIT 라이선스입니다. 그러나 InsightFace에서 제공하는 사전학습 모델팩은 비상업적 연구 목적으로 제공됩니다. 상용 서비스에 적용하려면 모델팩에 대한 별도 상용 라이선스를 확인해야 합니다.

---

## 5. 고객 이탈 예측 모델 설명

고객 이탈 예측 기능은 scikit-learn의 `Pipeline`으로 구성되어 있습니다.

사용 모델:

```text
ColumnTransformer
 ├─ 숫자형 컬럼: StandardScaler
 └─ 범주형 컬럼: OneHotEncoder
RandomForestClassifier
```

RandomForestClassifier는 여러 개의 의사결정나무를 만들고 그 결과를 종합하여 분류하는 앙상블 모델입니다. 고객 이탈 예측처럼 숫자형 변수와 범주형 변수가 섞여 있는 문제에서 안정적인 기준 모델로 사용할 수 있습니다.

앱에서는 다음 파일을 사전 학습된 모델로 사용합니다.

```text
models/churn_model.joblib
```

Streamlit 앱은 실행 중 이 파일을 읽어서 사용하며, 로그인 후 입력한 고객 정보를 바탕으로 다음 값을 출력합니다.

```text
- 이탈 위험 / 잔류 가능성 높음
- 이탈 확률
- 잔류 확률
- 간단한 고객 관리 안내 메시지
```

---

## 6. 고객 이탈 데이터셋 설명

이 프로젝트의 데이터 구조는 IBM Telco Customer Churn 데이터셋을 참고했습니다.

IBM Telco Customer Churn 데이터셋은 통신사 고객의 서비스 가입 정보, 계약 정보, 요금 정보, 인구통계 정보 등을 바탕으로 고객이 지난달 이탈했는지 예측하는 데 사용됩니다.

대표 컬럼은 다음과 같습니다.

```text
gender              성별
SeniorCitizen       고령 고객 여부
Partner             배우자 여부
Dependents          부양가족 여부
tenure              가입 기간
PhoneService        전화 서비스 사용 여부
MultipleLines       복수 회선 사용 여부
InternetService     인터넷 서비스 유형
OnlineSecurity      온라인 보안 서비스 여부
OnlineBackup        온라인 백업 서비스 여부
DeviceProtection    기기 보호 서비스 여부
TechSupport         기술 지원 여부
StreamingTV         스트리밍 TV 여부
StreamingMovies     스트리밍 영화 여부
Contract            계약 유형
PaperlessBilling    전자 청구서 사용 여부
PaymentMethod       결제 방식
MonthlyCharges      월 요금
TotalCharges        총 요금
Churn               고객 이탈 여부
```

실습 편의를 위해 이 프로젝트에는 `data/telco_churn_sample.csv`가 포함되어 있습니다. 이 파일은 IBM Telco Customer Churn의 컬럼 구조를 참고하여 생성한 교육용 샘플 데이터입니다.

실제 공개 데이터셋을 사용하려면 Kaggle 또는 IBM Sample Data에서 원본 Telco Customer Churn 데이터를 내려받은 뒤 `data/telco_churn_sample.csv`와 동일한 컬럼 구조로 저장하고 `python train_churn_model.py`를 다시 실행하면 됩니다.

공식/공개 참고 경로:

- IBM Telco customer churn sample: https://www.ibm.com/docs/en/cognos-analytics/12.0.x?topic=samples-telco-customer-churn
- Kaggle Telco Customer Churn: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

## 7. 주요 파일별 역할

### app.py

Streamlit 화면 전체를 구성합니다. 로그인 전에는 얼굴 등록과 얼굴 로그인 탭을 보여 주고, 로그인 후에는 고객 이탈 예측 입력 폼을 보여 줍니다.

### app/face_auth.py

InsightFace 모델을 초기화하고, 얼굴 임베딩을 추출하고, 등록된 얼굴 DB와 로그인 얼굴을 비교합니다.

### app/churn_service.py

`models/churn_model.joblib` 파일을 불러와 고객 이탈 확률을 예측합니다.

### train_churn_model.py

고객 이탈 예측 모델을 학습하고 `models/churn_model.joblib` 파일로 저장합니다.

---

## 8. 실습 시 자주 발생하는 오류와 해결

### 8.1 insightface 설치 오류

Windows에서 `insightface` 설치 중 빌드 오류가 발생하면 Python 3.10 또는 3.11 환경을 권장합니다.

```bash
python --version
```

권장 버전:

```text
Python 3.10.x 또는 Python 3.11.x
```

### 8.2 onnxruntime 오류

CPU 환경에서는 다음 패키지를 사용합니다.

```bash
pip install onnxruntime
```

GPU 환경에서는 CUDA 버전에 맞는 `onnxruntime-gpu`를 별도로 설치해야 합니다.

### 8.3 얼굴을 찾지 못하는 경우

다음 조건을 확인합니다.

```text
- 정면 얼굴인지 확인
- 얼굴이 너무 작지 않은지 확인
- 조명이 너무 어둡지 않은지 확인
- 마스크나 손으로 얼굴이 가려지지 않았는지 확인
- 한 이미지에 여러 명이 같이 있지 않은지 확인
```

### 8.4 고객 이탈 모델 파일이 없다는 오류

다음 명령으로 모델을 다시 생성합니다.

```bash
python train_churn_model.py
```

---

## 9. 교육용 확장 과제

1. 얼굴 로그인 성공/실패 로그를 CSV 또는 SQLite에 저장하기
2. 얼굴 등록 시 같은 사용자 ID 중복 등록 여부 확인하기
3. 관리자 계정만 고객 이탈 예측 결과를 볼 수 있도록 권한 추가하기
4. 고객 이탈 예측 결과를 DB에 저장하기
5. 이탈 확률이 높은 고객에게 자동 추천 전략을 출력하기
6. 실제 IBM Telco Customer Churn 원본 데이터로 모델 다시 학습하기
7. RandomForestClassifier 대신 LogisticRegression, XGBoost, LightGBM 모델과 비교하기

---

## 10. 보안 주의사항

이 프로젝트는 교육용 실습 앱입니다. 실제 서비스에 적용하려면 다음 보완이 필요합니다.

```text
- 얼굴 이미지와 임베딩 파일 암호화
- HTTPS 적용
- 사용자 계정/비밀번호 인증과 얼굴 인증의 2단계 인증 구성
- 로그인 실패 횟수 제한
- 위조 얼굴 이미지 방어 기능 적용
- 모델 라이선스 검토
- 개인정보 처리방침 및 생체정보 수집 동의 절차 구현
```

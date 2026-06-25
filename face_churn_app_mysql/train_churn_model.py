"""Telco 고객 이탈 예측용 사전 학습 모델을 생성하는 스크립트입니다."""

# 경로 관리를 위해 pathlib의 Path 클래스를 가져옵니다.
from pathlib import Path

# 모델 저장을 위해 joblib을 가져옵니다.
import joblib

# NumPy는 재현 가능한 샘플 데이터 생성을 위해 사용합니다.
import numpy as np

# pandas는 학습 데이터를 표 형태로 구성하기 위해 사용합니다.
import pandas as pd

# ColumnTransformer는 숫자형과 범주형 컬럼에 서로 다른 전처리를 적용하기 위해 사용합니다.
from sklearn.compose import ColumnTransformer

# RandomForestClassifier는 비선형 패턴을 잘 학습하는 고객 이탈 분류 모델로 사용합니다.
from sklearn.ensemble import RandomForestClassifier

# Pipeline은 전처리와 모델을 하나의 객체로 묶어 저장하기 위해 사용합니다.
from sklearn.pipeline import Pipeline

# OneHotEncoder는 범주형 문자를 머신러닝 모델이 처리할 수 있는 숫자 벡터로 변환합니다.
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# train_test_split은 학습/검증 데이터를 나누는 데 사용합니다.
from sklearn.model_selection import train_test_split

# classification_report는 모델 성능을 간단히 확인하기 위해 사용합니다.
from sklearn.metrics import classification_report


# 학습 데이터 파일 경로를 정의합니다.
DATA_PATH = Path("data/telco_churn_sample.csv")

# 모델 저장 폴더 경로를 정의합니다.
MODEL_DIR = Path("models")

# 모델 저장 파일 경로를 정의합니다.
MODEL_PATH = MODEL_DIR / "churn_model.joblib"


# 모델에 사용할 숫자형 컬럼 목록입니다.
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]

# 모델에 사용할 범주형 컬럼 목록입니다.
CATEGORICAL_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def generate_sample_dataset(n_rows: int = 1200) -> pd.DataFrame:
    """IBM Telco Customer Churn 구조를 참고한 실습용 샘플 데이터를 생성합니다."""

    # 난수 발생기를 고정하여 매번 동일한 학습 데이터가 생성되도록 합니다.
    rng = np.random.default_rng(42)

    # 고객 유지 기간을 0~72개월 범위에서 생성합니다.
    tenure = rng.integers(0, 73, size=n_rows)

    # 월 요금을 18~120 사이의 값으로 생성합니다.
    monthly = rng.uniform(18, 120, size=n_rows).round(2)

    # 총 요금은 유지 기간과 월 요금의 곱에 약간의 잡음을 더해 생성합니다.
    total = np.maximum(0, tenure * monthly + rng.normal(0, 80, size=n_rows)).round(2)

    # 주요 범주형 고객 특성을 무작위로 생성합니다.
    df = pd.DataFrame({
        "gender": rng.choice(["Male", "Female"], size=n_rows),
        "SeniorCitizen": rng.choice(["0", "1"], size=n_rows, p=[0.84, 0.16]),
        "Partner": rng.choice(["Yes", "No"], size=n_rows),
        "Dependents": rng.choice(["Yes", "No"], size=n_rows, p=[0.3, 0.7]),
        "tenure": tenure,
        "PhoneService": rng.choice(["Yes", "No"], size=n_rows, p=[0.9, 0.1]),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], size=n_rows),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], size=n_rows, p=[0.35, 0.45, 0.2]),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], size=n_rows),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], size=n_rows),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], size=n_rows),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], size=n_rows),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], size=n_rows),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], size=n_rows),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n_rows, p=[0.55, 0.25, 0.2]),
        "PaperlessBilling": rng.choice(["Yes", "No"], size=n_rows),
        "PaymentMethod": rng.choice([
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ], size=n_rows),
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    })

    # 이탈 위험 점수를 만들기 위한 기본 로짓 값을 계산합니다.
    logit = -1.0

    # 월 단위 계약 고객은 이탈 위험을 높입니다.
    logit += (df["Contract"] == "Month-to-month").astype(float) * 1.25

    # 장기 계약 고객은 이탈 위험을 낮춥니다.
    logit -= (df["Contract"] == "Two year").astype(float) * 1.0

    # 광케이블 인터넷 고객은 높은 요금과 결합되어 이탈 위험이 커질 수 있습니다.
    logit += (df["InternetService"] == "Fiber optic").astype(float) * 0.55

    # 전자수표 결제 고객은 원본 Telco 데이터에서도 이탈 위험이 높은 편으로 자주 분석됩니다.
    logit += (df["PaymentMethod"] == "Electronic check").astype(float) * 0.55

    # 기술 지원이 없는 고객은 서비스 불만족 가능성이 높다고 가정합니다.
    logit += (df["TechSupport"] == "No").astype(float) * 0.45

    # 온라인 보안 서비스가 없는 고객도 이탈 위험이 높다고 가정합니다.
    logit += (df["OnlineSecurity"] == "No").astype(float) * 0.35

    # 가입 기간이 길수록 이탈 가능성이 낮아지도록 반영합니다.
    logit -= df["tenure"] * 0.035

    # 월 요금이 높을수록 이탈 가능성이 높아지도록 반영합니다.
    logit += (df["MonthlyCharges"] - 65) * 0.012

    # 로짓을 0~1 사이 확률로 변환합니다.
    churn_probability = 1 / (1 + np.exp(-logit))

    # 확률을 기준으로 이탈 여부를 샘플링합니다.
    df["Churn"] = rng.binomial(1, churn_probability)

    # 학습용 CSV를 반환합니다.
    return df


def load_or_create_dataset() -> pd.DataFrame:
    """학습 데이터가 있으면 불러오고, 없으면 샘플 데이터를 생성합니다."""

    # data 폴더가 없으면 생성합니다.
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    # CSV 파일이 있으면 해당 파일을 읽어 사용합니다.
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)

    # CSV 파일이 없으면 실습용 샘플 데이터를 생성합니다.
    df = generate_sample_dataset()

    # 생성한 데이터를 CSV로 저장하여 이후 동일한 데이터를 재사용합니다.
    df.to_csv(DATA_PATH, index=False)

    # 생성된 DataFrame을 반환합니다.
    return df


def build_pipeline() -> Pipeline:
    """전처리와 RandomForest 모델을 하나로 묶은 파이프라인을 생성합니다."""

    # 숫자형 컬럼은 평균 0, 표준편차 1 기준으로 표준화합니다.
    numeric_transformer = StandardScaler()

    # 범주형 컬럼은 원-핫 인코딩하고, 학습 때 없던 값이 들어와도 오류가 나지 않게 처리합니다.
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # 컬럼 종류별 전처리기를 하나로 묶습니다.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ]
    )

    # RandomForest는 여러 의사결정나무를 조합하여 안정적인 분류 결과를 만드는 앙상블 모델입니다.
    classifier = RandomForestClassifier(
        n_estimators=250,
        max_depth=9,
        random_state=42,
        class_weight="balanced",
    )

    # 전처리와 모델을 순서대로 실행하는 파이프라인을 만듭니다.
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", classifier)])

    # 완성된 파이프라인 객체를 반환합니다.
    return pipeline


def main() -> None:
    """학습 데이터를 이용해 고객 이탈 예측 모델을 학습하고 저장합니다."""

    # 모델 저장 폴더가 없으면 생성합니다.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 학습 데이터를 불러오거나 생성합니다.
    df = load_or_create_dataset()

    # 입력 특성 X와 정답 y를 분리합니다.
    X = df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
    y = df["Churn"].astype(int)

    # 학습 데이터와 검증 데이터를 8:2 비율로 나눕니다.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 전처리와 모델이 포함된 파이프라인을 생성합니다.
    model = build_pipeline()

    # 학습 데이터를 이용해 모델을 학습합니다.
    model.fit(X_train, y_train)

    # 검증 데이터에 대한 예측 결과를 계산합니다.
    y_pred = model.predict(X_test)

    # 검증 성능 리포트를 터미널에 출력합니다.
    print(classification_report(y_test, y_pred, digits=4))

    # 학습이 완료된 모델 파이프라인을 joblib 파일로 저장합니다.
    joblib.dump(model, MODEL_PATH)

    # 저장 위치를 출력합니다.
    print(f"모델 저장 완료: {MODEL_PATH}")


# 이 파일을 직접 실행할 때만 main 함수를 실행합니다.
if __name__ == "__main__":
    main()

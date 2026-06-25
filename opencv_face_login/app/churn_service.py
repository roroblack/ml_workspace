"""고객 이탈 예측 모델 로딩과 예측 기능을 담당하는 모듈입니다."""

# 파일 경로를 안전하게 다루기 위해 Path 클래스를 가져옵니다.
from pathlib import Path

# 타입 힌트로 반환 구조를 명확하게 표현하기 위해 Dict를 가져옵니다.
from typing import Dict

# joblib은 scikit-learn 파이프라인 모델을 저장하고 불러올 때 사용합니다.
import joblib

# pandas는 입력값을 모델이 예측할 수 있는 표 형태 데이터로 변환하는 데 사용합니다.
import pandas as pd


# 사전 학습된 고객 이탈 예측 모델 파일 경로를 정의합니다.
MODEL_PATH = Path("models/churn_model.joblib")


# 모델을 반복해서 디스크에서 읽지 않도록 메모리에 캐시하기 위한 전역 변수입니다.
_CHURN_MODEL = None


def load_churn_model():
    """사전 학습된 고객 이탈 예측 모델을 로딩합니다."""

    # 전역 캐시 변수를 함수 내부에서 수정하기 위해 global 키워드를 사용합니다.
    global _CHURN_MODEL

    # 이미 모델이 로딩되어 있으면 기존 객체를 반환합니다.
    if _CHURN_MODEL is not None:
        return _CHURN_MODEL

    # 모델 파일이 없으면 사용자가 학습 스크립트를 실행하도록 명확한 오류를 발생시킵니다.
    if not MODEL_PATH.exists():
        raise FileNotFoundError("models/churn_model.joblib 파일이 없습니다. 먼저 python train_churn_model.py를 실행하세요.")

    # joblib로 저장된 scikit-learn 파이프라인 모델을 불러옵니다.
    _CHURN_MODEL = joblib.load(MODEL_PATH)

    # 로딩된 모델 객체를 반환합니다.
    return _CHURN_MODEL


def make_input_dataframe(values: Dict[str, object]) -> pd.DataFrame:
    """Streamlit 입력값 딕셔너리를 모델 입력용 DataFrame으로 변환합니다."""

    # 모델은 2차원 표 형태 입력을 기대하므로 단일 고객 입력도 리스트로 감싸 DataFrame을 생성합니다.
    return pd.DataFrame([values])


def predict_churn(values: Dict[str, object]) -> Dict[str, object]:
    """고객 정보 입력값을 받아 이탈 여부와 확률을 예측합니다."""

    # 저장된 사전 학습 모델을 불러옵니다.
    model = load_churn_model()

    # 사용자 입력 딕셔너리를 모델 입력 DataFrame으로 변환합니다.
    input_df = make_input_dataframe(values)

    # predict_proba는 [잔류 확률, 이탈 확률] 형태의 확률 배열을 반환합니다.
    probabilities = model.predict_proba(input_df)[0]

    # 이탈 클래스는 학습 데이터에서 1로 인코딩했으므로 두 번째 확률을 사용합니다.
    churn_probability = float(probabilities[1])

    # 확률이 0.5 이상이면 이탈 위험 고객으로 분류합니다.
    prediction = int(churn_probability >= 0.5)

    # 화면 표시용 한글 라벨을 만듭니다.
    label = "이탈 위험" if prediction == 1 else "잔류 가능성 높음"

    # 예측 결과를 딕셔너리로 반환합니다.
    return {
        "prediction": prediction,
        "label": label,
        "churn_probability": churn_probability,
        "retention_probability": float(probabilities[0]),
    }

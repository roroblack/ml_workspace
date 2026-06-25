# -*- coding: utf-8 -*-
"""
학습된 모델로 새 고객의 이탈 확률 예측 (배포/추론 예시)
========================================================
저장된 전처리기(preprocessor.joblib) + 최적 모델(GradientBoosting.joblib)을
불러와, 원본 형식의 새 고객 데이터를 그대로 넣어 이탈 확률을 출력합니다.
"""
import os, sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import pandas as pd
import joblib

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(HERE, "models")

# 원본 컬럼 형식(식별자 제외)으로 새 고객 입력
NEW_CUSTOMERS = pd.DataFrame([
    {"CreditScore": 600, "Geography": "France", "Gender": "Female", "Age": 42,
     "Tenure": 2, "Balance": 0.0, "Num Of Products": 1,
     "Has Credit Card": 1, "Is Active Member": 1, "Estimated Salary": 101348.88},
    {"CreditScore": 700, "Geography": "Germany", "Gender": "Male", "Age": 58,
     "Tenure": 1, "Balance": 130000.0, "Num Of Products": 1,
     "Has Credit Card": 1, "Is Active Member": 0, "Estimated Salary": 90000.0},
])

THRESHOLD = 0.5  # 이탈 방지 캠페인 대상을 넓히려면 0.3 등으로 낮출 수 있음


def main():
    pre = joblib.load(os.path.join(MODELS, "preprocessor.joblib"))
    model = joblib.load(os.path.join(MODELS, "GradientBoosting.joblib"))

    X = pre.transform(NEW_CUSTOMERS)
    proba = model.predict_proba(X)[:, 1]

    for i, p in enumerate(proba):
        verdict = "이탈 위험" if p >= THRESHOLD else "유지 예상"
        print(f"고객 {i+1}: 이탈 확률 {p*100:.1f}% → {verdict}")


if __name__ == "__main__":
    main()

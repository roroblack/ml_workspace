# -*- coding: utf-8 -*-
"""
가입 고객 이탈 예측 — 데이터 전처리 파이프라인
=================================================
원본(data/raw/Churn_Modelling.csv)을 받아
  1) 식별자 제거
  2) 결측치/이상치 점검
  3) 범주형 인코딩 + 수치형 스케일링 (학습셋에 fit, 테스트셋에 transform)
  4) 층화(stratified) 학습/테스트 분할
을 수행하고, 전처리 결과 CSV와 전처리 요약(JSON)을 저장합니다.

산출물:
  - data/processed/train.csv, test.csv  (인코딩+스케일링 완료)
  - models/preprocessor.joblib          (배포 시 동일 전처리 재현용)
  - outputs/preprocess_summary.json      (전처리 결과서 작성 근거)
"""
import os, json, sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "data", "raw", "Churn_Modelling.csv")
PROC = os.path.join(HERE, "data", "processed")
MODELS = os.path.join(HERE, "models")
OUT = os.path.join(HERE, "outputs")
for d in (PROC, MODELS, OUT):
    os.makedirs(d, exist_ok=True)

TARGET = "Churn"
DROP_COLS = ["CustomerId", "Surname"]          # 식별자 — 예측에 무의미
CATEGORICAL = ["Geography", "Gender"]          # 범주형
SEED = 42


def main():
    df = pd.read_csv(RAW)
    summary = {"raw_shape": list(df.shape), "columns": list(df.columns)}

    # --- 1) 식별자 제거 ---
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # --- 2) 결측치/이상치 점검 ---
    na_counts = df.isna().sum()
    summary["missing_values"] = {k: int(v) for k, v in na_counts.items() if v > 0}
    summary["n_missing_total"] = int(na_counts.sum())

    # 타깃 분포 (클래스 불균형 확인)
    vc = df[TARGET].value_counts().sort_index()
    summary["target_distribution"] = {int(k): int(v) for k, v in vc.items()}
    summary["churn_rate"] = round(float(df[TARGET].mean()), 4)

    # 수치형 컬럼 IQR 이상치 비율(참고용 — 본 파이프라인은 트리/스케일러로 흡수하므로 제거하지 않음)
    numeric = [c for c in df.columns if c != TARGET and c not in CATEGORICAL]
    iqr_report = {}
    for c in numeric:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[c] < lo) | (df[c] > hi)).sum())
        iqr_report[c] = {"outliers": n_out, "pct": round(100 * n_out / len(df), 2)}
    summary["iqr_outliers"] = iqr_report
    summary["numeric_features"] = numeric
    summary["categorical_features"] = CATEGORICAL

    # --- 3) 분할 (층화) ---
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    summary["train_shape"] = list(X_train.shape)
    summary["test_shape"] = list(X_test.shape)

    # --- 4) 인코딩 + 스케일링 (학습셋에 fit) ---
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    Xtr = pre.fit_transform(X_train)
    Xte = pre.transform(X_test)
    feat_names = list(pre.get_feature_names_out())
    summary["n_features_after_encoding"] = len(feat_names)
    summary["encoded_feature_names"] = feat_names

    # --- 저장 ---
    pd.DataFrame(Xtr, columns=feat_names).assign(Churn=y_train.values).to_csv(
        os.path.join(PROC, "train.csv"), index=False)
    pd.DataFrame(Xte, columns=feat_names).assign(Churn=y_test.values).to_csv(
        os.path.join(PROC, "test.csv"), index=False)
    joblib.dump(pre, os.path.join(MODELS, "preprocessor.joblib"))

    with open(os.path.join(OUT, "preprocess_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("전처리 완료")
    print(f"  원본 {summary['raw_shape']} → 인코딩 후 피처 {summary['n_features_after_encoding']}개")
    print(f"  결측치 총 {summary['n_missing_total']}개")
    print(f"  이탈률 {summary['churn_rate']*100:.2f}% (불균형)")
    print(f"  학습 {summary['train_shape']} / 테스트 {summary['test_shape']}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""v4 호환·실시간 적합성 테스트 (사용자 #4 우려 검증).
① 실시간 안전: '최근 이벤트 스트림' → 10피처 재계산 → prep_{Model}.joblib 로 즉시 예측되는가?
② 10컬럼 호환: 정본 10피처를 이름으로 읽어 예측 + 추가 컬럼이 섞여도 깨지지 않는가?
실행: python preprocessing_project/v4_model_prep/src/realtime_compat_test.py [Model]
"""
import os, sys, json
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, joblib

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
D = os.path.join(HERE, "sample_project", "data", "processed_5m")
OUT = os.path.join(HERE, "preprocessing_project", "v4_model_prep", "output")
FEAT = ["recency_days", "tenure_days", "ndays", "n_events", "n_view", "n_cart",
        "n_remove_from_cart", "n_purchase", "avg_price", "purch_amt"]
EVENT_TYPES = ["view", "cart", "remove_from_cart", "purchase"]


def compute_features(events, now):
    """최근 이벤트 리스트(dict: event_type,price,event_time) → 정본 10피처. (실시간 backend가 그대로 호출 가능)"""
    df = pd.DataFrame(events)
    df["event_time"] = pd.to_datetime(df["event_time"])
    last = df["event_time"].max(); first = df["event_time"].min()
    days = df["event_time"].dt.normalize()
    vc = df["event_type"].value_counts()
    f = {
        "recency_days": (now - last).total_seconds() / 86400,
        "tenure_days": (last - first).total_seconds() / 86400,
        "ndays": int(days.nunique()),
        "n_events": int(len(df)),
        "n_view": int(vc.get("view", 0)), "n_cart": int(vc.get("cart", 0)),
        "n_remove_from_cart": int(vc.get("remove_from_cart", 0)), "n_purchase": int(vc.get("purchase", 0)),
        "avg_price": float(df["price"].mean()),
        "purch_amt": float(df.loc[df.event_type == "purchase", "price"].sum()),
    }
    return np.array([[f[c] for c in FEAT]], dtype=float)


def predict(artifact, X):
    a = X.copy()
    prep = artifact["prep"]
    cidx = [FEAT.index(c) for c in ["ndays", "n_events", "n_view", "n_cart", "n_remove_from_cart", "n_purchase", "purch_amt"]]
    if prep["log_counts"]:
        a[:, cidx] = np.log1p(a[:, cidx])
    if artifact.get("scaler") is not None:
        a = artifact["scaler"].transform(a)
    p = artifact["calibrator"].predict_proba(a)[:, 1]
    thr = artifact["threshold"]
    return float(p[0]), ("high" if p[0] >= 0.65 else "medium" if p[0] >= 0.35 else "low"), int(p[0] >= thr)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "DecisionTree"
    art = joblib.load(os.path.join(OUT, name, f"prep_{name}.joblib"))   # 모델별 폴더
    ok = True

    # ① 실시간 안전: 가짜 최근 이벤트 스트림
    now = pd.Timestamp("2020-02-21 00:00:00")
    events = [
        {"event_type": "view", "price": 5.0, "event_time": "2020-02-10 10:00:00"},
        {"event_type": "view", "price": 6.2, "event_time": "2020-02-10 10:03:00"},
        {"event_type": "cart", "price": 6.2, "event_time": "2020-02-10 10:05:00"},
        {"event_type": "purchase", "price": 6.2, "event_time": "2020-02-11 09:00:00"},
    ]
    X = compute_features(events, now)
    prob, risk, churn = predict(art, X)
    rt_ok = isinstance(prob, float) and 0 <= prob <= 1 and X.shape == (1, 10)
    ok &= rt_ok
    print(f"① 실시간: 이벤트4건→10피처 재계산→예측 prob={prob:.3f} risk={risk} churn={churn}  [{'OK' if rt_ok else 'FAIL'}]")
    print(f"   feature_order(정본10) 일치: {art['feature_order'] == FEAT}")

    # ② 10컬럼 호환: 정본 parquet 이름 기반 + 추가컬럼 섞어도 무영향
    te = pd.read_parquet(os.path.join(D, "test_cohort_tabular.parquet")).head(200).copy()
    te["EXTRA_new_feature_v2"] = 1.23                      # 미래 추가 컬럼(예: last_cat_id) 섞기
    Xc = np.nan_to_num(te[FEAT].values.astype(float))     # 이름으로 10개만 선택 → 추가컬럼 무시
    cidx = [FEAT.index(c) for c in ["ndays", "n_events", "n_view", "n_cart", "n_remove_from_cart", "n_purchase", "purch_amt"]]
    a = Xc.copy()
    if art["prep"]["log_counts"]:
        a[:, cidx] = np.log1p(a[:, cidx])
    if art.get("scaler") is not None:
        a = art["scaler"].transform(a)
    p = art["calibrator"].predict_proba(a)[:, 1]
    compat_ok = len(p) == 200 and np.all((p >= 0) & (p <= 1))
    ok &= compat_ok
    print(f"② 10컬럼 호환: 추가컬럼(EXTRA_*) 섞인 parquet에서 이름기반 10피처 예측 {len(p)}건  [{'OK' if compat_ok else 'FAIL'}]")
    print(f"   → 추가형 컬럼은 이름 선택으로 무시됨(기존 모델 깨지지 않음). 새 모델만 추가컬럼 사용.")

    print("=" * 56)
    print(f"호환·실시간 테스트: {'PASS ✅' if ok else 'FAIL ❌'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

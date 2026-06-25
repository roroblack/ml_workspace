# -*- coding: utf-8 -*-
"""
REES46 화장품 이벤트 데이터(1개월) — 시계열 라벨링 + 피처 생성
==============================================================
입력: data/raw/ 안의 월별 CSV 1개 (예: 2019-Nov.csv)
  컬럼: event_time, event_type(view/cart/remove_from_cart/purchase),
        product_id, category_id, category_code, brand, price, user_id, user_session

churn 정의: REES46은 view 이벤트가 있어 "활동 기반(접속형)" 정의가 가능.
  관찰기간(월의 앞 70%)으로 피처 생성 → 결과기간(뒤 30%)에 "어떤 이벤트도 없음 = 이탈(1)".
  (누수 방지: 피처는 관찰기간만)
산출물: data/tabular.csv, data/seq.npz, data/split.npz, outputs/label_summary.json
"""
import os, sys, glob, json
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "outputs")
os.makedirs(DATA, exist_ok=True); os.makedirs(OUT, exist_ok=True)

SEED = 42
OBS_FRAC = 0.7          # 관찰/결과 분할 비율 (앞 70% 관찰)
USER_CAP = 60000        # CPU 고려 사용자 표본 상한 (모집단이 크면 표본추출)
EVENTS = ["view", "cart", "purchase"]


def find_csv():
    files = sorted(glob.glob(os.path.join(RAW, "*.csv")) + glob.glob(os.path.join(RAW, "*.zip")))
    if not files:
        raise SystemExit(f"[안내] data/raw/ 에 REES46 월별 CSV(또는 .zip) 1개를 넣어주세요.\n경로: {RAW}")
    print("사용 파일:", os.path.basename(files[0]))
    return files[0]   # pandas가 .zip 확장자를 자동 압축 해제


def main():
    path = find_csv()
    usecols = ["event_time", "event_type", "product_id", "brand", "price", "user_id", "user_session"]
    df = pd.read_csv(path, usecols=usecols)
    df["event_time"] = pd.to_datetime(df["event_time"].str.replace(" UTC", "", regex=False),
                                      errors="coerce", utc=False)
    df = df.dropna(subset=["event_time", "user_id"])
    df["date"] = df["event_time"].dt.normalize()

    t0, t1 = df["event_time"].min(), df["event_time"].max()
    cut = t0 + (t1 - t0) * OBS_FRAC
    obs = df[df["event_time"] < cut].copy()
    out = df[df["event_time"] >= cut].copy()

    pop = obs["user_id"].unique()
    out_users = set(out["user_id"].unique())
    # 표본추출 (모집단이 크면)
    rng = np.random.RandomState(SEED)
    if len(pop) > USER_CAP:
        pop = rng.choice(pop, USER_CAP, replace=False)
    pop = np.sort(pop)
    obs = obs[obs["user_id"].isin(pop)].copy()
    churn = np.array([0 if u in out_users else 1 for u in pop], dtype=int)

    # ---- 정형 피처 (관찰기간) ----
    g = obs.groupby("user_id")
    f = pd.DataFrame(index=pop); f.index.name = "user_id"
    last = g["event_time"].max(); first = g["event_time"].min()
    f["recency_days"] = (cut - last).dt.total_seconds() / 86400
    f["active_days"] = g["date"].nunique()
    f["n_sessions"] = g["user_session"].nunique()
    f["n_events"] = g.size()
    for et in EVENTS:
        f[f"n_{et}"] = obs[obs["event_type"] == et].groupby("user_id").size().reindex(pop).fillna(0).values
    f["n_unique_products"] = g["product_id"].nunique()
    f["n_unique_brands"] = g["brand"].nunique()
    f["avg_price"] = g["price"].mean()
    f["view2purchase"] = f["n_purchase"] / f["n_view"].replace(0, np.nan)
    f["view2purchase"] = f["view2purchase"].fillna(0)
    f["tenure_days"] = (last - first).dt.total_seconds() / 86400
    f = f.fillna(0)
    f["churn"] = churn
    f = f.reset_index()
    f.to_csv(os.path.join(DATA, "tabular.csv"), index=False)

    # ---- 일별 시퀀스 [N, D, 3] : view/cart/purchase 일별 횟수 ----
    days = pd.date_range(t0.normalize(), (cut - pd.Timedelta(seconds=1)).normalize(), freq="D")
    didx = {d: i for i, d in enumerate(days)}
    uidx = {u: i for i, u in enumerate(pop)}
    X = np.zeros((len(pop), len(days), len(EVENTS)), dtype=np.float32)
    daily = obs.groupby(["user_id", "date", "event_type"]).size()
    for (u, d, et), c in daily.items():
        if et in EVENTS and d in didx and u in uidx:
            X[uidx[u], didx[d], EVENTS.index(et)] = c
    np.savez(os.path.join(DATA, "seq.npz"), X=X, y=churn, user_id=pop)

    # 공통 분할 저장
    idx = np.arange(len(churn))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=churn)
    np.savez(os.path.join(DATA, "split.npz"), tr=tr, te=te)

    summary = {
        "file": os.path.basename(path), "rows": int(len(df)),
        "date_range": [str(t0), str(t1)], "cut": str(cut),
        "obs_frac": OBS_FRAC, "population": int(len(pop)),
        "churn_count": int(churn.sum()), "churn_rate": round(float(churn.mean()), 4),
        "tabular_features": [c for c in f.columns if c not in ("user_id", "churn")],
        "seq_shape": list(X.shape), "seq_feature_order": EVENTS,
        "n_days": len(days),
    }
    json.dump(summary, open(os.path.join(OUT, "label_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"완료 | 모집단 {len(pop)} | 이탈률 {summary['churn_rate']*100:.2f}% | "
          f"정형 {len(summary['tabular_features'])}피처 | 시퀀스 {X.shape} ({len(days)}일)")


if __name__ == "__main__":
    main()

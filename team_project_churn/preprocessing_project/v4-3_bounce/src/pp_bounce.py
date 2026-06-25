# -*- coding: utf-8 -*-
"""v4-3 바운스 전처리 — 이벤트레벨 세션바운스(churn30) 예측 데이터셋.

근거: v4 pp_session_bounce.py. 각 이벤트의 '진행 중 세션 상태'(8피처) → churn30(이 행동 후 30분 무이벤트=1) 예측(이진).
v4(집계 churn)와 별개 트랙. 모델팀이 부스트 등으로 학습하도록 train/test parquet 산출.

시간분할(누수차단): train=2019-Nov 표본유저 / test=2019-Dec 표본유저 (월 분리).
산출: output/{train_bounce,test_bounce}.parquet (8피처 + churn30 + user_id + user_session) + meta.
실행: python pp_bounce.py
"""
import os, sys, json, zipfile, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(HERE, "src")
OUT = os.path.join(HERE, "preprocessing_project", "v4-3_bounce", "output"); os.makedirs(OUT, exist_ok=True)
GAP = 1800   # 30분(초). 이 행동 후 다음 이벤트까지 gap>GAP(또는 없음) = 바운스
FEAT = ["step", "dt_prev_log", "n_view_sf", "n_cart_sf", "n_purchase_sf", "price_log", "price_mean_sf_log", "is_first"]
TRAIN = ("2019-Nov", 60000)
TEST = ("2019-Dec", 30000)


def build(month, user_cap, tag):
    t0 = time.time()
    dt = {"event_type": "category", "price": "float32", "user_id": "int64", "user_session": "category"}
    with zipfile.ZipFile(os.path.join(SRC, f"{month}.csv.zip")) as zf:
        df = pd.read_csv(zf.open(f"{month}.csv"), usecols=list(dt) + ["event_time"], dtype=dt)
    df["event_time"] = pd.to_datetime(df["event_time"].str.slice(0, 19), format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["event_time", "user_id"]).sort_values(["user_id", "event_time"])
    users = df["user_id"].drop_duplicates()
    users = users.sample(min(user_cap, len(users)), random_state=42)
    df = df[df.user_id.isin(users)].reset_index(drop=True)
    df["t"] = df["event_time"].astype("int64") // 10**9
    g = df.groupby("user_id", sort=False)
    df["dt_next"] = g["t"].shift(-1) - df["t"]
    df["churn30"] = ((df["dt_next"] > GAP) | (df["dt_next"].isna())).astype("int8")   # 라벨(바운스)
    gs = df.groupby("user_session", sort=False)
    df["step"] = gs.cumcount() + 1
    df["is_first"] = (df["step"] == 1).astype("int8")
    df["dt_prev"] = (df["t"] - gs["t"].shift(1)).fillna(0).clip(0, GAP)
    df["dt_prev_log"] = np.log1p(df["dt_prev"])
    et = df["event_type"].astype(str)
    df["n_view_sf"] = (et == "view").groupby(df.user_session).cumsum()
    df["n_cart_sf"] = (et == "cart").groupby(df.user_session).cumsum()
    df["n_purchase_sf"] = (et == "purchase").groupby(df.user_session).cumsum()
    pc = df["price"].clip(lower=0).fillna(0)
    df["price_log"] = np.log1p(pc)
    df["price_mean_sf_log"] = np.log1p(pc.groupby(df.user_session).cumsum() / df["step"].clip(lower=1))
    out = df[["user_id", "user_session"] + FEAT + ["churn30"]].copy()
    path = os.path.join(OUT, f"{tag}_bounce.parquet"); out.to_parquet(path, index=False)
    meta = {"block": tag, "month": month, "rows": int(len(out)), "users": int(out.user_id.nunique()),
            "bounce_rate": round(float(out.churn30.mean()), 4), "n_features": len(FEAT), "features": FEAT,
            "label": "churn30 (이 이벤트 후 30분 내 무이벤트=1)", "GAP_sec": GAP}
    json.dump(meta, open(os.path.join(OUT, f"meta_bounce_{tag}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[v4-3:{tag}] {len(out):,}행 / {meta['users']:,}유저 | 바운스율 {meta['bounce_rate']*100:.1f}% | {len(FEAT)}피처 | {time.time()-t0:.0f}s", flush=True)
    return out


def main():
    tr = build(*TRAIN, "train")
    build(*TEST, "test")
    with open(os.path.join(OUT, "first30_bounce.txt"), "w", encoding="utf-8") as f:
        f.write(f"# v4-3 바운스(churn30) train 앞30행 | {len(FEAT)}피처 + churn30\n\n")
        f.write(tr.head(30).to_string(index=False))
    print(f"[v4-3] 완료 → {OUT}", flush=True)


if __name__ == "__main__":
    main()

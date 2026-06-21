# -*- coding: utf-8 -*-
"""SB — 바운스/세션 이탈 실시간 트랙. 이벤트별 '진행 중 세션 상태' → churn30(이 행동 후 30분 무이벤트=이탈) 예측.
경량(LogReg)으로 빠르게 학습 + 대시보드 리플레이용 샘플 세션 저장. (집계 churn과 별개 트랙)
산출: output/session_bounce/{model.joblib, meta.json, sample_sessions.json}
실행: python preprocessing_project/v4_model_prep/src/pp_session_bounce.py [zip]
"""
import os, sys, json, zipfile, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(HERE, "src")
OUT = os.path.join(HERE, "preprocessing_project", "v4_model_prep", "output", "session_bounce"); os.makedirs(OUT, exist_ok=True)
DEFAULT_ZIP = os.path.join(SRC, "2019-Nov.csv.zip")
USER_CAP = 4000
GAP = 1800            # 30분(초) — 이 행동 후 다음 이벤트까지 GAP 초과 = 세션 이탈/바운스
FEAT = ["step", "dt_prev_log", "n_view_sf", "n_cart_sf", "n_purchase_sf", "price_log", "price_mean_sf_log", "is_first"]


def main(zip_path=None):
    zip_path = zip_path or DEFAULT_ZIP
    t0 = time.time()
    dt = {"event_type": "category", "price": "float32", "user_id": "int64", "user_session": "category", "category_id": "int64", "brand": "category"}
    with zipfile.ZipFile(zip_path) as zf:
        df = pd.read_csv(zf.open(os.path.basename(zip_path)[:-4]), usecols=list(dt) + ["event_time"], dtype=dt)
    df["event_time"] = pd.to_datetime(df["event_time"].str.slice(0, 19), format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["event_time", "user_id"]).sort_values(["user_id", "event_time"])
    users = df["user_id"].drop_duplicates()
    users = users.sample(min(USER_CAP, len(users)), random_state=42)
    df = df[df.user_id.isin(users)].reset_index(drop=True)
    df["t"] = df["event_time"].astype("int64") // 10**9
    g = df.groupby("user_id", sort=False)
    df["dt_next"] = (g["t"].shift(-1) - df["t"])
    df["churn30"] = ((df["dt_next"] > GAP) | (df["dt_next"].isna())).astype(int)     # 라벨
    # 세션 내 진행 상태(running)
    gs = df.groupby("user_session", sort=False)
    df["step"] = gs.cumcount() + 1
    df["is_first"] = (df["step"] == 1).astype(int)
    df["dt_prev"] = (df["t"] - gs["t"].shift(1)).fillna(0).clip(0, GAP)
    df["dt_prev_log"] = np.log1p(df["dt_prev"])
    et = df["event_type"].astype(str)
    df["n_view_sf"] = (et == "view").groupby(df.user_session).cumsum()
    df["n_cart_sf"] = (et == "cart").groupby(df.user_session).cumsum()
    df["n_purchase_sf"] = (et == "purchase").groupby(df.user_session).cumsum()
    pc = df["price"].clip(lower=0).fillna(0)
    df["price_log"] = np.log1p(pc)
    # 세션 내 누적 평균가(벡터화: cumsum/step) — expanding.apply 파이썬 루프 회피
    df["price_mean_sf_log"] = np.log1p(pc.groupby(df.user_session).cumsum() / df["step"].clip(lower=1))

    X = np.nan_to_num(df[FEAT].values.astype(float)); y = df["churn30"].values
    # 시간 분할(앞 80% train)
    cut = int(len(df) * 0.8)
    pipe = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    pipe.fit(X[:cut], y[:cut])
    auc = roc_auc_score(y[cut:], pipe.predict_proba(X[cut:])[:, 1])
    joblib.dump({"pipeline": pipe, "feat": FEAT, "gap_sec": GAP}, os.path.join(OUT, "model.joblib"))

    # 리플레이용 샘플 세션(바운스1 / 장바구니이탈 / 구매) — 이벤트별 누적피처 + 실시간 bounce 확률
    df["bounce_prob"] = pipe.predict_proba(X)[:, 1]
    sess = df.assign(is_p=(et == "purchase").values.astype(int), is_c=(et == "cart").values.astype(int)) \
             .groupby("user_session").agg(n=("step", "max"), has_p=("is_p", "max"), has_c=("is_c", "max"))
    pick = lambda mask: sess[mask].index.tolist()[:1]
    bounce_s = pick(sess.n == 1)
    eng_s = pick((sess.has_p == 1) & sess.n.between(5, 9))
    cart_s = pick((sess.has_p == 0) & (sess.has_c == 1) & sess.n.between(4, 8))
    samples = {}
    for tag, ss in [("BOUNCE", bounce_s), ("CART_ABANDON", cart_s), ("PURCHASE", eng_s)]:
        if not ss: continue
        s = df[df.user_session == ss[0]]
        samples[tag] = [{"step": int(r.step), "event": str(r.event_type), "brand": (None if pd.isna(r.brand) else str(r.brand)),
                         "price": round(float(r.price), 2), "dt_prev_s": int(r.dt_prev),
                         "bounce_prob": round(float(r.bounce_prob), 3), "churn30": int(r.churn30)}
                        for r in s.itertuples()]
    json.dump(samples, open(os.path.join(OUT, "sample_sessions.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump({"auc": round(float(auc), 4), "label": f"churn30(이 행동 후 {GAP//60}분 무활동=이탈)", "feat": FEAT,
               "n_events": int(len(df)), "churn30_rate": round(float(y.mean()), 4), "model": "LogisticRegression(balanced)",
               "note": "집계 churn(7일)과 별개 세션/바운스 트랙. 경량 데모. 풀데이터·GRU는 발표 후."},
              open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[SB] AUC {auc:.4f} | churn30율 {y.mean()*100:.1f}% | 이벤트 {len(df):,} | 샘플 {list(samples)} | {time.time()-t0:.0f}s")
    print(f"[SB] 저장 → {OUT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

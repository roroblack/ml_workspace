# -*- coding: utf-8 -*-
"""
REES46(2019-Nov) — 순서 의존 라벨 A/B/C 데이터셋 생성
=====================================================
A (세션 즉시 이탈): 세션 단위. 구매 없이 종료=이탈(1). 입력=세션 내 '구매 이전' 이벤트 순서.
B (단기 조기 이탈): 고객 단위. 관찰(11/1~11/23) → 결과(11/24~30) 활동 없음=이탈(1).
C (다음 활동까지 시간): 고객 단위. cutoff 이후 다음 이벤트까지 일수(회귀, 0~7, 미발생=7).
모두 누수 차단(A: 구매 이벤트 이후 제외 / B,C: 피처는 관찰기간만).
"""
import os, sys, json
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP = os.path.join(HERE, "..", "src", "2019-Nov.csv.zip")
DA, DBC = os.path.join(HERE, "data", "A"), os.path.join(HERE, "data", "BC")
os.makedirs(DA, exist_ok=True); os.makedirs(DBC, exist_ok=True)
SEED = 42
USECOLS = ["event_time", "event_type", "product_id", "price", "user_id", "user_session"]
EVENTS3 = ["view", "cart", "purchase"]


def load():
    d = pd.read_csv(ZIP, usecols=USECOLS)
    d["event_time"] = pd.to_datetime(d["event_time"], format="%Y-%m-%d %H:%M:%S UTC", errors="coerce")
    d = d.dropna(subset=["event_time", "user_id", "user_session"])
    d["user_id"] = d["user_id"].astype(np.int64)
    return d


def build_A(df, cap=120000, L=15):
    df = df.sort_values(["user_session", "event_time"])
    df["is_purchase"] = (df["event_type"] == "purchase").astype(np.int8)
    # 세션별 구매 누적 → 첫 구매 이전 이벤트만 keep
    df["cum_pur"] = df.groupby("user_session")["is_purchase"].cumsum()
    has_pur = df.groupby("user_session")["is_purchase"].max()           # 1=전환,0=이탈
    kept = df[df["cum_pur"] == 0].copy()                                # 구매 이전 이벤트
    # 세션 라벨 (이탈=1): 구매 없음
    lab = (1 - has_pur).rename("churn")
    # keep 비어있지 않은 세션만
    valid = kept["user_session"].unique()
    lab = lab.reindex(valid).dropna().astype(int)
    # 표본
    rng = np.random.RandomState(SEED)
    sess = lab.index.values
    if len(sess) > cap:
        sess = rng.choice(sess, cap, replace=False)
    sess = np.sort(sess)
    kept = kept[kept["user_session"].isin(sess)].copy()
    lab = lab.reindex(sess)
    sidx = {s: i for i, s in enumerate(sess)}

    # step features
    kept["gap"] = kept.groupby("user_session")["event_time"].diff().dt.total_seconds().fillna(0)
    kept["is_view"] = (kept["event_type"] == "view").astype(np.float32)
    kept["is_cart"] = (kept["event_type"] == "cart").astype(np.float32)
    kept["is_remove"] = (kept["event_type"] == "remove_from_cart").astype(np.float32)
    kept["lprice"] = np.log1p(kept["price"].clip(lower=0)).astype(np.float32)
    kept["lgap"] = np.log1p(kept["gap"].clip(lower=0)).astype(np.float32)
    # 마지막 L개만
    kept["rank"] = kept.groupby("user_session").cumcount()
    slen = kept.groupby("user_session")["rank"].transform("max") + 1
    kept["step"] = kept["rank"] - (slen - L).clip(lower=0)
    last = kept[kept["step"] >= 0]
    feats = ["is_view", "is_cart", "is_remove", "lprice", "lgap"]
    X = np.zeros((len(sess), L, len(feats)), dtype=np.float32)
    si = last["user_session"].map(sidx).values.astype(int)
    X[si, last["step"].values.astype(int), :] = last[feats].values

    # 정형 집계
    g = kept.groupby("user_session")
    tab = pd.DataFrame(index=sess); tab.index.name = "session"
    tab["n_events"] = g.size().reindex(sess).fillna(0)
    tab["n_view"] = kept[kept.is_view == 1].groupby("user_session").size().reindex(sess).fillna(0)
    tab["n_cart"] = kept[kept.is_cart == 1].groupby("user_session").size().reindex(sess).fillna(0)
    tab["n_remove"] = kept[kept.is_remove == 1].groupby("user_session").size().reindex(sess).fillna(0)
    tab["n_unique_products"] = g["product_id"].nunique().reindex(sess).fillna(0)
    tab["dur_sec"] = (g["event_time"].max() - g["event_time"].min()).dt.total_seconds().reindex(sess).fillna(0)
    tab["avg_lprice"] = g["lprice"].mean().reindex(sess).fillna(0)
    tab["has_cart"] = (tab["n_cart"] > 0).astype(int)
    tab["churn"] = lab.values
    tab.reset_index().to_csv(os.path.join(DA, "tabular.csv"), index=False)
    np.savez(os.path.join(DA, "seq.npz"), X=X)
    idx = np.arange(len(sess))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=lab.values)
    np.savez(os.path.join(DA, "split.npz"), tr=tr, te=te)
    print(f"[A] 세션 {len(sess)} | 이탈(미구매)률 {lab.mean()*100:.1f}% | seq {X.shape}")
    return {"unit": "session", "n": int(len(sess)), "pos_rate": round(float(lab.mean()), 4), "seq": list(X.shape)}


def build_BC(df, cap=60000, obs_end="2019-11-24", horizon=7):
    cut = pd.Timestamp(obs_end)
    start = pd.Timestamp("2019-11-01")
    obs = df[df["event_time"] < cut].copy()
    fut = df[(df["event_time"] >= cut)].copy()
    pop = obs["user_id"].unique()
    rng = np.random.RandomState(SEED)
    if len(pop) > cap:
        pop = rng.choice(pop, cap, replace=False)
    pop = np.sort(pop)
    obs = obs[obs["user_id"].isin(pop)].copy()
    obs["date"] = obs["event_time"].dt.normalize()

    # 타깃 B: 결과기간 활동 없음=1 ; C: 다음 이벤트까지 일수(없으면 horizon)
    fut_first = fut.groupby("user_id")["event_time"].min()
    ttne = ((fut_first - cut).dt.total_seconds() / 86400).reindex(pop)
    label_b = ttne.isna().astype(int).values            # 미래 이벤트 없음=이탈
    target_c = ttne.fillna(horizon).clip(0, horizon).astype(np.float32).values

    g = obs.groupby("user_id")
    f = pd.DataFrame(index=pop); f.index.name = "user_id"
    last = g["event_time"].max(); first = g["event_time"].min()
    f["recency_days"] = (cut - last).dt.total_seconds() / 86400
    f["tenure_days"] = (last - first).dt.total_seconds() / 86400
    f["active_days"] = g["date"].nunique()
    f["n_events"] = g.size()
    for et in EVENTS3:
        f[f"n_{et}"] = obs[obs.event_type == et].groupby("user_id").size().reindex(pop).fillna(0).values
    f["n_unique_products"] = g["product_id"].nunique()
    f["avg_price"] = g["price"].mean()
    f = f.fillna(0)
    f["label_b"] = label_b; f["target_c"] = target_c
    f.reset_index().to_csv(os.path.join(DBC, "tabular.csv"), index=False)

    # 일별 시퀀스 (관찰 23일) [N, D, 3]
    days = pd.date_range(start, cut - pd.Timedelta(days=1), freq="D")
    didx = {d: i for i, d in enumerate(days)}; uidx = {u: i for i, u in enumerate(pop)}
    X = np.zeros((len(pop), len(days), len(EVENTS3)), dtype=np.float32)
    daily = obs.groupby(["user_id", "date", "event_type"]).size()
    for (u, d, et), c in daily.items():
        if et in EVENTS3 and d in didx and u in uidx:
            X[uidx[u], didx[d], EVENTS3.index(et)] = c
    np.savez(os.path.join(DBC, "seq.npz"), X=X)
    idx = np.arange(len(pop))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=label_b)
    np.savez(os.path.join(DBC, "split.npz"), tr=tr, te=te)
    print(f"[B] 고객 {len(pop)} | 단기이탈률 {label_b.mean()*100:.1f}% | seq {X.shape}")
    print(f"[C] 회귀 target(다음활동 일수) 평균 {target_c.mean():.2f}, 미발생비율 {label_b.mean()*100:.1f}%")
    return {"unit": "user", "n": int(len(pop)), "b_pos_rate": round(float(label_b.mean()), 4),
            "c_mean": round(float(target_c.mean()), 3), "seq": list(X.shape)}


def main():
    print("2019-Nov 로딩...")
    df = load()
    print(f"  이벤트 {len(df):,}")
    a = build_A(df)
    bc = build_BC(df)
    json.dump({"A": a, "BC": bc}, open(os.path.join(HERE, "outputs", "prep_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("완료")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Mobile Game churn — base/A/B/C 라벨 데이터셋 생성 (REES46와 동일 비교 환경)
========================================================================
데이터: level_seq.csv.zip(219만 플레이), train/dev.csv(공식 이탈 라벨), 기간 2020-02-01~04.
컬럼: user_id, level_id, f_success, f_duration, f_reststep, f_help, time

base: 공식 이탈 라벨(train+dev) — 고객 단위. 피처=전체 플레이 집계, 시퀀스=레벨플레이 순서.
A   : 세션(30분 gap) 단위. 라벨=이 세션 이후 플레이 없음(마지막 세션=즉시이탈). 순서형.
B   : 고객 단위. 관찰(2/1~2/3) → 결과(2/4)에 플레이 없음=단기이탈. recency형.
C   : 고객 단위. cutoff(2/4 00:00) 이후 다음 플레이까지 시간(시간 단위, 회귀, 미발생=24).
스텝 피처(공통): [level_norm, f_success, log1p(duration), f_reststep, f_help, log1p(gap_sec)]
"""
import os, sys, io, zipfile, json
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src", "mobile_game")
D = {k: os.path.join(HERE, "data", k) for k in ("base", "A", "BC")}
for p in D.values(): os.makedirs(p, exist_ok=True)
SEED = 42
LU, LS = 50, 30          # 유저/세션 시퀀스 길이
NLV = 1509
CUT = pd.Timestamp("2020-02-04 00:00:00")
STEP_FEATS = ["level_norm", "f_success", "ldur", "f_reststep", "f_help", "lgap"]


def load():
    z = zipfile.ZipFile(os.path.join(SRC, "level_seq.csv.zip"))
    seq = pd.read_csv(io.BytesIO(z.read("level_seq.csv")), sep="\t")
    seq["time"] = pd.to_datetime(seq["time"])
    seq = seq.sort_values(["user_id", "time"]).reset_index(drop=True)
    # 스텝 피처
    seq["level_norm"] = seq["level_id"] / NLV
    seq["ldur"] = np.log1p(seq["f_duration"].clip(lower=0))
    gap = seq.groupby("user_id")["time"].diff().dt.total_seconds().fillna(0)
    seq["lgap"] = np.log1p(gap.clip(lower=0))
    seq["gap_s"] = gap
    tr = pd.read_csv(os.path.join(SRC, "train.csv"), sep="\t")
    dv = pd.read_csv(os.path.join(SRC, "dev.csv"), sep="\t")
    labels = pd.concat([tr, dv], ignore_index=True).drop_duplicates("user_id").set_index("user_id")["label"]
    return seq, labels


def seq_array(df, key, order, L, feats=STEP_FEATS):
    """key별 마지막 L스텝 시퀀스 텐서 [N,L,F]. order=정렬된 df, key2idx 매핑 반환."""
    keys = order
    kidx = {k: i for i, k in enumerate(keys)}
    df = df.copy()
    df["rank"] = df.groupby(key).cumcount()
    ln = df.groupby(key)["rank"].transform("max") + 1
    df["step"] = df["rank"] - (ln - L).clip(lower=0)
    last = df[df["step"] >= 0]
    X = np.zeros((len(keys), L, len(feats)), dtype=np.float32)
    ki = last[key].map(kidx).values.astype(int)
    X[ki, last["step"].values.astype(int), :] = last[feats].values
    return X


def agg_tabular(df, key, keys):
    g = df.groupby(key)
    t = pd.DataFrame(index=keys)
    t["n_plays"] = g.size().reindex(keys).fillna(0)
    t["success_rate"] = g["f_success"].mean().reindex(keys).fillna(0)
    t["avg_duration"] = g["f_duration"].mean().reindex(keys).fillna(0)
    t["max_level"] = g["level_id"].max().reindex(keys).fillna(0)
    t["n_help"] = g["f_help"].sum().reindex(keys).fillna(0)
    t["avg_reststep"] = g["f_reststep"].mean().reindex(keys).fillna(0)
    t["span_hours"] = (g["time"].max() - g["time"].min()).dt.total_seconds().reindex(keys).fillna(0) / 3600
    t["n_levels"] = g["level_id"].nunique().reindex(keys).fillna(0)
    t["fail_rate"] = 1 - t["success_rate"]
    return t


def save(sub, tab, X, y_col, strat):
    tab.to_csv(os.path.join(D[sub], "tabular.csv"), index=False)
    np.savez(os.path.join(D[sub], "seq.npz"), X=X)
    idx = np.arange(len(tab))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=strat)
    np.savez(os.path.join(D[sub], "split.npz"), tr=tr, te=te)


def main():
    print("로딩...", flush=True)
    seq, labels = load()
    print(f"  이벤트 {len(seq):,} / 라벨유저 {len(labels):,}", flush=True)

    # ---------- base (공식 라벨) ----------
    users = np.sort(labels.index.values)
    sb = seq[seq["user_id"].isin(set(users))]
    tab = agg_tabular(sb, "user_id", users).reset_index().rename(columns={"index": "user_id"})
    tab["churn"] = labels.reindex(users).values.astype(int)
    Xb = seq_array(sb, "user_id", users, LU)
    save("base", tab, Xb, "churn", tab["churn"].values)
    print(f"[base] 유저 {len(users)} | 이탈률 {tab.churn.mean()*100:.1f}% | seq {Xb.shape}", flush=True)

    # ---------- A (세션, 마지막세션=이탈) ----------
    newsess = (seq["gap_s"] > 1800) | (seq["user_id"] != seq["user_id"].shift())
    seq["sid"] = newsess.cumsum()
    last_sid = seq.groupby("user_id")["sid"].transform("max")
    seq["is_last_sess"] = (seq["sid"] == last_sid).astype(int)
    sess_label = seq.groupby("sid")["is_last_sess"].max()       # 1=마지막세션(이탈)
    sids = np.sort(seq["sid"].unique())
    # 표본 상한(세션 많으면)
    rng = np.random.RandomState(SEED)
    if len(sids) > 120000:
        sids = np.sort(rng.choice(sids, 120000, replace=False))
    sa = seq[seq["sid"].isin(set(sids))]
    tabA = agg_tabular(sa, "sid", sids).reset_index().rename(columns={"index": "session"})
    tabA["churn"] = sess_label.reindex(sids).values.astype(int)
    XA = seq_array(sa, "sid", sids, LS)
    save("A", tabA, XA, "churn", tabA["churn"].values)
    print(f"[A] 세션 {len(sids)} | 마지막세션(이탈)률 {tabA.churn.mean()*100:.1f}% | seq {XA.shape}", flush=True)

    # ---------- B / C (관찰 2/1~2/3 → 결과 2/4) ----------
    obs = seq[seq["time"] < CUT]
    fut = seq[seq["time"] >= CUT]
    pop = np.sort(obs["user_id"].unique())
    fut_first = fut.groupby("user_id")["time"].min()
    ttne_h = ((fut_first - CUT).dt.total_seconds() / 3600).reindex(pop)
    label_b = ttne_h.isna().astype(int).values
    target_c = ttne_h.fillna(24).clip(0, 24).astype(np.float32).values
    tabBC = agg_tabular(obs, "user_id", pop).reset_index().rename(columns={"index": "user_id"})
    tabBC["label_b"] = label_b; tabBC["target_c"] = target_c
    XBC = seq_array(obs, "user_id", pop, LU)
    save("BC", tabBC, XBC, "label_b", label_b)
    print(f"[B] 유저 {len(pop)} | 단기이탈률 {label_b.mean()*100:.1f}% | seq {XBC.shape}", flush=True)
    print(f"[C] 회귀 target(다음플레이 시간h) 평균 {target_c.mean():.2f}", flush=True)

    json.dump({"base_users": int(len(users)), "base_churn": round(float(tab.churn.mean()), 4),
               "A_sessions": int(len(sids)), "A_pos": round(float(tabA.churn.mean()), 4),
               "B_users": int(len(pop)), "B_pos": round(float(label_b.mean()), 4),
               "C_mean": round(float(target_c.mean()), 3),
               "seq_feats": STEP_FEATS, "LU": LU, "LS": LS},
              open(os.path.join(HERE, "outputs", "prep_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("완료")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""MAXWIN으로 버려진 윈도우 복원 — nextcat 데이터셋의 '전량(full-window)' 버전 저장.
추린 버전(nextcat_dataset.npz, 세션당 ≤4)은 그대로 두고, 같은 40k 세션의 *모든* 윈도우를
nextcat_dataset_fullwin.npz로 추가 저장. (버린 것 + 남은 것 둘 다 보존)"""
import os, sys, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import session_full_1734 as M
import nextcat_1735 as NC

PROC = NC.PROC; L = NC.L; SEED = NC.SEED; SESS_CAP = NC.SESS_CAP; NUMC = NC.NUMC


def main():
    df = M.load_full()
    slen = df.groupby("user_session", observed=True)["event_type"].transform("size")
    df = df[slen >= L + 1].copy()
    df["user_session"] = df["user_session"].cat.remove_unused_categories()
    sids = df["user_session"].cat.categories.to_numpy()
    rng = np.random.default_rng(SEED)                       # nextcat과 동일 시퀀스
    if len(sids) > SESS_CAP:
        keep = set(rng.choice(sids, SESS_CAP, replace=False))
        df = df[df["user_session"].isin(keep)].copy()
        df["user_session"] = df["user_session"].cat.remove_unused_categories()
    users = df["user_id"].unique(); rng.shuffle(users)
    tr_users = set(users[:int(len(users) * 0.75)])
    cats = sorted(df[df.user_id.isin(tr_users)]["category_id"].unique())
    cat2idx = {c: i + 1 for i, c in enumerate(cats)}; K = len(cats)

    W = int(len(df) - L * df.user_session.nunique())        # 전량 윈도우 수(미리 카운트→preallocate)
    print(f"세션 {df.user_session.nunique():,} | 전량 윈도우 {W:,} (추린 143,915 → 복원 {W-143915:,} 추가)", flush=True)
    Xn = np.zeros((W, L, NUMC), np.float32); Xc = np.zeros((W, L), np.int32)
    Y = np.zeros(W, np.int32); G = np.zeros(W, np.int64); TR = np.zeros(W, bool)
    p = 0
    for sid, s in df.groupby("user_session", sort=False, observed=True):
        et = s["event_type"].to_numpy(str); pr = s["price"].to_numpy(np.float32)
        ts = s["event_time"].values.astype("datetime64[s]").astype(np.int64)
        gap = np.concatenate([[0], np.diff(ts)]).astype(np.float32)
        cidx = np.array([cat2idx.get(int(c), 0) for c in s["category_id"].to_numpy()], np.int64)
        num = np.stack([(et == "view").astype(np.float32), (et == "cart").astype(np.float32),
                        (et == "remove_from_cart").astype(np.float32), (et == "purchase").astype(np.float32),
                        np.log1p(gap), np.log1p(pr)], 1).astype(np.float32)
        n = len(s); uid = int(s["user_id"].iloc[0]); is_tr = uid in tr_users
        for t in range(L, n):
            Xn[p] = num[t - L:t]; Xc[p] = cidx[t - L:t]; Y[p] = cidx[t]; G[p] = uid; TR[p] = is_tr; p += 1
    assert p == W

    out = os.path.join(PROC, "nextcat_dataset_fullwin.npz")
    np.savez_compressed(out, X_num=Xn, X_cat=Xc, y=Y, user_id=G, is_train=TR)
    meta = {"variant": "full-window (MAXWIN 미적용, 전량)", "L": L, "n_category": K,
            "n_windows_full": int(W), "n_windows_trimmed": 143915, "n_windows_discarded_recovered": int(W - 143915),
            "n_sessions": int(df.user_session.nunique()), "size_KB": round(os.path.getsize(out) / 1024, 1),
            "note": "추린본 nextcat_dataset.npz는 이 전량의 세션당 ≤4 무작위 부분집합"}
    json.dump(meta, open(os.path.join(PROC, "nextcat_fullwin_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"저장 {os.path.relpath(out, M.HERE)} ({meta['size_KB']:.0f} KB) | 전량 {W:,} (추린 143,915 + 복원 {W-143915:,})", flush=True)


if __name__ == "__main__":
    main()

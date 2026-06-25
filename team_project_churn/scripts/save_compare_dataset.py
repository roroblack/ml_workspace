# -*- coding: utf-8 -*-
"""17-3-3 전처리 데이터셋(세션 누적 윈도우)을 저장 + 원본 대비 포맷/압축 용량 비교.

- 전처리 로직은 forecast_session_level.py 재사용(동일 결과).
- 저장: sample_project/data/processed/session_level_dataset.npz (savez_compressed, 정식 보관본)
- 비교: 원본 events.csv vs 전처리 배열을 npy/npz/npz압축/float16압축/parquet로 측정.
"""
import os, sys, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
import forecast_session_level as F   # load_sessions/step_features/level_of/calibrate_g + 상수

SP = F.SP; OUTP = os.path.join(SP, "data", "processed")
RAW = F.RAW
L, H, C, RHO, OVERLOAD = F.L, F.H, F.C, F.RHO, F.OVERLOAD


def build():
    df = F.load_sessions()
    sess = list(df.groupby("session_id"))
    uid_of = {sid: s["user_id"].iloc[0] for sid, s in sess}
    np.random.seed(F.SEED)
    users = df["user_id"].unique(); np.random.shuffle(users)
    tr_u = set(users[:int(len(users) * 0.75)])
    feats = {sid: F.step_features(s) for sid, s in sess}
    cap = float(np.percentile(df[df.user_id.isin(tr_u)]["price"].clip(lower=0).replace(0, np.nan).dropna(), 90))
    g, _ = F.calibrate_g([feats[sid][1] for sid, _ in sess if uid_of[sid] in tr_u], cap)
    X, Y, G, isin_tr = [], [], [], []
    for sid, s in sess:
        feat, inc = feats[sid]; lv = F.level_of(inc, g, cap); n = len(s)
        for t in range(L, n - H + 1):
            X.append(feat[t - L:t]); Y.append(lv[t:t + H])
            G.append(int(uid_of[sid])); isin_tr.append(uid_of[sid] in tr_u)
    X = np.stack(X).astype(np.float32); Y = np.stack(Y).astype(np.float32)
    G = np.array(G, np.int64); tr = np.array(isin_tr, bool)
    meta = {"dataset": "REES46 2019-Nov 표본(sample_project)", "unit": "user_session",
            "L": L, "H": H, "C": C, "rho": RHO, "g": g, "cap_price_p90": round(cap, 2),
            "feature_cols": ["gap_log", "is_view", "is_cart", "is_remove", "is_purchase",
                             "price_log", "is_new_cat", "is_new_brand"],
            "n_sessions": len(sess), "n_windows": int(len(X)),
            "n_events_used": int(len(df)), "n_events_raw": int(sum(1 for _ in open(RAW, encoding="utf-8")) - 1)}
    return X, Y, G, tr, meta


def kb(p): return os.path.getsize(p) / 1024


def main():
    os.makedirs(OUTP, exist_ok=True)
    X, Y, G, tr, meta = build()
    print(f"[빌드] 윈도우 {len(X):,} | X{X.shape} Y{Y.shape} | float32")

    # === 정식 보관본 저장(압축) ===
    keep = os.path.join(OUTP, "session_level_dataset.npz")
    np.savez_compressed(keep, X=X, Y=Y, user_id=G, is_train=tr)
    json.dump(meta, open(os.path.join(OUTP, "session_level_meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[저장] {os.path.relpath(keep, F.HERE)}  ({kb(keep):.1f} KB) + session_level_meta.json")

    # === 포맷별 용량 비교 ===
    tmp = OUTP
    rows = []
    raw_kb = kb(RAW)
    # 1) npy 비압축 (X만)
    p = os.path.join(tmp, "_x.npy"); np.save(p, X); rows.append(("npy 비압축 (X만)", kb(p)));
    npy_x = kb(p)
    # 2) npz 비압축 (X+Y+meta arr)
    p2 = os.path.join(tmp, "_d.npz"); np.savez(p2, X=X, Y=Y, user_id=G, is_train=tr); rows.append(("npz 비압축 (savez)", kb(p2)))
    # 3) npz 압축 (정식 보관본)
    rows.append(("npz 압축 (savez_compressed) ★", kb(keep)))
    # 4) float16 + 압축
    p4 = os.path.join(tmp, "_d16.npz"); np.savez_compressed(p4, X=X.astype(np.float16), Y=Y.astype(np.float16), user_id=G, is_train=tr)
    rows.append(("float16 + npz압축", kb(p4)))
    # 5) parquet (long 형으로 X 평탄화는 비효율 → 참고용 X flatten csv.gz)
    try:
        dfX = pd.DataFrame(X.reshape(len(X), -1).astype(np.float32))
        pq = os.path.join(tmp, "_x.parquet"); dfX.to_parquet(pq, index=False); rows.append(("parquet(snappy, X flatten)", kb(pq)))
        gz = os.path.join(tmp, "_x.csv.gz"); dfX.to_csv(gz, index=False, compression="gzip"); rows.append(("csv+gzip (X flatten)", kb(gz)))
    except Exception as e:
        rows.append(("parquet", -1))

    print(f"\n[원본] events.csv = {raw_kb:.1f} KB ({meta['n_events_raw']:,}행)")
    print(f"[전처리] 세션필터(≥{L+H}) → {meta['n_events_used']:,}행 → 윈도우 {meta['n_windows']:,}개 [{L}×{C}] float32")
    print("\n[포맷별 용량]")
    base = next(s for l, s in rows if "비압축 (savez)" in l)
    for l, s in rows:
        if s < 0: print(f"  {l:32s}  불가"); continue
        ratio = f"(압축 {base/s:4.1f}x↓)" if s > 0 and "비압축 (savez)" not in l and s < base else ""
        print(f"  {l:32s} {s:9.1f} KB  {ratio}")

    # 정리
    for f in ["_x.npy", "_d.npz", "_d16.npz", "_x.parquet", "_x.csv.gz"]:
        fp = os.path.join(tmp, f)
        if os.path.exists(fp): os.remove(fp)

    # 비교표 JSON
    json.dump({"raw_events_csv_KB": round(raw_kb, 1), "n_events_raw": meta["n_events_raw"],
               "n_events_used": meta["n_events_used"], "n_windows": meta["n_windows"],
               "formats_KB": {l: round(s, 1) for l, s in rows if s > 0}},
              open(os.path.join(SP, "outputs", "realtime", "dataset_size_compare.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n저장 → outputs/realtime/dataset_size_compare.json")


if __name__ == "__main__":
    main()

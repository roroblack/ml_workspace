# -*- coding: utf-8 -*-
"""17-3-4. 풀 REES46(2019-Nov 전체) 세션 누적 level 예측 + 추천컬럼 보존 + 임베딩 개선.

17-3-3 대비 변경:
  ① 데이터: 8천 표본 → 2019-Nov 전체(463만 이벤트, 94만 세션)  ※USER_CAP 미적용
  ② 컬럼 보존: 추천 기능용으로 category_id/brand/product_id 유지(룰 §4) → 추천 피처표 별도 저장
  ③ 적분구간 확대: L=8→10, H=4→5 (세션 다수라 가능)
  ④ 개선안: 시퀀스 step에 category_id 임베딩 추가(트리는 못 쓰는 DL 전용 신호)

저장:
  data/processed_full/session_level_full.npz       (X증분, cat_idx, Y level, user_id, is_train) [압축]
  data/processed_full/recommend_user_interest.parquet (유저별 관심 카테고리/브랜드 — 추천용)
  data/processed_full/session_level_full_meta.json
  outputs/realtime/forecast_session_full.json + .png
"""
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # repo root(team_project_churn)
SP = os.path.join(HERE, "sample_project")
ZIP = os.path.join(HERE, "src", "2019-Nov.csv.zip")
PROC = os.path.join(SP, "data", "processed_full"); os.makedirs(PROC, exist_ok=True)
OUT = os.path.join(SP, "outputs", "realtime"); os.makedirs(OUT, exist_ok=True)
TYPES = ["view", "cart", "remove_from_cart", "purchase"]
SIGN = {"view": 0.0, "cart": 1.0, "remove_from_cart": -1.0, "purchase": 1.0}
WEIGHT = {"view": 1.0, "cart": 3.0, "remove_from_cart": -1.0, "purchase": 5.0}  # 추천 가중(룰 §4)
L, H = 10, 5
RHO, OVERLOAD = 0.85, 0.85
MAXWIN = 4               # 세션당 최대 윈도우(메모리/속도 캡)
EMB_DIM = 16
SEED = 42
C = 8
np.random.seed(SEED); torch.manual_seed(SEED)


def load_full():
    t = time.time()
    # dtype 명시로 메모리 절감(category/int32). product_id는 미사용이라 제외.
    dtypes = {"event_type": "category", "category_id": "int32", "brand": "category",
              "price": "float32", "user_id": "int64", "user_session": "category"}
    df = pd.read_csv(ZIP, usecols=list(dtypes) + ["event_time"], dtype=dtypes)
    df["event_time"] = pd.to_datetime(df["event_time"], format="%Y-%m-%d %H:%M:%S UTC", errors="coerce")
    df = df.dropna(subset=["event_time", "user_id", "user_session"])
    df["price"] = df["price"].clip(lower=0)
    if "unknown" not in df["brand"].cat.categories:
        df["brand"] = df["brand"].cat.add_categories("unknown")
    df["brand"] = df["brand"].fillna("unknown")
    df = df.sort_values(["user_session", "event_time"])
    print(f"  로드 {time.time()-t:.0f}s | {len(df):,}행 | 유저 {df.user_id.nunique():,} | 세션 {df.user_session.nunique():,}", flush=True)
    return df


def recommend_table(df):
    """유저별 관심 카테고리/브랜드 (view1/cart3/purchase5, remove −1) — 추천 기능 입력."""
    w = df["event_type"].astype(str).map(WEIGHT).fillna(0).to_numpy()
    d = pd.DataFrame({"user_id": df["user_id"].values, "category_id": df["category_id"].values,
                      "brand": df["brand"].astype(str).values, "w": w})
    topcat = (d.groupby(["user_id", "category_id"], observed=True)["w"].sum().reset_index()
              .sort_values("w").groupby("user_id").tail(1).rename(columns={"category_id": "top_category_id", "w": "cat_score"}))
    topbr = (d.groupby(["user_id", "brand"], observed=True)["w"].sum().reset_index()
             .sort_values("w").groupby("user_id").tail(1).rename(columns={"brand": "top_brand", "w": "brand_score"}))
    out = topcat.merge(topbr, on="user_id", how="outer")
    p = os.path.join(PROC, "recommend_user_interest.parquet")
    out.to_parquet(p, index=False)
    return p, len(out)


def step_feats(et, price, gap, cat, br):
    new_cat = np.concatenate([[0], (cat[1:] != cat[:-1]).astype(np.float32)])
    new_br = np.concatenate([[0], (br[1:] != br[:-1]).astype(np.float32)])
    return np.stack([np.log1p(gap),
                     (et == "view").astype(np.float32), (et == "cart").astype(np.float32),
                     (et == "remove_from_cart").astype(np.float32), (et == "purchase").astype(np.float32),
                     np.log1p(price), new_cat, new_br], 1).astype(np.float32)


def level_of(inc, g, cap):
    lv = 0.0; out = np.empty(len(inc), np.float32)
    for i, x in enumerate(inc):
        lv = min(max(RHO * lv + g * x / cap, 0.0), 1.0); out[i] = lv
    return out


def build(df, cat2idx, cap, g, tr_users, rng):
    X, CI, Y, G, TR = [], [], [], [], []
    for sid, s in df.groupby("user_session", sort=False, observed=True):
        n = len(s)
        if n < L + H:
            continue
        et = s["event_type"].values; price = s["price"].values.astype(np.float32)
        ts = s["event_time"].values.astype("datetime64[s]").astype(np.int64)
        gap = np.concatenate([[0], np.diff(ts)]).astype(np.float32)
        cat = s["category_id"].values; br = s["brand"].values
        feat = step_feats(et, price, gap, cat, br)
        cidx = np.array([cat2idx.get(int(c), 0) for c in cat], np.int64)
        inc = np.array([SIGN.get(e, 0.0) for e in et], np.float32) * price
        lv = level_of(inc, g, cap)
        uid = int(s["user_id"].iloc[0]); is_tr = uid in tr_users
        starts = list(range(L, n - H + 1))
        if len(starts) > MAXWIN:
            starts = list(rng.choice(starts, MAXWIN, replace=False))
        for t in starts:
            X.append(feat[t - L:t]); CI.append(cidx[t - L:t]); Y.append(lv[t:t + H])
            G.append(uid); TR.append(is_tr)
    return (np.stack(X).astype(np.float32), np.stack(CI).astype(np.int64),
            np.stack(Y).astype(np.float32), np.array(G, np.int64), np.array(TR, bool))


# ---------- DL (옵션 임베딩) ----------
class SeqForecast(nn.Module):
    def __init__(self, kind, f=C, h=64, horizon=H, n_cat=0):
        super().__init__(); self.kind = kind; self.use_emb = n_cat > 0
        fin = f
        if self.use_emb:
            self.emb = nn.Embedding(n_cat + 1, EMB_DIM, padding_idx=0); fin = f + EMB_DIM
        if kind == "lstm": self.rnn = nn.LSTM(fin, h, batch_first=True)
        elif kind == "gru": self.rnn = nn.GRU(fin, h, batch_first=True)
        elif kind == "transformer":
            self.proj = nn.Linear(fin, h); self.pos = nn.Parameter(torch.zeros(1, L, h))
            enc = nn.TransformerEncoderLayer(h, nhead=4, dim_feedforward=128, batch_first=True, dropout=0.1)
            self.rnn = nn.TransformerEncoder(enc, num_layers=2)
        self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, horizon))

    def forward(self, x, ci=None):
        if self.use_emb:
            x = torch.cat([x, self.emb(ci)], -1)
        if self.kind == "transformer":
            return self.head((self.rnn(self.proj(x) + self.pos)).mean(1))
        o, _ = self.rnn(x); return self.head(o[:, -1])


def train_dl(kind, Xtr, CItr, Ytr, Xte, CIte, n_cat=0, epochs=18):
    torch.manual_seed(SEED)
    mu = Xtr.reshape(-1, C).mean(0); sd = Xtr.reshape(-1, C).std(0) + 1e-6
    xtr = torch.tensor((Xtr - mu) / sd); xte = torch.tensor((Xte - mu) / sd)
    ytr = torch.tensor(Ytr); citr = torch.tensor(CItr); cite = torch.tensor(CIte)
    net = SeqForecast(kind, n_cat=n_cat); opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    lossf = nn.SmoothL1Loss(); n = len(xtr); bs = 1024
    for _ in range(epochs):
        net.train(); idx = torch.randperm(n)
        for i in range(0, n, bs):
            b = idx[i:i + bs]; opt.zero_grad()
            out = net(xtr[b], citr[b] if n_cat else None)
            lossf(out, ytr[b]).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(xte), 4096):
            preds.append(net(xte[i:i + 4096], cite[i:i + 4096] if n_cat else None).numpy())
    return np.concatenate(preds)


def metrics(Yt, Yp):
    per_h = [round(float(mean_squared_error(Yt[:, h], Yp[:, h]) ** 0.5), 4) for h in range(H)]
    over = (Yt.max(1) > OVERLOAD).astype(int)
    auc = float(roc_auc_score(over, Yp.max(1))) if 0 < over.mean() < 1 else float("nan")
    return {"mae": round(float(mean_absolute_error(Yt, Yp)), 4), "rmse": round(float(mean_squared_error(Yt, Yp) ** 0.5), 4),
            "rmse_per_h": per_h, "overload_auc": round(auc, 4), "overload_rate": round(float(over.mean()), 4)}


def main():
    print("[1] 풀 2019-Nov 로드…", flush=True)
    df = load_full()

    print("[2] 추천 피처표 생성(컬럼 보존: category_id/brand 가중)…", flush=True)
    rec_path, n_rec = recommend_table(df)
    print(f"  저장 {os.path.relpath(rec_path, HERE)} (유저 {n_rec:,})", flush=True)

    # 세션 길이 필터 + 유저 분할 + 캘리브(train만)
    slen = df.groupby("user_session", observed=True)["event_type"].transform("size")
    df = df[slen >= L + H].copy()
    df["user_session"] = df["user_session"].cat.remove_unused_categories()
    print(f"[3] 세션≥{L+H} → {df.user_session.nunique():,}세션 / {len(df):,}이벤트", flush=True)
    users = df["user_id"].unique(); rng = np.random.default_rng(SEED); rng.shuffle(users)
    tr_users = set(users[:int(len(users) * 0.75)])
    cap = float(np.percentile(df[df.user_id.isin(tr_users)]["price"].replace(0, np.nan).dropna(), 90))
    # g 캘리브(train 세션 inc로 step-overload≈0.3)
    sample_sids = df[df.user_id.isin(tr_users)]["user_session"].drop_duplicates().sample(min(3000, df.user_session.nunique()), random_state=SEED)
    incs = []
    for sid, s in df[df.user_session.isin(set(sample_sids))].groupby("user_session", observed=True):
        incs.append(np.array([SIGN.get(e, 0.0) for e in s["event_type"].values], np.float32) * s["price"].values.astype(np.float32))
    g = min([0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0],
            key=lambda gg: abs(np.mean([(level_of(i, gg, cap) > OVERLOAD).mean() for i in incs]) - 0.3))
    cat_ids = sorted(df["category_id"].unique()); cat2idx = {c: i + 1 for i, c in enumerate(cat_ids)}
    n_cat = len(cat_ids)
    print(f"  캘리브 cap={cap:.2f} g={g} | category {n_cat}종 임베딩", flush=True)

    print("[4] 윈도우 생성(세션당 ≤%d)…" % MAXWIN, flush=True)
    X, CI, Y, G, TR = build(df, cat2idx, cap, g, tr_users, rng)
    print(f"  윈도우 {len(X):,} | X{X.shape} CI{CI.shape} Y{Y.shape} | train {TR.sum():,}/test {(~TR).sum():,}", flush=True)

    # 저장(압축)
    ds = os.path.join(PROC, "session_level_full.npz")
    np.savez_compressed(ds, X=X, cat_idx=CI, Y=Y, user_id=G, is_train=TR)
    meta = {"dataset": "REES46 2019-Nov 전체(full)", "rows_raw": int(4635837), "unit": "user_session",
            "L": L, "H": H, "C": C, "rho": RHO, "g": g, "cap_price_p90": round(cap, 2),
            "n_category": n_cat, "emb_dim": EMB_DIM, "maxwin_per_session": MAXWIN,
            "feature_cols": ["gap_log", "is_view", "is_cart", "is_remove", "is_purchase", "price_log", "is_new_cat", "is_new_brand"],
            "n_windows": int(len(X)), "n_sessions_used": int(df.user_session.nunique()),
            "files": {"dataset": os.path.relpath(ds, HERE), "recommend": os.path.relpath(rec_path, HERE)}}
    json.dump(meta, open(os.path.join(PROC, "session_level_full_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  저장 {os.path.relpath(ds, HERE)} ({os.path.getsize(ds)/1024:.0f} KB)", flush=True)

    Xtr, Xte = X[TR], X[~TR]; CItr, CIte = CI[TR], CI[~TR]; Ytr, Yte = Y[TR], Y[~TR]
    Xtr_f, Xte_f = Xtr.reshape(len(Xtr), -1), Xte.reshape(len(Xte), -1)

    res = {}
    print("[5] Tabular", flush=True)
    for name, m in {"Ridge": Ridge(alpha=1.0),
                    "GBM": MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=300, random_state=SEED)),
                    "MLP": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=200, random_state=SEED)}.items():
        t = time.time(); m.fit(Xtr_f, Ytr); pred = m.predict(Xte_f).clip(0, 1); res[name] = metrics(Yte, pred)
        print(f"  {name:7s} RMSE {res[name]['rmse']:.4f} AUC {res[name]['overload_auc']:.4f} ({time.time()-t:.0f}s)", flush=True)
    print("[6] 시퀀스 DL", flush=True)
    for kind, disp in [("lstm", "LSTM"), ("gru", "GRU"), ("transformer", "Transformer")]:
        t = time.time(); pred = train_dl(kind, Xtr, CItr, Ytr, Xte, CIte, n_cat=0).clip(0, 1); res[disp] = metrics(Yte, pred)
        print(f"  {disp:11s} RMSE {res[disp]['rmse']:.4f} AUC {res[disp]['overload_auc']:.4f} ({time.time()-t:.0f}s)", flush=True)
    print("[7] 개선안: Transformer + category 임베딩", flush=True)
    pred = train_dl("transformer", Xtr, CItr, Ytr, Xte, CIte, n_cat=n_cat).clip(0, 1)
    res["Transformer+Emb"] = metrics(Yte, pred)
    print(f"  Transformer+Emb RMSE {res['Transformer+Emb']['rmse']:.4f} AUC {res['Transformer+Emb']['overload_auc']:.4f}", flush=True)
    naive = np.tile(Ytr.mean(0), (len(Yte), 1)); res["naive_climatology"] = metrics(Yte, naive)

    json.dump({"meta": meta, "results": res}, open(os.path.join(OUT, "forecast_session_full.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("저장 → forecast_session_full.json", flush=True)

    # ---------- 시각화: 한눈에 ----------
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        order = ["Ridge", "GBM", "MLP", "LSTM", "GRU", "Transformer", "Transformer+Emb"]
        colors = ["#7fa6cf", "#3b6fb0", "#7fa6cf", "#d98a8a", "#d98a8a", "#c0504d", "#7b1d1d"]
        fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))
        for m in order: ax[0].plot(range(1, H + 1), res[m]["rmse_per_h"], "-o", ms=3, label=m)
        ax[0].set_xlabel("horizon (step ahead)"); ax[0].set_ylabel("RMSE"); ax[0].legend(fontsize=7)
        ax[0].set_title("(a) RMSE by horizon")
        x = np.arange(len(order))
        ax[1].bar(x, [res[m]["rmse"] for m in order], color=colors)
        for i, m in enumerate(order): ax[1].text(i, res[m]["rmse"] + .002, f"{res[m]['rmse']:.3f}", ha="center", fontsize=7)
        ax[1].set_xticks(x); ax[1].set_xticklabels(order, rotation=35, ha="right"); ax[1].set_title("(b) overall RMSE ↓")
        ax[2].bar(x, [res[m]["overload_auc"] for m in order], color=colors)
        for i, m in enumerate(order): ax[2].text(i, res[m]["overload_auc"] + .002, f"{res[m]['overload_auc']:.3f}", ha="center", fontsize=7)
        ax[2].set_xticks(x); ax[2].set_xticklabels(order, rotation=35, ha="right"); ax[2].set_ylim(0.5, 1); ax[2].set_title("(c) high-intent AUC ↑")
        fig.suptitle("17-3-4  REES46 2019-Nov FULL — session basket-level forecast (all models)", fontsize=12)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "forecast_session_full.png"), dpi=110)
        print("저장 → forecast_session_full.png", flush=True)
    except Exception as e:
        print("[plot skip]", e, flush=True)


if __name__ == "__main__":
    main()

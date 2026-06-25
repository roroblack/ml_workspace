# -*- coding: utf-8 -*-
"""17-4-0(데이터 추출) + 17-3-5(모델 비교→1개 선정): REES46 다음-카테고리 예측(순서민감).

과제: 세션 내 직전 L 이벤트 → '다음 이벤트의 category_id'(491종) 예측. 순서가 곧 신호.
  - 이게 SASRec/BERT4Rec(시퀀스 추천 트랜스포머)가 트리/집계를 '확실히' 이기는 영역.
  - 추천 기능과 직결(다음 관심 카테고리).

17-4-0 산출(데이터): data/processed_nextcat/nextcat_dataset.npz (X_num, X_cat, y, user_id, is_train) + vocab/meta
17-3-5 산출(모델): outputs/realtime/nextcat_models.json + .png  → 베스트 모델 선정
"""
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, torch, torch.nn as nn
import session_full_1734 as M   # load_full 재사용

PROC = os.path.join(M.SP, "data", "processed_nextcat"); os.makedirs(PROC, exist_ok=True)
OUT = M.OUT
L = 10
SESS_CAP = 40000            # 세션 표본(속도)
MAXWIN = 4
SEED = 42
NUMC = 6                    # is_view,is_cart,is_remove,is_purchase,gap_log,price_log
np.random.seed(SEED); torch.manual_seed(SEED)


# ---------- 17-4-0: 데이터 추출 ----------
def build_dataset():
    df = M.load_full()
    slen = df.groupby("user_session", observed=True)["event_type"].transform("size")
    df = df[slen >= L + 1].copy()
    df["user_session"] = df["user_session"].cat.remove_unused_categories()
    sids = df["user_session"].cat.categories.to_numpy()
    rng = np.random.default_rng(SEED)
    if len(sids) > SESS_CAP:
        keep = set(rng.choice(sids, SESS_CAP, replace=False))
        df = df[df["user_session"].isin(keep)].copy()
        df["user_session"] = df["user_session"].cat.remove_unused_categories()
    print(f"[17-4-0] 세션≥{L+1} 표본 {df.user_session.nunique():,} / {len(df):,}이벤트", flush=True)

    # 유저 분할 → vocab은 train 카테고리로만
    users = df["user_id"].unique(); rng.shuffle(users)
    tr_users = set(users[:int(len(users) * 0.75)])
    cats = sorted(df[df.user_id.isin(tr_users)]["category_id"].unique())
    cat2idx = {c: i + 1 for i, c in enumerate(cats)}      # 0=unknown
    K = len(cats)
    price = df["price"].to_numpy(np.float32)
    gap_cap = float(np.percentile(price[price > 0], 90))  # (미사용; price log만)
    Xn, Xc, Y, G, TR = [], [], [], [], []
    for sid, s in df.groupby("user_session", sort=False, observed=True):
        et = s["event_type"].to_numpy(str)
        pr = s["price"].to_numpy(np.float32)
        ts = s["event_time"].values.astype("datetime64[s]").astype(np.int64)
        gap = np.concatenate([[0], np.diff(ts)]).astype(np.float32)
        cat = s["category_id"].to_numpy()
        cidx = np.array([cat2idx.get(int(c), 0) for c in cat], np.int64)
        num = np.stack([(et == "view").astype(np.float32), (et == "cart").astype(np.float32),
                        (et == "remove_from_cart").astype(np.float32), (et == "purchase").astype(np.float32),
                        np.log1p(gap), np.log1p(pr)], 1).astype(np.float32)
        n = len(s); uid = int(s["user_id"].iloc[0]); is_tr = uid in tr_users
        starts = list(range(L, n))                        # 입력 [t-L:t], 타깃 cidx[t]
        if len(starts) > MAXWIN:
            starts = list(rng.choice(starts, MAXWIN, replace=False))
        for t in starts:
            Xn.append(num[t - L:t]); Xc.append(cidx[t - L:t]); Y.append(cidx[t]); G.append(uid); TR.append(is_tr)
    Xn = np.stack(Xn).astype(np.float32); Xc = np.stack(Xc).astype(np.int64)
    Y = np.array(Y, np.int64); G = np.array(G, np.int64); TR = np.array(TR, bool)
    # 저장(17-4-0)
    ds = os.path.join(PROC, "nextcat_dataset.npz")
    np.savez_compressed(ds, X_num=Xn, X_cat=Xc, y=Y, user_id=G, is_train=TR)
    meta = {"dataset": "REES46 2019-Nov 전체→세션표본", "task": "next-category(491) prediction",
            "L": L, "n_category": K, "numeric_cols": ["is_view", "is_cart", "is_remove", "is_purchase", "gap_log", "price_log"],
            "n_windows": int(len(Y)), "n_sessions": int(df.user_session.nunique()),
            "sess_cap": SESS_CAP, "maxwin": MAXWIN,
            "file": os.path.relpath(ds, M.HERE), "size_KB": round(os.path.getsize(ds) / 1024, 1)}
    json.dump(meta, open(os.path.join(PROC, "nextcat_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[17-4-0] 저장 {meta['file']} ({meta['size_KB']:.0f} KB) | 윈도우 {len(Y):,} | category {K}", flush=True)
    return Xn, Xc, Y, G, TR, K


# ---------- 지표 ----------
def topk_metrics(logits, y, k=10):
    top1 = (logits.argmax(1) == y).mean()
    idx = np.argsort(-logits, 1)[:, :k]
    hit = np.array([yi in row for yi, row in zip(y, idx)]).mean()
    mrr = 0.0
    for yi, row in zip(y, idx):
        pos = np.where(row == yi)[0]
        if len(pos): mrr += 1.0 / (pos[0] + 1)
    return {"top1": round(float(top1), 4), "hit@10": round(float(hit), 4), "mrr@10": round(float(mrr / len(y)), 4)}


def rank_metrics(rank_lists, y, k=10):
    top1 = np.mean([r[0] == yi for r, yi in zip(rank_lists, y)])
    hit = np.mean([yi in r[:k] for r, yi in zip(rank_lists, y)])
    mrr = 0.0
    for r, yi in zip(rank_lists, y):
        if yi in r[:k]: mrr += 1.0 / (r[:k].index(yi) + 1)
    return {"top1": round(float(top1), 4), "hit@10": round(float(hit), 4), "mrr@10": round(float(mrr / len(y)), 4)}


# ---------- DL ----------
class NextCat(nn.Module):
    def __init__(self, kind, K, d=64, emb=32):
        super().__init__(); self.kind = kind; self.K = K
        self.cat_emb = nn.Embedding(K + 1, emb, padding_idx=0)
        fin = emb + NUMC
        if kind in ("gru", "lstm"):
            self.rnn = (nn.GRU if kind == "gru" else nn.LSTM)(fin, d, batch_first=True)
            self.head = nn.Linear(d, K + 1)
        else:  # transformer(bi) / sasrec(causal)
            self.proj = nn.Linear(fin, d); self.pos = nn.Parameter(torch.zeros(1, L, d))
            enc = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=128, batch_first=True, dropout=0.1)
            self.tr = nn.TransformerEncoder(enc, num_layers=2)
            self.head = nn.Linear(d, K + 1)
            self.causal = kind == "sasrec"

    def forward(self, xc, xn):
        x = torch.cat([self.cat_emb(xc), xn], -1)
        if self.kind in ("gru", "lstm"):
            o, _ = self.rnn(x); return self.head(o[:, -1])
        z = self.proj(x) + self.pos
        mask = None
        if self.causal:
            mask = torch.triu(torch.ones(L, L) * float("-inf"), diagonal=1)
        z = self.tr(z, mask=mask)
        return self.head(z[:, -1])           # 마지막 위치 → 다음 카테고리


def train_dl(kind, Xc, Xn, y, Xc_te, Xn_te, K, mu, sd, epochs=14):
    torch.manual_seed(SEED)
    xn = torch.tensor((Xn - mu) / sd); xnt = torch.tensor((Xn_te - mu) / sd)
    xc = torch.tensor(Xc); xct = torch.tensor(Xc_te); yt = torch.tensor(y)
    net = NextCat(kind, K); opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    lossf = nn.CrossEntropyLoss(); n = len(xc); bs = 512
    for _ in range(epochs):
        net.train(); idx = torch.randperm(n)
        for i in range(0, n, bs):
            b = idx[i:i + bs]; opt.zero_grad()
            lossf(net(xc[b], xn[b]), yt[b]).backward(); opt.step()
    net.eval(); outs = []
    with torch.no_grad():
        for i in range(0, len(xct), 4096):
            outs.append(net(xct[i:i + 4096], xnt[i:i + 4096]).numpy())
    return np.concatenate(outs)


def main():
    Xn, Xc, Y, G, TR, K = build_dataset()
    te = ~TR
    Xn_tr, Xn_te = Xn[TR], Xn[te]; Xc_tr, Xc_te = Xc[TR], Xc[te]; y_tr, y_te = Y[TR], Y[te]
    mu = Xn_tr.reshape(-1, NUMC).mean(0); sd = Xn_tr.reshape(-1, NUMC).std(0) + 1e-6
    print(f"[17-3-5] train {TR.sum():,} / test {te.sum():,} | category {K}", flush=True)

    res = {}
    # 베이스라인(비학습 휴리스틱)
    pop_order = list(np.argsort(-np.bincount(y_tr, minlength=K + 1)))   # 인기순
    res["Popularity"] = rank_metrics([pop_order for _ in y_te], list(y_te))
    # Last-category & Recency+Pop
    rl_last, rl_recpop = [], []
    for row in Xc_te:
        wc = [int(c) for c in row if c != 0]
        seen, rec = set(), []
        for c in reversed(wc):
            if c not in seen: seen.add(c); rec.append(c)
        rl_last.append(rec + [c for c in pop_order if c not in seen])
        rl_recpop.append(rec + [c for c in pop_order if c not in seen])
    res["Last-category"] = rank_metrics(rl_last, list(y_te))            # top1=마지막 카테고리
    print(f"  [base] Popularity hit@10 {res['Popularity']['hit@10']} | Last-cat top1 {res['Last-category']['top1']}", flush=True)

    for kind, disp in [("lstm", "LSTM"), ("gru", "GRU"), ("transformer", "Transformer"), ("sasrec", "SASRec")]:
        t = time.time()
        logits = train_dl(kind, Xc_tr, Xn_tr, y_tr, Xc_te, Xn_te, K, mu, sd)
        res[disp] = topk_metrics(logits, y_te)
        print(f"  [DL] {disp:11s} top1 {res[disp]['top1']:.4f} hit@10 {res[disp]['hit@10']:.4f} mrr {res[disp]['mrr@10']:.4f} ({time.time()-t:.0f}s)", flush=True)

    best = max(res, key=lambda m: res[m]["hit@10"])
    print(f"\n  ★ 베스트(hit@10): {best} = {res[best]}", flush=True)
    json.dump({"K": K, "n_windows": int(len(Y)), "best": best, "results": res},
              open(os.path.join(OUT, "nextcat_models.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        order = ["Popularity", "Last-category", "LSTM", "GRU", "Transformer", "SASRec"]
        order = [m for m in order if m in res]
        x = np.arange(len(order)); w = 0.27
        fig, ax = plt.subplots(figsize=(11, 4.5))
        for j, (mk, lab) in enumerate([("top1", "Top-1 acc"), ("hit@10", "Hit@10"), ("mrr@10", "MRR@10")]):
            b = ax.bar(x + (j - 1) * w, [res[m][mk] for m in order], w, label=lab)
        cols = ["#9bbcd6", "#9bbcd6", "#d98a8a", "#d98a8a", "#c0504d", "#7b1d1d"]
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=20)
        ax.set_title("17-3-5  Next-category prediction (order-sensitive) — baselines vs DL vs SASRec")
        ax.legend(); ax.axvspan(1.5, len(order) - 0.5, color="#f5f5f5", zorder=0)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "nextcat_models.png"), dpi=110)
        print("저장 → nextcat_models.png", flush=True)
    except Exception as e:
        print("[plot skip]", e, flush=True)


if __name__ == "__main__":
    main()

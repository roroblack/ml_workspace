# -*- coding: utf-8 -*-
"""실시간(슬라이딩 윈도우) 이탈예측 — 여러 모델 비교 + recency 변화 수치화.

설계: 한 시점의 단일 스냅샷이 아니라, 각 유저의 로그를 매일 흘리며
  - 평가일 t 마다 '직전 14일 창'을 만들고
  - 라벨 = (t+1 ~ t+7) 무활동이면 이탈(1)        ← 실시간 라벨(슬라이딩)
  - 같은 창에서 (a) 시퀀스[14,3] (b) 집계피처 를 함께 생성
=> recency가 '하나의 숫자'가 아니라 '창 안 어디에 활동이 있나'(순서)로 바뀐다.

비교(같은 user-level split):
  ML  counts_only   (recency 제외, 볼륨만)         : LogReg/GBM/MLP
  ML  counts+recency(recency 숫자 추가)            : LogReg/GBM/MLP
  DL  sequence      (14일 시퀀스 모양 전체)         : LSTM/GRU/Transformer
ablation으로 'recency 숫자가 주는 이득' vs '시퀀스 모양이 recency 너머로 주는 이득'을 분리.

출력: sample_project/outputs/realtime/modelcompare.json + 콘솔 표.
"""
import os, sys, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # repo root(team_project_churn)
SP = os.path.join(HERE, "sample_project")
RAW = os.path.join(SP, "data", "raw", "events.csv")
OUT = os.path.join(SP, "outputs", "realtime"); os.makedirs(OUT, exist_ok=True)
ETYPES = ["view", "cart", "purchase"]
L = 14            # 관찰 창
H = 7             # 결과 창(무활동=이탈)
SEED = 42
USER_SAMPLE = 5000
np.random.seed(SEED); torch.manual_seed(SEED)


# ---------- 데이터셋: 슬라이딩 윈도우 샘플 ----------
def build_dataset():
    df = pd.read_csv(RAW, usecols=["user_id", "event_time", "event_type"])
    df["event_time"] = pd.to_datetime(df["event_time"])
    base = df["event_time"].min().normalize()
    df["day"] = (df["event_time"].dt.normalize() - base).dt.days
    ndays = int(df["day"].max()) + 1
    df = df[df["event_type"].isin(ETYPES)]
    et_i = {e: i for i, e in enumerate(ETYPES)}

    users = df["user_id"].unique()
    if len(users) > USER_SAMPLE:
        users = np.random.choice(users, USER_SAMPLE, replace=False)
    df = df[df["user_id"].isin(users)]

    # 유저별 [ndays,3] 카운트 행렬
    grp = df.groupby(["user_id", "day", "event_type"]).size().reset_index(name="c")
    mats = {u: np.zeros((ndays, 3), np.float32) for u in users}
    for u, d, e, c in grp.itertuples(index=False):
        mats[u][int(d), et_i[e]] = c

    eval_days = [t for t in range(L - 1, ndays - H)]   # 14일 창 + 7일 미래 확보
    SEQ, TAB, Y, G = [], [], [], []
    for u in users:
        arr = mats[u]; tot = arr.sum(1)
        for t in eval_days:
            w = arr[t - L + 1:t + 1]                    # [14,3]
            if w.sum() == 0:                            # 창에 활동 없으면 제외(예측 대상 아님)
                continue
            future_active = tot[t + 1:t + 1 + H].sum() > 0
            label = 0 if future_active else 1
            actday = np.where(w.sum(1) > 0)[0]
            recency = float(L - 1 - actday[-1])         # 창 끝 기준 마지막 활동까지 일수
            counts = w.sum(0)                           # [view,cart,purchase]
            n_events = float(w.sum()); active_days = float((w.sum(1) > 0).sum())
            SEQ.append(w.copy())
            TAB.append([recency, counts[0], counts[1], counts[2], n_events, active_days])
            Y.append(label); G.append(u)
    SEQ = np.stack(SEQ); TAB = np.array(TAB, np.float32)
    Y = np.array(Y, np.int64); G = np.array(G)
    return SEQ, TAB, Y, G, base, ndays


TAB_COLS = ["recency", "n_view", "n_cart", "n_purchase", "n_events", "active_days"]


# ---------- recency 변화 수치화 ----------
def recency_report(TAB, Y):
    rec = TAB[:, 0]; ev = TAB[:, 4]
    lines = []
    lines.append(f"샘플 {len(Y):,} | 이탈률 {Y.mean()*100:.1f}% | recency 평균 {rec.mean():.2f}일 "
                 f"(중앙 {np.median(rec):.0f}, 표준편차 {rec.std():.2f}, 최대 {rec.max():.0f})")
    # recency 버킷별 이탈률
    buckets = [(0, 0, "0일(오늘 활동)"), (1, 2, "1-2일"), (3, 4, "3-4일"),
               (5, 6, "5-6일"), (7, 13, "7-13일")]
    by_rec = []
    for lo, hi, name in buckets:
        m = (rec >= lo) & (rec <= hi)
        if m.sum():
            by_rec.append((name, int(m.sum()), float(Y[m].mean())))
    # 활동량 고정(동일 n_events 구간) 안에서 recency별 이탈률 → 볼륨과 분리된 recency 효과
    q = np.quantile(ev, [0, .33, .66, 1.0])
    strata = []
    for si, (a, b) in enumerate([(q[0], q[1]), (q[1], q[2]), (q[2], q[3])]):
        m = (ev >= a) & (ev <= b)
        sub_rec = rec[m]; sub_y = Y[m]
        lo_m = sub_rec <= 1; hi_m = sub_rec >= 5
        row = (f"활동량 {['하','중','상'][si]}(n_events {a:.0f}~{b:.0f})", int(m.sum()),
               float(sub_y[lo_m].mean()) if lo_m.sum() else float("nan"),
               float(sub_y[hi_m].mean()) if hi_m.sum() else float("nan"))
        strata.append(row)
    return lines, by_rec, strata


# ---------- DL 모델 ----------
class SeqNet(nn.Module):
    def __init__(self, kind, f=3, h=32):
        super().__init__()
        self.kind = kind
        if kind == "lstm":
            self.rnn = nn.LSTM(f, h, batch_first=True)
        elif kind == "gru":
            self.rnn = nn.GRU(f, h, batch_first=True)
        elif kind == "transformer":
            self.proj = nn.Linear(f, h)
            self.pos = nn.Parameter(torch.zeros(1, L, h))
            enc = nn.TransformerEncoderLayer(h, nhead=4, dim_feedforward=64,
                                             batch_first=True, dropout=0.1)
            self.rnn = nn.TransformerEncoder(enc, num_layers=2)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(h, 1))

    def forward(self, x):
        if self.kind == "transformer":
            z = self.proj(x) + self.pos
            z = self.rnn(z)
            return self.head(z.mean(1)).squeeze(1)
        o, _ = self.rnn(x)
        return self.head(o[:, -1]).squeeze(1)


def train_dl(kind, Xtr, ytr, Xte, yte, epochs=25):
    torch.manual_seed(SEED)
    mu = Xtr.reshape(-1, 3).mean(0); sd = Xtr.reshape(-1, 3).std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    xtr = torch.tensor(Xtr); ytr_t = torch.tensor(ytr, dtype=torch.float32)
    xte = torch.tensor(Xte)
    pos_w = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float32)
    net = SeqNet(kind); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    n = len(xtr); bs = 512
    for ep in range(epochs):
        net.train(); idx = torch.randperm(n)
        for i in range(0, n, bs):
            b = idx[i:i + bs]
            opt.zero_grad(); out = net(xtr[b]); loss = lossf(out, ytr_t[b])
            loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(xte)).numpy()
    return roc_auc_score(yte, p), average_precision_score(yte, p)


# ---------- ML ----------
def ml_models():
    return {
        "LogReg": lambda: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        "GBM": lambda: GradientBoostingClassifier(random_state=SEED),
        "MLP": lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=SEED),
    }


def eval_ml(make, Xtr, ytr, Xte, yte, scale=True):
    if scale:
        sc = StandardScaler().fit(Xtr); Xtr = sc.transform(Xtr); Xte = sc.transform(Xte)
    clf = make().fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, p), average_precision_score(yte, p)


def main():
    print("[1/3] 슬라이딩 윈도우 데이터셋 생성…")
    SEQ, TAB, Y, G, base, ndays = build_dataset()
    lines, by_rec, strata = recency_report(TAB, Y)

    # user-level split (누수 차단)
    uniq = np.unique(G); np.random.seed(SEED); np.random.shuffle(uniq)
    cut = int(len(uniq) * 0.75); tr_u = set(uniq[:cut])
    tr = np.array([g in tr_u for g in G]); te = ~tr
    print(f"  train {tr.sum():,} / test {te.sum():,} (유저 {len(uniq):,}명 75/25 분할)")

    print("[2/3] recency 변화 수치화")
    for l in lines: print("  " + l)
    print("  · recency 버킷별 이탈률:")
    for name, n, r in by_rec: print(f"     {name:14s} n={n:6d}  이탈 {r*100:5.1f}%")
    print("  · 활동량 고정 시 recency 효과(이탈률, recency≤1d vs ≥5d):")
    for name, n, lo, hi in strata:
        print(f"     {name:24s} n={n:6d}  ≤1일 {lo*100:5.1f}%  →  ≥5일 {hi*100:5.1f}%  (Δ{(hi-lo)*100:+.1f}p)")

    print("[3/3] 모델 비교 (test ROC-AUC / PR-AUC)")
    res = {"counts_only": {}, "counts_recency": {}, "sequence": {}}
    # 집계피처 두 세트
    cnt_idx = [1, 2, 3, 4, 5]               # recency 제외(볼륨만)
    full_idx = [0, 1, 2, 3, 4, 5]           # recency 포함
    Xc_tr, Xc_te = TAB[tr][:, cnt_idx], TAB[te][:, cnt_idx]
    Xf_tr, Xf_te = TAB[tr][:, full_idx], TAB[te][:, full_idx]
    ytr, yte = Y[tr], Y[te]
    for name, make in ml_models().items():
        sc = name != "GBM"
        res["counts_only"][name] = eval_ml(make, Xc_tr, ytr, Xc_te, yte, scale=sc)
        res["counts_recency"][name] = eval_ml(make, Xf_tr, ytr, Xf_te, yte, scale=sc)
        print(f"  [ML] {name:8s} counts_only AUC {res['counts_only'][name][0]:.4f} | "
              f"+recency AUC {res['counts_recency'][name][0]:.4f}")
    for kind in ["lstm", "gru", "transformer"]:
        auc, pr = train_dl(kind, SEQ[tr], ytr, SEQ[te], yte)
        res["sequence"][kind] = (auc, pr)
        print(f"  [DL] {kind:11s} sequence    AUC {auc:.4f} | PR {pr:.4f}")

    best_ml = max(v[0] for v in res["counts_recency"].values())
    best_dl = max(v[0] for v in res["sequence"].values())
    print(f"\n  최고 ML(+recency) {best_ml:.4f}  vs  최고 DL(sequence) {best_dl:.4f}  → Δ{(best_dl-best_ml)*100:+.2f}p")

    out = {
        "n_samples": int(len(Y)), "churn_rate": round(float(Y.mean()), 4),
        "recency_stats": lines,
        "churn_by_recency": [{"bucket": n, "n": c, "churn": round(r, 4)} for n, c, r in by_rec],
        "recency_within_activity": [{"stratum": n, "n": c, "churn_le1d": round(lo, 4),
                                     "churn_ge5d": round(hi, 4)} for n, c, lo, hi in strata],
        "results": {k: {m: {"auc": round(a, 4), "pr": round(p, 4)} for m, (a, p) in v.items()}
                    for k, v in res.items()},
        "best_ml_recency": round(best_ml, 4), "best_dl_sequence": round(best_dl, 4),
    }
    json.dump(out, open(os.path.join(OUT, "modelcompare.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n저장 → sample_project/outputs/realtime/modelcompare.json")


if __name__ == "__main__":
    main()

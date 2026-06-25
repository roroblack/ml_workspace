# -*- coding: utf-8 -*-
"""17-3-1. 유저 '활동강도(load)' 궤적 다스텝 예측 — RNN계열 vs Tabular.

레짐 2(누적/동역학 시계열 예측) 검증: churn을 이진분류가 아니라
'각 유저=노드, 활동강도(events/day) 라는 부하의 궤적'을 예측하는 문제로 재프레이밍.
  입력: 과거 N_IN일 일별 [view,cart,purchase]   →  출력: 향후 H일 일별 활동강도(events/day)
  파생 churn: 향후 H일 합계 강도 == 0 (완전 휴면) → 이탈
풀 로스터(Ridge/GBM/MLP + LSTM/GRU/Transformer)로 멀티스텝 예측오차(호라이즌별) + churn AUC 비교.

대상: 실제 REES46(2019-Nov). 산출: sample_project/outputs/realtime/forecast_user.json + .png
"""
import os, sys, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # repo root(team_project_churn)
SP = os.path.join(HERE, "sample_project")
RAW = os.path.join(SP, "data", "raw", "events.csv")
OUT = os.path.join(SP, "outputs", "realtime"); os.makedirs(OUT, exist_ok=True)
ETYPES = ["view", "cart", "purchase"]
N_IN = 10            # 입력 과거일
H = 5               # 예측 호라이즌(일)
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)


def build_windows():
    df = pd.read_csv(RAW, usecols=["user_id", "event_time", "event_type"])
    df["event_time"] = pd.to_datetime(df["event_time"])
    base = df["event_time"].min().normalize()
    df["day"] = (df["event_time"].dt.normalize() - base).dt.days
    df = df[df["event_type"].isin(ETYPES)]
    ndays = int(df["day"].max()) + 1
    eti = {e: i for i, e in enumerate(ETYPES)}
    grp = df.groupby(["user_id", "day", "event_type"]).size().reset_index(name="c")
    users = df["user_id"].unique()
    mats = {u: np.zeros((ndays, 3), np.float32) for u in users}
    for u, d, e, c in grp.itertuples(index=False):
        mats[u][int(d), eti[e]] = c
    # 활동일 5일 이상 유저만(궤적에 형태가 있어야 예측이 의미)
    active5 = [u for u in users if (mats[u].sum(1) > 0).sum() >= 5]
    Xin, Yout, G = [], [], []
    for u in active5:
        arr = mats[u]; intensity = arr.sum(1)            # 일별 활동강도
        for t in range(N_IN, ndays - H + 1):
            win = arr[t - N_IN:t]
            if win.sum() == 0:                           # 입력이 전부 0이면 제외
                continue
            Xin.append(win.copy())                       # [N_IN,3]
            Yout.append(intensity[t:t + H].copy())       # [H]
            G.append(u)
    return np.stack(Xin), np.stack(Yout), np.array(G), len(active5)


# ---------- DL seq2(multi-step) ----------
class SeqForecast(nn.Module):
    def __init__(self, kind, f=3, h=48, horizon=H):
        super().__init__(); self.kind = kind
        if kind == "lstm": self.rnn = nn.LSTM(f, h, batch_first=True)
        elif kind == "gru": self.rnn = nn.GRU(f, h, batch_first=True)
        elif kind == "transformer":
            self.proj = nn.Linear(f, h); self.pos = nn.Parameter(torch.zeros(1, N_IN, h))
            enc = nn.TransformerEncoderLayer(h, nhead=4, dim_feedforward=96, batch_first=True, dropout=0.1)
            self.rnn = nn.TransformerEncoder(enc, num_layers=2)
        self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, horizon))

    def forward(self, x):
        if self.kind == "transformer":
            return self.head((self.rnn(self.proj(x) + self.pos)).mean(1))
        o, _ = self.rnn(x); return self.head(o[:, -1])


def train_dl(kind, Xtr, Ytr, Xte, epochs=40):
    torch.manual_seed(SEED)
    mu = Xtr.reshape(-1, 3).mean(0); sd = Xtr.reshape(-1, 3).std(0) + 1e-6
    xtr = torch.tensor((Xtr - mu) / sd); xte = torch.tensor((Xte - mu) / sd)
    ytr = torch.tensor(Ytr)                              # 이미 log1p 변환된 타깃
    net = SeqForecast(kind); opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    lossf = nn.SmoothL1Loss(); n = len(xtr); bs = 256
    for _ in range(epochs):
        net.train(); idx = torch.randperm(n)
        for i in range(0, n, bs):
            b = idx[i:i + bs]; opt.zero_grad()
            lossf(net(xtr[b]), ytr[b]).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(xte).numpy()


def metrics(Ytrue_raw, Ypred_raw):
    """호라이즌별 + 전체 MAE/RMSE, churn(향후합=0) AUC."""
    per_h = []
    for h in range(H):
        rmse = mean_squared_error(Ytrue_raw[:, h], Ypred_raw[:, h]) ** 0.5
        per_h.append(round(float(rmse), 4))
    mae = float(mean_absolute_error(Ytrue_raw, Ypred_raw))
    rmse = float(mean_squared_error(Ytrue_raw, Ypred_raw) ** 0.5)
    churn = (Ytrue_raw.sum(1) == 0).astype(int)
    score = -Ypred_raw.sum(1)                            # 예측 강도 낮을수록 이탈
    auc = float(roc_auc_score(churn, score)) if 0 < churn.mean() < 1 else float("nan")
    return {"mae": round(mae, 4), "rmse": round(rmse, 4),
            "rmse_per_h": per_h, "churn_auc": round(auc, 4), "churn_rate": round(float(churn.mean()), 4)}


def main():
    print("[1] 윈도우 생성(실제 REES46, 활동5일+ 유저)…")
    X, Y, G, nuser = build_windows()
    print(f"  유저 {nuser:,} | 윈도우 {len(X):,} | 입력 {X.shape} → 출력 {Y.shape} | "
          f"향후{H}일 휴면(churn) {(Y.sum(1)==0).mean():.1%}")

    # user-level 분할
    uniq = np.unique(G); np.random.shuffle(uniq)
    tr_u = set(uniq[:int(len(uniq) * 0.75)])
    trm = np.array([g in tr_u for g in G]); tem = ~trm
    Ylog = np.log1p(Y)                                   # 스큐 보정 타깃(모든 모델 공통)
    Xtr, Xte = X[trm], X[tem]; Ytr_log, Yte_raw = Ylog[trm], Y[tem]
    Xtr_flat = Xtr.reshape(len(Xtr), -1); Xte_flat = Xte.reshape(len(Xte), -1)
    print(f"  train {trm.sum():,} / test {tem.sum():,}")

    res = {}
    print("[2] Tabular (lag flatten 입력, log1p 타깃, 멀티아웃풋)")
    tab = {"Ridge": Ridge(alpha=1.0),
           "GBM": MultiOutputRegressor(GradientBoostingRegressor(random_state=SEED)),
           "MLP": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=SEED)}
    for name, m in tab.items():
        m.fit(Xtr_flat, Ytr_log)
        pred_raw = np.expm1(m.predict(Xte_flat)).clip(min=0)
        res[name] = metrics(Yte_raw, pred_raw)
        print(f"  {name:7s} RMSE {res[name]['rmse']:.3f} MAE {res[name]['mae']:.3f} "
              f"| churnAUC {res[name]['churn_auc']:.4f} | per-h {res[name]['rmse_per_h']}")

    print("[3] 시퀀스 DL (seq2multi-step)")
    for kind, disp in [("lstm", "LSTM"), ("gru", "GRU"), ("transformer", "Transformer")]:
        pred_raw = np.expm1(train_dl(kind, Xtr, Ytr_log, Xte)).clip(min=0)
        res[disp] = metrics(Yte_raw, pred_raw)
        print(f"  {disp:11s} RMSE {res[disp]['rmse']:.3f} MAE {res[disp]['mae']:.3f} "
              f"| churnAUC {res[disp]['churn_auc']:.4f} | per-h {res[disp]['rmse_per_h']}")

    # 베이스라인: 마지막 값 유지(persistence) & 입력평균
    last = np.repeat(X[tem].sum(2)[:, -1:], H, axis=1)
    res["naive_persist"] = metrics(Yte_raw, last)
    print(f"  {'naive(지속)':11s} RMSE {res['naive_persist']['rmse']:.3f} "
          f"| churnAUC {res['naive_persist']['churn_auc']:.4f}")

    json.dump({"n_users": int(nuser), "n_windows": int(len(X)), "N_IN": N_IN, "H": H,
               "results": res}, open(os.path.join(OUT, "forecast_user.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("저장 → sample_project/outputs/realtime/forecast_user.json")

    # 그래프: 호라이즌별 RMSE 곡선 + churnAUC 막대
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        order = ["Ridge", "GBM", "MLP", "LSTM", "GRU", "Transformer"]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
        for m in order:
            a1.plot(range(1, H + 1), res[m]["rmse_per_h"], "-o", label=m)
        a1.plot(range(1, H + 1), res["naive_persist"]["rmse_per_h"], "--", c="grey", label="naive")
        a1.set_xlabel("forecast horizon (day ahead)"); a1.set_ylabel("RMSE"); a1.legend(fontsize=8)
        a1.set_title("User activity-load forecast: RMSE by horizon")
        x = np.arange(len(order))
        a2.bar(x, [res[m]["churn_auc"] for m in order],
               color=["#3b6fb0"] * 3 + ["#c0504d"] * 3)
        for i, m in enumerate(order):
            a2.text(i, res[m]["churn_auc"] + .003, f"{res[m]['churn_auc']:.3f}", ha="center", fontsize=7)
        a2.set_xticks(x); a2.set_xticklabels(order, rotation=30); a2.set_ylim(0.5, 1)
        a2.set_title("Churn-from-forecast AUC"); a2.axvspan(2.5, 5.5, color="#f3f3f3", zorder=0)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "forecast_user.png"), dpi=110)
        print("저장 → sample_project/outputs/realtime/forecast_user.png")
    except Exception as e:
        print("[plot skip]", e)


if __name__ == "__main__":
    main()

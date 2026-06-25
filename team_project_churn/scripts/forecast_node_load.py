# -*- coding: utf-8 -*-
"""17-3-2. 다(多)노드 서비스 부하 예측 — RNN계열 vs Tabular (레짐 2의 정공법).

'노드 하나의 부하가 커지는' 동역학을 의도적으로 구성:
  잠재수요 demand_t = a·demand_{t-1} + 계절성(diurnal) + 노드추세 + 충격   (모멘텀=누적)
  부하 load_t = sigmoid(k·(demand_t - d0))                                  (비선형 포화)
→ hidden state가 과거를 적분해야 하고(모멘텀), 포화 비선형 + 다(多)노드 공유동역학 +
  장기 호라이즌 → RNN/시퀀스가 lag-트리 대비 유리해야 하는 전형적 레짐.

입력: 과거 N_IN스텝 [load, sin(시각), cos(시각)] → 출력: 향후 H스텝 load(용량대비 비율).
파생 분류: 향후 H스텝 내 과부하(load>0.85) 발생 여부 AUC.
풀 로스터(Ridge/GBM/MLP + LSTM/GRU/Transformer) + naive 베이스라인.

산출: sample_project/outputs/realtime/forecast_node.json + .png
"""
import os, sys, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, torch, torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # repo root(team_project_churn)
OUT = os.path.join(HERE, "sample_project", "outputs", "realtime"); os.makedirs(OUT, exist_ok=True)
N_NODES = 400
T = 240              # 스텝(시간)
P = 24               # 일주기
N_IN = 24
H = 12
STRIDE = 6
OVERLOAD = 0.85
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)


def simulate_nodes():
    rng = np.random.default_rng(SEED)
    loads = np.zeros((N_NODES, T), np.float32)
    for i in range(N_NODES):
        a = rng.uniform(0.55, 0.9)                 # 모멘텀 강도
        amp = rng.uniform(0.6, 1.8)                # 계절 진폭
        phase = rng.uniform(0, P)
        trend = rng.normal(0, 0.004)               # 노드별 추세
        d0 = rng.uniform(0.3, 1.2); k = rng.uniform(1.2, 2.2)
        d = rng.normal(0, 0.5)
        for t in range(T):
            shock = rng.normal(0, 0.25) + (rng.random() < 0.03) * rng.uniform(1.0, 2.5)
            d = a * d + amp * np.sin(2 * np.pi * (t + phase) / P) + trend * t + shock
            load = 1 / (1 + np.exp(-k * (d - d0)))            # 포화 비선형
            loads[i, t] = np.clip(load + rng.normal(0, 0.02), 0, 1)
    return loads


def windows(loads):
    Xin, Yout, G = [], [], []
    tvec = np.arange(T)
    sin = np.sin(2 * np.pi * tvec / P).astype(np.float32)
    cos = np.cos(2 * np.pi * tvec / P).astype(np.float32)
    for i in range(N_NODES):
        for t in range(N_IN, T - H + 1, STRIDE):
            sl = slice(t - N_IN, t)
            step = np.stack([loads[i, sl], sin[sl], cos[sl]], 1)   # [N_IN,3]
            Xin.append(step); Yout.append(loads[i, t:t + H].copy()); G.append(i)
    return np.stack(Xin), np.stack(Yout), np.array(G)


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
    ytr = torch.tensor(Ytr)
    net = SeqForecast(kind); opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    lossf = nn.SmoothL1Loss(); n = len(xtr); bs = 256
    for _ in range(epochs):
        net.train(); idx = torch.randperm(n)
        for i in range(0, n, bs):
            b = idx[i:i + bs]; opt.zero_grad(); lossf(net(xtr[b]), ytr[b]).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(xte).numpy()


def metrics(Ytrue, Ypred):
    per_h = [round(float(mean_squared_error(Ytrue[:, h], Ypred[:, h]) ** 0.5), 4) for h in range(H)]
    over = (Ytrue.max(1) > OVERLOAD).astype(int)
    score = Ypred.max(1)
    auc = float(roc_auc_score(over, score)) if 0 < over.mean() < 1 else float("nan")
    return {"mae": round(float(mean_absolute_error(Ytrue, Ypred)), 4),
            "rmse": round(float(mean_squared_error(Ytrue, Ypred) ** 0.5), 4),
            "rmse_per_h": per_h, "overload_auc": round(auc, 4),
            "overload_rate": round(float(over.mean()), 4)}


def main():
    print("[1] 다노드 부하 시뮬(모멘텀+포화+계절+충격)…")
    loads = simulate_nodes()
    X, Y, G = windows(loads)
    print(f"  노드 {N_NODES} | 스텝 {T} | 윈도우 {len(X):,} | 입력 {X.shape}→출력 {Y.shape} | "
          f"과부하(>{OVERLOAD}) 윈도우 {(Y.max(1)>OVERLOAD).mean():.1%}")

    uniq = np.unique(G); np.random.shuffle(uniq)
    tr_u = set(uniq[:int(len(uniq) * 0.75)])
    trm = np.array([g in tr_u for g in G]); tem = ~trm
    Xtr, Xte = X[trm], X[tem]; Ytr, Yte = Y[trm], Y[tem]
    Xtr_f = Xtr.reshape(len(Xtr), -1); Xte_f = Xte.reshape(len(Xte), -1)
    print(f"  train {trm.sum():,} / test {tem.sum():,} (노드 단위 분할)")

    res = {}
    print("[2] Tabular (lag flatten 멀티아웃풋)")
    tab = {"Ridge": Ridge(alpha=1.0),
           "GBM": MultiOutputRegressor(GradientBoostingRegressor(random_state=SEED)),
           "MLP": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=SEED)}
    for name, m in tab.items():
        m.fit(Xtr_f, Ytr); pred = m.predict(Xte_f).clip(0, 1)
        res[name] = metrics(Yte, pred)
        print(f"  {name:7s} RMSE {res[name]['rmse']:.4f} MAE {res[name]['mae']:.4f} "
              f"| overloadAUC {res[name]['overload_auc']:.4f} | per-h {res[name]['rmse_per_h']}")

    print("[3] 시퀀스 DL")
    for kind, disp in [("lstm", "LSTM"), ("gru", "GRU"), ("transformer", "Transformer")]:
        pred = train_dl(kind, Xtr, Ytr, Xte).clip(0, 1)
        res[disp] = metrics(Yte, pred)
        print(f"  {disp:11s} RMSE {res[disp]['rmse']:.4f} MAE {res[disp]['mae']:.4f} "
              f"| overloadAUC {res[disp]['overload_auc']:.4f} | per-h {res[disp]['rmse_per_h']}")

    naive = np.repeat(Xte[:, -1, 0:1], H, axis=1)               # 마지막 load 지속
    res["naive_persist"] = metrics(Yte, naive)
    print(f"  naive(지속) RMSE {res['naive_persist']['rmse']:.4f} | overloadAUC {res['naive_persist']['overload_auc']:.4f}")

    json.dump({"n_nodes": N_NODES, "T": T, "N_IN": N_IN, "H": H, "n_windows": int(len(X)),
               "overload_thr": OVERLOAD, "results": res},
              open(os.path.join(OUT, "forecast_node.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("저장 → sample_project/outputs/realtime/forecast_node.json")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        order = ["Ridge", "GBM", "MLP", "LSTM", "GRU", "Transformer"]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
        for m in order: a1.plot(range(1, H + 1), res[m]["rmse_per_h"], "-o", ms=3, label=m)
        a1.plot(range(1, H + 1), res["naive_persist"]["rmse_per_h"], "--", c="grey", label="naive")
        a1.set_xlabel("forecast horizon (step ahead)"); a1.set_ylabel("RMSE (frac of capacity)")
        a1.legend(fontsize=8); a1.set_title("Multi-node load forecast: RMSE by horizon")
        x = np.arange(len(order))
        a2.bar(x, [res[m]["overload_auc"] for m in order], color=["#3b6fb0"] * 3 + ["#c0504d"] * 3)
        for i, m in enumerate(order):
            a2.text(i, res[m]["overload_auc"] + .003, f"{res[m]['overload_auc']:.3f}", ha="center", fontsize=7)
        a2.set_xticks(x); a2.set_xticklabels(order, rotation=30); a2.set_ylim(0.5, 1)
        a2.axvspan(2.5, 5.5, color="#f3f3f3", zorder=0); a2.set_title("Overload (next-H) detection AUC")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "forecast_node.png"), dpi=110)
        print("저장 → sample_project/outputs/realtime/forecast_node.png")
    except Exception as e:
        print("[plot skip]", e)


if __name__ == "__main__":
    main()

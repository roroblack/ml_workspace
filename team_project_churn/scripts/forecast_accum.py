# -*- coding: utf-8 -*-
"""17-3-2 (B-2). '적분이 필요한' 잠재상태 부하 예측 — RNN/선형이 트리를 이기는 경계.

B(관측가능 부하)에선 lag가 충분통계량이라 GBM이 이겼다. 여기선 상태를 직접 주지 않는다:
  관측 입력 = '증분(inflow, 노이즈 포함)'만.   실제 수준(level) = 증분의 (감쇠)누적.
  level_t = clip(rho·level_{t-1} + g·inflow_t, 0, 1)     ← 적분 + 포화
  타깃 = 향후 H스텝 level / 과부하(level>0.85) 여부.
트리는 '합(적분)'을 분기로 흉내내기 어렵고, 순환(RNN)·선형은 자연히 누적 → 여기서 역전 기대.

산출: sample_project/outputs/realtime/forecast_accum.json + .png
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
N_NODES, T, P = 400, 240, 24
N_IN, H, STRIDE = 24, 12, 6
RHO, G_IN, OVERLOAD = 0.97, 0.05, 0.85
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)


def simulate():
    rng = np.random.default_rng(SEED)
    inflow = np.zeros((N_NODES, T), np.float32)     # 관측(노이즈 포함 증분)
    level = np.zeros((N_NODES, T), np.float32)      # 잠재 수준(미관측)
    for i in range(N_NODES):
        amp = rng.uniform(0.5, 1.6); phase = rng.uniform(0, P); trend = rng.normal(0, 0.02)
        base = rng.uniform(-0.2, 0.4); L = rng.uniform(0, 0.3)
        for t in range(T):
            true_in = base + amp * np.sin(2 * np.pi * (t + phase) / P) + trend * t \
                      + rng.normal(0, 0.4) + (rng.random() < 0.03) * rng.uniform(1.5, 3.5)
            L = float(np.clip(RHO * L + G_IN * true_in, 0, 1))      # 적분 + 포화
            level[i, t] = L
            inflow[i, t] = true_in + rng.normal(0, 0.3)             # 관측은 노이즈 증분
    return inflow, level


def windows(inflow, level):
    tvec = np.arange(T)
    sin = np.sin(2 * np.pi * tvec / P).astype(np.float32); cos = np.cos(2 * np.pi * tvec / P).astype(np.float32)
    Xin, Yout, G = [], [], []
    for i in range(N_NODES):
        for t in range(N_IN, T - H + 1, STRIDE):
            sl = slice(t - N_IN, t)
            Xin.append(np.stack([inflow[i, sl], sin[sl], cos[sl]], 1))   # 관측=증분만
            Yout.append(level[i, t:t + H].copy()); G.append(i)
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
    xtr = torch.tensor((Xtr - mu) / sd); xte = torch.tensor((Xte - mu) / sd); ytr = torch.tensor(Ytr)
    net = SeqForecast(kind); opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    lossf = nn.SmoothL1Loss(); n = len(xtr); bs = 256
    for _ in range(epochs):
        net.train(); idx = torch.randperm(n)
        for i in range(0, n, bs):
            b = idx[i:i + bs]; opt.zero_grad(); lossf(net(xtr[b]), ytr[b]).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(xte).numpy()


def metrics(Yt, Yp):
    per_h = [round(float(mean_squared_error(Yt[:, h], Yp[:, h]) ** 0.5), 4) for h in range(H)]
    over = (Yt.max(1) > OVERLOAD).astype(int); auc = float(roc_auc_score(over, Yp.max(1))) if 0 < over.mean() < 1 else float("nan")
    return {"mae": round(float(mean_absolute_error(Yt, Yp)), 4), "rmse": round(float(mean_squared_error(Yt, Yp) ** 0.5), 4),
            "rmse_per_h": per_h, "overload_auc": round(auc, 4), "overload_rate": round(float(over.mean()), 4)}


def main():
    print("[1] 잠재상태(적분) 부하 시뮬 — 관측=증분만…")
    inflow, level = simulate()
    X, Y, G = windows(inflow, level)
    print(f"  노드 {N_NODES} | 윈도우 {len(X):,} | 입력 {X.shape}→출력 {Y.shape} | 과부하 윈도우 {(Y.max(1)>OVERLOAD).mean():.1%}")
    uniq = np.unique(G); np.random.shuffle(uniq); tr_u = set(uniq[:int(len(uniq) * 0.75)])
    trm = np.array([g in tr_u for g in G]); tem = ~trm
    Xtr, Xte, Ytr, Yte = X[trm], X[tem], Y[trm], Y[tem]
    Xtr_f, Xte_f = Xtr.reshape(len(Xtr), -1), Xte.reshape(len(Xte), -1)
    print(f"  train {trm.sum():,} / test {tem.sum():,}")

    res = {}
    print("[2] Tabular")
    for name, m in {"Ridge": Ridge(alpha=1.0),
                    "GBM": MultiOutputRegressor(GradientBoostingRegressor(random_state=SEED)),
                    "MLP": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=SEED)}.items():
        m.fit(Xtr_f, Ytr); pred = m.predict(Xte_f).clip(0, 1); res[name] = metrics(Yte, pred)
        print(f"  {name:7s} RMSE {res[name]['rmse']:.4f} MAE {res[name]['mae']:.4f} | overloadAUC {res[name]['overload_auc']:.4f}")
    print("[3] 시퀀스 DL")
    for kind, disp in [("lstm", "LSTM"), ("gru", "GRU"), ("transformer", "Transformer")]:
        pred = train_dl(kind, Xtr, Ytr, Xte).clip(0, 1); res[disp] = metrics(Yte, pred)
        print(f"  {disp:11s} RMSE {res[disp]['rmse']:.4f} MAE {res[disp]['mae']:.4f} | overloadAUC {res[disp]['overload_auc']:.4f}")
    naive = np.repeat(Xte[:, -1, 0:1].clip(0, 1), H, axis=1); res["naive_persist"] = metrics(Yte, naive)
    print(f"  naive(증분지속) RMSE {res['naive_persist']['rmse']:.4f} | overloadAUC {res['naive_persist']['overload_auc']:.4f}")

    json.dump({"n_nodes": N_NODES, "N_IN": N_IN, "H": H, "n_windows": int(len(X)),
               "rho": RHO, "overload_thr": OVERLOAD, "results": res},
              open(os.path.join(OUT, "forecast_accum.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("저장 → forecast_accum.json")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        order = ["Ridge", "GBM", "MLP", "LSTM", "GRU", "Transformer"]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
        for m in order: a1.plot(range(1, H + 1), res[m]["rmse_per_h"], "-o", ms=3, label=m)
        a1.set_xlabel("horizon (step ahead)"); a1.set_ylabel("RMSE (level)"); a1.legend(fontsize=8)
        a1.set_title("Integrate-to-level forecast: RMSE by horizon (obs=flow only)")
        x = np.arange(len(order)); a2.bar(x, [res[m]["overload_auc"] for m in order], color=["#3b6fb0"] * 3 + ["#c0504d"] * 3)
        for i, m in enumerate(order): a2.text(i, res[m]["overload_auc"] + .003, f"{res[m]['overload_auc']:.3f}", ha="center", fontsize=7)
        a2.set_xticks(x); a2.set_xticklabels(order, rotation=30); a2.set_ylim(0.5, 1); a2.axvspan(2.5, 5.5, color="#f3f3f3", zorder=0)
        a2.set_title("Overload (next-H) detection AUC")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "forecast_accum.png"), dpi=110); print("저장 → forecast_accum.png")
    except Exception as e:
        print("[plot skip]", e)


if __name__ == "__main__":
    main()

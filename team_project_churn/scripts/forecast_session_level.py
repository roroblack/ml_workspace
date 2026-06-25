# -*- coding: utf-8 -*-
"""17-3-3. REES46 세션 누적 장바구니 level 예측 — 증분→비선형누적 (실데이터 B-2).

DL 승리조건(17-3-2 B-2)을 실제 REES46로 검증:
  관측 입력 = 스텝별 '증분'만 (gap·행동·price·탐색전환) — 러닝 누적은 입력에서 제외.
  타깃 = 부호있는 감쇠누적 장바구니 level (cart/purchase +price, remove_from_cart −price) → 정규화+clip(포화).
  level_t = clip(rho*level_{t-1} + g*inc_t/cap, 0, 1)
→ 상태(level)가 관측 안 되고 증분을 적분해야 + 비선형 → GRU/Transformer가 트리를 이길지 검증.

데이터: REES46 2019-Nov 표본(sample_project, per-event). 세션(user_session) 단위.
산출: sample_project/outputs/realtime/forecast_session.json + .png
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
TYPES = ["view", "cart", "remove_from_cart", "purchase"]
SIGN = {"view": 0.0, "cart": +1.0, "remove_from_cart": -1.0, "purchase": +1.0}
L, H = 8, 4                  # 입력 8스텝 → 출력 4스텝
RHO = 0.85                  # level 감쇠(누적 메모리)
OVERLOAD = 0.85
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
C = 8                       # 증분 피처 수


def load_sessions():
    df = pd.read_csv(RAW, usecols=["user_id", "session_id", "event_type", "category_id", "brand", "price", "event_time"])
    df["event_time"] = pd.to_datetime(df["event_time"])
    df["price"] = df["price"].clip(lower=0).fillna(0)
    df["brand"] = df["brand"].fillna("unknown")
    df = df.sort_values(["session_id", "event_time"])
    cnt = df.groupby("session_id")["event_type"].transform("size")
    df = df[cnt >= L + H]                       # 길이 충분 세션만
    return df


def step_features(s):
    """세션 s → 스텝별 증분피처 [n,C] + 부호증분 inc[n] + price[n]."""
    ts = s["event_time"].values.astype("datetime64[s]").astype(np.int64)
    gap = np.concatenate([[0], np.diff(ts)]).astype(np.float32)
    et = s["event_type"].values
    price = s["price"].values.astype(np.float32)
    cat = s["category_id"].values; br = s["brand"].values
    new_cat = np.concatenate([[0], (cat[1:] != cat[:-1]).astype(np.float32)])
    new_br = np.concatenate([[0], (br[1:] != br[:-1]).astype(np.float32)])
    feat = np.stack([
        np.log1p(gap),
        (et == "view").astype(np.float32), (et == "cart").astype(np.float32),
        (et == "remove_from_cart").astype(np.float32), (et == "purchase").astype(np.float32),
        np.log1p(price), new_cat, new_br], 1).astype(np.float32)   # [n,8]
    inc = np.array([SIGN.get(e, 0.0) for e in et], np.float32) * price
    return feat, inc


def level_of(inc, g, cap):
    lv = 0.0; out = np.empty(len(inc), np.float32)
    for i, x in enumerate(inc):
        lv = min(max(RHO * lv + g * x / cap, 0.0), 1.0)
        out[i] = lv
    return out


def calibrate_g(sess_incs, cap, target=0.30):
    """train 세션들의 스텝 level overload 비율이 target 근처가 되도록 g 선택."""
    best = (None, 1e9)
    for g in [0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0]:
        rate = np.mean([(level_of(inc, g, cap) > OVERLOAD).mean() for inc in sess_incs])
        if abs(rate - target) < best[1]:
            best = (g, abs(rate - target), rate)
    return best[0], best[2]


class SeqForecast(nn.Module):
    def __init__(self, kind, f=C, h=48, horizon=H):
        super().__init__(); self.kind = kind
        if kind == "lstm": self.rnn = nn.LSTM(f, h, batch_first=True)
        elif kind == "gru": self.rnn = nn.GRU(f, h, batch_first=True)
        elif kind == "transformer":
            self.proj = nn.Linear(f, h); self.pos = nn.Parameter(torch.zeros(1, L, h))
            enc = nn.TransformerEncoderLayer(h, nhead=4, dim_feedforward=96, batch_first=True, dropout=0.1)
            self.rnn = nn.TransformerEncoder(enc, num_layers=2)
        self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, horizon))

    def forward(self, x):
        if self.kind == "transformer":
            return self.head((self.rnn(self.proj(x) + self.pos)).mean(1))
        o, _ = self.rnn(x); return self.head(o[:, -1])


def train_dl(kind, Xtr, Ytr, Xte, epochs=40):
    torch.manual_seed(SEED)
    mu = Xtr.reshape(-1, C).mean(0); sd = Xtr.reshape(-1, C).std(0) + 1e-6
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
    over = (Yt.max(1) > OVERLOAD).astype(int)
    auc = float(roc_auc_score(over, Yp.max(1))) if 0 < over.mean() < 1 else float("nan")
    return {"mae": round(float(mean_absolute_error(Yt, Yp)), 4), "rmse": round(float(mean_squared_error(Yt, Yp) ** 0.5), 4),
            "rmse_per_h": per_h, "overload_auc": round(auc, 4), "overload_rate": round(float(over.mean()), 4)}


def main():
    print("[1] REES46 세션 로드(길이>=%d)…" % (L + H))
    df = load_sessions()
    sess = list(df.groupby("session_id"))
    uid_of = {sid: s["user_id"].iloc[0] for sid, s in sess}
    print(f"  세션 {len(sess):,} | 이벤트 {len(df):,} | 유저 {df.user_id.nunique():,}")

    # 유저 단위 분할(누수 차단) → cap·g 캘리브는 train만
    users = df["user_id"].unique(); np.random.shuffle(users)
    tr_u = set(users[:int(len(users) * 0.75)])
    feats = {sid: step_features(s) for sid, s in sess}
    cap = float(np.percentile(df[df.user_id.isin(tr_u)]["price"].clip(lower=0).replace(0, np.nan).dropna(), 90))
    tr_incs = [feats[sid][1] for sid, _ in sess if uid_of[sid] in tr_u]
    g, rate = calibrate_g(tr_incs, cap)
    print(f"  캘리브: cap(train price p90)={cap:.2f} | g={g} | RHO={RHO} → train step-overload {rate:.1%}")

    Xtr, Ytr, Xte, Yte = [], [], [], []
    for sid, s in sess:
        feat, inc = feats[sid]; lv = level_of(inc, g, cap); n = len(s)
        is_tr = uid_of[sid] in tr_u
        for t in range(L, n - H + 1):
            x = feat[t - L:t]; y = lv[t:t + H]
            (Xtr if is_tr else Xte).append(x); (Ytr if is_tr else Yte).append(y)
    Xtr, Ytr, Xte, Yte = map(lambda a: np.stack(a).astype(np.float32), (Xtr, Ytr, Xte, Yte))
    print(f"  윈도우 train {len(Xtr):,} / test {len(Xte):,} | 입력 {Xtr.shape}→출력 {Ytr.shape} | "
          f"test 과부하윈도우 {(Yte.max(1)>OVERLOAD).mean():.1%}")
    Xtr_f, Xte_f = Xtr.reshape(len(Xtr), -1), Xte.reshape(len(Xte), -1)

    res = {}
    print("[2] Tabular (증분 lag flatten)")
    for name, m in {"Ridge": Ridge(alpha=1.0),
                    "GBM": MultiOutputRegressor(GradientBoostingRegressor(random_state=SEED)),
                    "MLP": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=SEED)}.items():
        m.fit(Xtr_f, Ytr); pred = m.predict(Xte_f).clip(0, 1); res[name] = metrics(Yte, pred)
        print(f"  {name:7s} RMSE {res[name]['rmse']:.4f} MAE {res[name]['mae']:.4f} | overloadAUC {res[name]['overload_auc']:.4f}")
    print("[3] 시퀀스 DL")
    for kind, disp in [("lstm", "LSTM"), ("gru", "GRU"), ("transformer", "Transformer")]:
        pred = train_dl(kind, Xtr, Ytr, Xte).clip(0, 1); res[disp] = metrics(Yte, pred)
        print(f"  {disp:11s} RMSE {res[disp]['rmse']:.4f} MAE {res[disp]['mae']:.4f} | overloadAUC {res[disp]['overload_auc']:.4f}")
    naive = np.tile(Ytr.mean(0), (len(Yte), 1)); res["naive_climatology"] = metrics(Yte, naive)
    print(f"  naive(기후값) RMSE {res['naive_climatology']['rmse']:.4f} | overloadAUC {res['naive_climatology']['overload_auc']:.4f}")

    json.dump({"dataset": "REES46 2019-Nov 표본(sample_project)", "unit": "user_session",
               "L": L, "H": H, "C": C, "rho": RHO, "g": g, "cap_price_p90": round(cap, 2),
               "n_sessions": len(sess), "n_windows_tr": int(len(Xtr)), "n_windows_te": int(len(Xte)),
               "test_overload_rate": round(float((Yte.max(1) > OVERLOAD).mean()), 4), "results": res},
              open(os.path.join(OUT, "forecast_session.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("저장 → forecast_session.json")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        order = ["Ridge", "GBM", "MLP", "LSTM", "GRU", "Transformer"]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
        for m in order: a1.plot(range(1, H + 1), res[m]["rmse_per_h"], "-o", ms=4, label=m)
        a1.set_xlabel("horizon (step ahead)"); a1.set_ylabel("RMSE (basket level)"); a1.legend(fontsize=8)
        a1.set_title("REES46 session basket-level forecast: RMSE by horizon (obs=increments only)")
        x = np.arange(len(order)); a2.bar(x, [res[m]["overload_auc"] for m in order], color=["#3b6fb0"] * 3 + ["#c0504d"] * 3)
        for i, m in enumerate(order): a2.text(i, res[m]["overload_auc"] + .003, f"{res[m]['overload_auc']:.3f}", ha="center", fontsize=7)
        a2.set_xticks(x); a2.set_xticklabels(order, rotation=30); a2.set_ylim(0.5, 1); a2.axvspan(2.5, 5.5, color="#f3f3f3", zorder=0)
        a2.set_title("High-intent (level>0.85, next-H) AUC")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "forecast_session.png"), dpi=110); print("저장 → forecast_session.png")
    except Exception as e:
        print("[plot skip]", e)


if __name__ == "__main__":
    main()

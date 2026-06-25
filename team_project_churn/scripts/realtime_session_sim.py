# -*- coding: utf-8 -*-
"""17-2. 초단위 실시간 세션-이탈 예측 + 이탈방지 프로토콜 (합성 다수유저 스트림).

배경: 실제 REES46는 '초 단위' 로깅이지만 유저당 중앙 2이벤트/1세션으로 희소 →
  초단위 '장기 churn'은 모집단 규모로 학습 불가. 대신 실시간 개입이 의미있는 타깃은
  '세션 이탈(무구매 이탈) 임박' 이다. 이를 보이려고 REES46의 측정 간격분포에 맞춰
  다수 유저가 동시에 브라우징하는 초단위 스트림을 합성하고, 매 이벤트 실시간 스코어 →
  임박 이탈 탐지 → 개입 프로토콜 발동을 시뮬레이션한다.

산출: sample_project/outputs/realtime/session_sim.json + sessions_demo.png
"""
import os, sys, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # repo root(team_project_churn)
SP = os.path.join(HERE, "sample_project")
RAW = os.path.join(SP, "data", "raw", "events.csv")
OUT = os.path.join(SP, "outputs", "realtime"); os.makedirs(OUT, exist_ok=True)
SEED = 42
N_SESSIONS = 6000
SESSION_TIMEOUT = 1800        # 30분 무활동 = 세션 종료(업계 표준)
H_LABEL = 120                 # '임박 이탈' 라벨 창: 이탈세션 마지막 120초 내 이벤트 = 1
L_SEQ = 12                    # 시퀀스 모델 입력: 최근 L_SEQ 이벤트
FEATS = ["step", "elapsed", "last_gap", "mean_gap", "gap_ratio",
         "n_view", "n_cart", "has_cart", "sec_since_cart", "view_run"]
np.random.seed(SEED); torch.manual_seed(SEED)


# ---------- 0. 실제 REES46 간격 측정(시뮬레이터 캘리브레이션 근거) ----------
def measure_real():
    df = pd.read_csv(RAW, usecols=["user_id", "event_time"]).sort_values(["user_id", "event_time"])
    df["event_time"] = pd.to_datetime(df["event_time"])
    g = df.groupby("user_id")["event_time"].diff().dt.total_seconds().dropna()
    intra = g[g <= SESSION_TIMEOUT]               # 세션 내 간격
    return {"p25": float(intra.quantile(.25)), "p50": float(intra.quantile(.5)),
            "p75": float(intra.quantile(.75)), "p90": float(intra.quantile(.9)),
            "share_le60": float((g <= 60).mean()), "share_gt30m": float((g > 1800).mean())}


# ---------- 1. 합성 초단위 다수유저 스트림 ----------
def simulate():
    rng = np.random.default_rng(SEED)
    rows = []   # sid, t(절대초), etype, gap
    purch = {}
    horizon = 6 * 3600
    for sid in range(N_SESSIONS):
        t = float(rng.uniform(0, horizon))
        # 세션 내 간격: 측정치(중앙 ~18s, p75 ~61s, p90 ~168s)에 맞춘 lognormal
        mu, sig = np.log(18), 1.5
        n_view = n_cart = 0; purchased = False; step = 0; prev_t = None
        engaged = rng.random() < 0.45            # 구매성향 유저 비율
        while True:
            if step == 0:
                etype = "view"
            else:
                if n_cart > 0 and rng.random() < (0.35 if engaged else 0.12):
                    etype = "purchase"
                elif rng.random() < (0.30 if engaged else 0.12):
                    etype = "cart"
                else:
                    etype = "view"
            gap = 0.0 if prev_t is None else float(rng.lognormal(mu, sig))
            t += gap; prev_t = t
            n_view += etype == "view"; n_cart += etype == "cart"
            rows.append([sid, t, etype, gap])
            if etype == "purchase":
                purchased = True; break          # 구매 = 긍정 종료
            step += 1
            # 이탈 확률(이벤트마다): 길어질수록↑, 뷰만 쌓이면↑, 카트 있으면↓
            p_leave = 0.10 + 0.015 * step + (0.07 if (n_view >= 3 and n_cart == 0) else 0) \
                      - (0.06 if n_cart > 0 else 0) - (0.05 if engaged else 0)
            p_leave = min(max(p_leave, 0.02), 0.9)
            if rng.random() < p_leave or step > 40:
                break                            # 무구매 이탈 종료
        purch[sid] = purchased
    ev = pd.DataFrame(rows, columns=["sid", "t", "etype", "gap"])
    ev["sess_purch"] = ev["sid"].map(purch)
    return ev


# ---------- 2. 실시간 러닝 피처 + 임박-이탈 라벨 ----------
def featurize(ev):
    """러닝 집계 피처(ML용) + 최근 L_SEQ 이벤트 시퀀스(DL용)를 같은 행 순서로 생성."""
    out = []; seqs = []
    for sid, s in ev.groupby("sid"):
        s = s.sort_values("t"); ts = s["t"].values; types = s["etype"].values
        t0 = ts[0]; t_last = ts[-1]; abandon = not bool(s["sess_purch"].iloc[0])
        gap_arr = np.concatenate([[0.0], np.diff(ts)])               # 이벤트별 직전 간격
        is_view = (types == "view").astype(np.float32)
        is_cart = (types == "cart").astype(np.float32)
        step_feat = np.stack([np.log1p(gap_arr), is_view, is_cart], 1).astype(np.float32)  # [n,3]
        n_view = n_cart = 0; gaps = []; view_run = 0
        for i in range(len(s)):
            et = types[i]; gap = gap_arr[i]
            if i > 0: gaps.append(gap)
            n_view += et == "view"; n_cart += et == "cart"
            view_run = view_run + 1 if et == "view" else 0
            mg = float(np.mean(gaps)) if gaps else 0.0
            label = int(abandon and (t_last - ts[i]) <= H_LABEL)     # 무구매 세션 마지막 120초 내 = 1
            out.append({
                "sid": sid, "t": ts[i], "abandon": int(abandon), "is_last": int(i == len(s) - 1),
                "step": i, "elapsed": ts[i] - t0, "last_gap": gap, "mean_gap": mg,
                "gap_ratio": gap / (mg + 1), "n_view": n_view, "n_cart": n_cart,
                "has_cart": int(n_cart > 0),
                "sec_since_cart": (ts[i] - (ts[:i + 1][types[:i + 1] == "cart"][-1])
                                   if (types[:i + 1] == "cart").any() else ts[i] - t0),
                "view_run": view_run, "label": label,
            })
            # 최근 L_SEQ 이벤트(좌측 0-패딩)
            w = step_feat[max(0, i - L_SEQ + 1):i + 1]
            pad = np.zeros((L_SEQ - len(w), 3), np.float32)
            seqs.append(np.concatenate([pad, w], 0))
    return pd.DataFrame(out), np.stack(seqs)


# ---------- 시퀀스 DL 모델 (최근 L_SEQ 이벤트) ----------
class SeqNet(nn.Module):
    def __init__(self, kind, f=3, h=32):
        super().__init__(); self.kind = kind
        if kind == "lstm": self.rnn = nn.LSTM(f, h, batch_first=True)
        elif kind == "gru": self.rnn = nn.GRU(f, h, batch_first=True)
        elif kind == "transformer":
            self.proj = nn.Linear(f, h); self.pos = nn.Parameter(torch.zeros(1, L_SEQ, h))
            enc = nn.TransformerEncoderLayer(h, nhead=4, dim_feedforward=64, batch_first=True, dropout=0.1)
            self.rnn = nn.TransformerEncoder(enc, num_layers=2)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(h, 1))

    def forward(self, x):
        if self.kind == "transformer":
            return self.head((self.rnn(self.proj(x) + self.pos)).mean(1)).squeeze(1)
        o, _ = self.rnn(x); return self.head(o[:, -1]).squeeze(1)


def train_dl(kind, Xtr, ytr, Xte, epochs=20):
    torch.manual_seed(SEED)
    mu = Xtr.reshape(-1, 3).mean(0); sd = Xtr.reshape(-1, 3).std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    xtr = torch.tensor(Xtr); ytr_t = torch.tensor(ytr, dtype=torch.float32); xte = torch.tensor(Xte)
    pos_w = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float32)
    net = SeqNet(kind); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w); n = len(xtr); bs = 512
    for _ in range(epochs):
        net.train(); idx = torch.randperm(n)
        for i in range(0, n, bs):
            b = idx[i:i + bs]; opt.zero_grad()
            lossf(net(xtr[b]), ytr_t[b]).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(xte)).numpy()


# ---------- 3. 이탈방지 프로토콜 시뮬: 케이던스별 탐지/리드타임 ----------
def run_protocol(feats_test, clf, thr, cadence):
    """cadence: 'event'(이벤트구동) 또는 초단위 정수(고정 tick).
    각 이탈세션을 prob>thr 처음 넘는 시점에 '플래그' → 마지막 이벤트까지를 리드타임으로."""
    leads = []; flagged_ab = total_ab = fp = total_cv = scorings = 0
    Xall = feats_test[FEATS].values
    feats_test = feats_test.assign(_p=clf.predict_proba(Xall)[:, 1])
    for sid, s in feats_test.groupby("sid"):
        s = s.sort_values("t").reset_index(drop=True)
        abandon = bool(s["abandon"].iloc[0]); t_last = s["t"].iloc[-1]
        if abandon: total_ab += 1
        else: total_cv += 1
        # 케이던스에 따라 평가 시점 선택
        if cadence == "event":
            idxs = list(range(len(s))); scorings += len(s)
        else:
            ticks = np.arange(s["t"].iloc[0], t_last + 1e-6, cadence)
            idxs = []
            for tk in ticks:
                j = s.index[s["t"] <= tk]
                if len(j): idxs.append(int(j[-1]))
            idxs = sorted(set(idxs)); scorings += len(ticks)
        flag_t = None
        for j in idxs:
            if s["_p"].iloc[j] >= thr:
                flag_t = s["t"].iloc[j]; break
        if flag_t is not None:
            if abandon:
                flagged_ab += 1; leads.append(t_last - flag_t)
            else:
                fp += 1
    recall = flagged_ab / max(total_ab, 1)
    precision = flagged_ab / max(flagged_ab + fp, 1)
    return {"recall": round(recall, 3), "precision": round(precision, 3),
            "median_lead_s": round(float(np.median(leads)) if leads else 0, 1),
            "p25_lead_s": round(float(np.percentile(leads, 25)) if leads else 0, 1),
            "scorings_per_session": round(scorings / (total_ab + total_cv), 2),
            "n_abandon": total_ab, "n_convert": total_cv}


def main():
    real = measure_real()
    print("[실제 REES46 간격 측정] 세션내 중앙 {p50:.0f}s p75 {p75:.0f}s p90 {p90:.0f}s | "
          "60s이내 {share_le60:.0%} | 30분초과(세션경계) {share_gt30m:.0%}".format(**real))

    print("[1] 합성 초단위 스트림 생성…")
    ev = simulate()
    nses = ev["sid"].nunique(); ab_rate = 1 - ev.groupby("sid")["sess_purch"].first().mean()
    print(f"  세션 {nses:,} | 이벤트 {len(ev):,} | 무구매 이탈률 {ab_rate:.1%} | "
          f"세션당 이벤트 중앙 {int(ev.groupby('sid').size().median())}")

    print("[2] 러닝 피처 + 임박이탈 라벨 + 시퀀스 생성…")
    F, SEQ = featurize(ev)
    print(f"  이벤트표본 {len(F):,} | 임박이탈(양성) {F.label.mean():.1%} | 시퀀스 {SEQ.shape}")

    # 세션 단위 분할(누수 차단) — F 행과 SEQ 행은 같은 순서
    rng = np.random.default_rng(SEED); sids = ev["sid"].unique(); rng.shuffle(sids)
    cut = int(len(sids) * 0.7); tr_s = set(sids[:cut])
    trm = F.sid.isin(tr_s).values; tem = ~trm
    tr = F[trm]; te = F[tem]
    Xtr, ytr = tr[FEATS].values, tr["label"].values
    Xte, yte = te[FEATS].values, te["label"].values

    print("[3] 실시간 스코어러 비교 (이벤트단위 임박이탈 예측) — ML 3 + DL 3")
    aucs = {}                                  # name -> (auc, pr)
    pred_for_protocol = None
    ml = {"LogReg": (LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED), True),
          "GBM": (GradientBoostingClassifier(random_state=SEED), False),
          "MLP": (MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=SEED), True)}
    for name, (clf, scale) in ml.items():
        Xa, Xb = Xtr, Xte
        if scale:
            sc = StandardScaler().fit(Xtr); Xa, Xb = sc.transform(Xtr), sc.transform(Xte)
        clf.fit(Xa, ytr); p = clf.predict_proba(Xb)[:, 1]
        aucs[name] = (round(roc_auc_score(yte, p), 4), round(average_precision_score(yte, p), 4))
        print(f"  [ML] {name:11s} AUC {aucs[name][0]:.4f} | PR {aucs[name][1]:.4f}")
        if name == "GBM": gbm_clf = clf
    for kind, disp in [("lstm", "LSTM"), ("gru", "GRU"), ("transformer", "Transformer")]:
        p = train_dl(kind, SEQ[trm], ytr, SEQ[tem])
        aucs[disp] = (round(roc_auc_score(yte, p), 4), round(average_precision_score(yte, p), 4))
        print(f"  [DL] {disp:11s} AUC {aucs[disp][0]:.4f} | PR {aucs[disp][1]:.4f}")
    scorer = gbm_clf                            # 프로토콜은 경량·저지연 GBM 사용

    # 모델 비교 막대그래프
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        order = ["LogReg", "GBM", "MLP", "LSTM", "GRU", "Transformer"]
        x = np.arange(len(order)); w = 0.38
        fig, ax = plt.subplots(figsize=(9, 4))
        b1 = ax.bar(x - w / 2, [aucs[m][0] for m in order], w, label="ROC-AUC", color="#3b6fb0")
        b2 = ax.bar(x + w / 2, [aucs[m][1] for m in order], w, label="PR-AUC", color="#e08a3c")
        for bb in (b1, b2):
            for r in bb: ax.text(r.get_x() + r.get_width() / 2, r.get_height() + .005,
                                 f"{r.get_height():.3f}", ha="center", va="bottom", fontsize=7)
        for i in range(3, len(order)):  # DL 영역 음영
            ax.axvspan(i - 0.5, i + 0.5, color="#f3f3f3", zorder=0)
        ax.set_xticks(x); ax.set_xticklabels(order); ax.set_ylim(0, 1)
        ax.set_ylabel("score (test)"); ax.set_title("Real-time session-abandonment scoring — model comparison")
        ax.text(1, 0.95, "ML (tabular running feats)", ha="center", fontsize=8)
        ax.text(4, 0.95, "DL (recent-event sequence)", ha="center", fontsize=8)
        ax.legend(loc="lower right"); fig.tight_layout()
        fig.savefig(os.path.join(OUT, "model_compare.png"), dpi=110)
        print("  저장 → sample_project/outputs/realtime/model_compare.png")
    except Exception as e:
        print("  [plot skip]", e)

    print("[4] 이탈방지 프로토콜 — 케이던스(예측 간격)별 탐지/리드타임  (thr=0.5)")
    cadences = [("event(이벤트구동)", "event"), ("5초 tick", 5), ("30초 tick", 30),
                ("60초 tick", 60), ("300초 tick", 300)]
    proto = {}
    for label, cad in cadences:
        r = run_protocol(te, scorer, 0.5, cad)
        proto[label] = r
        print(f"  {label:16s} recall {r['recall']:.2f} prec {r['precision']:.2f} "
              f"| 리드타임 중앙 {r['median_lead_s']:.0f}s (p25 {r['p25_lead_s']:.0f}s) "
              f"| 스코어링 {r['scorings_per_session']:.1f}/세션")

    out = {"real_gap_stats": real, "n_sessions": int(nses), "n_events": int(len(ev)),
           "abandon_rate": round(float(ab_rate), 4), "label_positive_rate": round(float(F.label.mean()), 4),
           "session_timeout_s": SESSION_TIMEOUT, "label_horizon_s": H_LABEL,
           "model_auc": {k: {"auc": v[0], "pr": v[1]} for k, v in aucs.items()},
           "protocol": proto}
    json.dump(out, open(os.path.join(OUT, "session_sim.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n저장 → sample_project/outputs/realtime/session_sim.json")

    # 데모 플롯: 이탈세션 3개의 실시간 prob 궤적
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        te2 = te.assign(_p=scorer.predict_proba(te[FEATS].values)[:, 1])
        demo = [s for s in te["sid"].unique() if te[(te.sid == s)]["abandon"].iloc[0] == 1][:3]
        fig, ax = plt.subplots(figsize=(10, 4))
        for s in demo:
            d = te2[te2.sid == s].sort_values("t")
            ax.plot(d["elapsed"], d["_p"], "-o", ms=4, label=f"session {s}")
        ax.axhline(0.5, ls="--", c="grey", label="intervention threshold")
        ax.set_ylim(0, 1)
        ax.set_xlabel("session elapsed time (sec)"); ax.set_ylabel("real-time abandonment probability")
        ax.set_title("Real-time session-abandonment scoring (example abandoning sessions)")
        ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(OUT, "sessions_demo.png"), dpi=110)
        print("저장 → sample_project/outputs/realtime/sessions_demo.png")
    except Exception as e:
        print("[plot skip]", e)


if __name__ == "__main__":
    main()

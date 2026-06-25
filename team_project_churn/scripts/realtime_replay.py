# -*- coding: utf-8 -*-
"""실시간 이탈확률 리플레이.
한 유저의 전체 이벤트 로그를 '지금 실시간으로 발생하는 스트림'으로 가정하고,
매 이벤트 시점마다 '직전 14일 창'으로 시퀀스를 다시 만들어 LSTM 이탈확률을 재계산한다.
=> 활동이 살아있으면 확률↓, 활동이 끊기면(공백 누적) 확률↑ 로 '살아 움직이는' 위험도.

사용:  python realtime_replay.py [user_id ...]
산출:  sample_project/outputs/realtime/<uid>_trajectory.csv,  realtime_replay_summary.md
"""
import os, sys, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # torch+matplotlib OpenMP 충돌 우회
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, torch, torch.nn as nn

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # repo root(team_project_churn)
SP = os.path.join(HERE, "sample_project")
MODELS = os.path.join(SP, "models")
RAW = os.path.join(SP, "data", "raw", "events.csv")
OUTDIR = os.path.join(SP, "outputs", "realtime"); os.makedirs(OUTDIR, exist_ok=True)
EVENT_TYPES = ["view", "cart", "purchase"]
SEQ_LEN = 14
RISK_LOW, RISK_HIGH = 0.35, 0.65


class LSTMClf(nn.Module):
    def __init__(self, f, h=32, p=0.3):
        super().__init__()
        self.lstm = nn.LSTM(f, h, batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(p), nn.Linear(h, 1))
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o[:, -1]).squeeze(1)


def load_model():
    meta = json.load(open(os.path.join(MODELS, "lstm_meta.json"), encoding="utf-8"))
    m = LSTMClf(meta["n_features"], meta.get("hidden", 32))
    m.load_state_dict(torch.load(os.path.join(MODELS, "lstm.pth"), map_location="cpu"))
    m.eval()
    z = np.load(os.path.join(MODELS, "seq_scaler.npz"))
    return m, (z["smu"], z["ssd"])


def seq_from(df_hist, now_day):
    """now_day(포함)에서 과거 SEQ_LEN일 창의 일별 [view,cart,purchase] 카운트 → [1,L,3]."""
    L = SEQ_LEN
    X = np.zeros((L, len(EVENT_TYPES)), dtype=np.float32)
    start = now_day - pd.Timedelta(days=L - 1)
    w = df_hist[(df_hist["d"] >= start) & (df_hist["d"] <= now_day)]
    if len(w):
        di = (w["d"] - start).dt.days.clip(0, L - 1)
        for (i, et), c in w.groupby([di, "event_type"]).size().items():
            if et in EVENT_TYPES:
                X[int(i), EVENT_TYPES.index(et)] = c
    return X[None]


def risk(p):
    return "high" if p >= RISK_HIGH else ("medium" if p >= RISK_LOW else "low")


def plot_daily(uid, dv, outpath):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.axhspan(0, RISK_LOW, color="#e7f5e7"); ax.axhspan(RISK_LOW, RISK_HIGH, color="#fff3cd")
        ax.axhspan(RISK_HIGH, 1, color="#f8d7da")
        x = range(len(dv))
        ax.plot(x, dv["churn_prob"], "-o", color="#222", ms=4, lw=1.5, label="churn prob")
        sil = dv[dv["today_events"] == 0]
        ax.scatter([dv.index.get_loc(i) for i in sil.index], sil["churn_prob"],
                   color="#c0392b", zorder=5, s=28, label="silent day")
        ax.set_xticks(list(x)); ax.set_xticklabels([d.strftime("%m-%d") for d in dv["day"]], rotation=90, fontsize=7)
        ax.set_ylim(0, 1); ax.set_ylabel("real-time churn probability")
        ax.set_title(f"user {uid} - real-time churn replay (daily, silence included)")
        ax.legend(loc="upper left", fontsize=8); fig.tight_layout(); fig.savefig(outpath, dpi=110); plt.close(fig)
        return True
    except Exception as e:
        print(f"  [plot skip] {e}"); return False


def replay(uid, df, model, scaler):
    smu, ssd = scaler
    u = df[df["user_id"] == uid].sort_values("event_time").reset_index(drop=True)
    u["d"] = u["event_time"].dt.normalize()
    rows = []
    last_day = None
    for i, r in u.iterrows():
        now = r["d"]
        hist = u.iloc[: i + 1]                      # 지금까지 발생한 로그만(미래 차단)
        X = seq_from(hist, now)
        Xs = (X - smu) / ssd
        with torch.no_grad():
            p = float(torch.sigmoid(model(torch.tensor(Xs, dtype=torch.float32))).item())
        # 직전 이벤트와의 공백(일)
        gap = 0 if last_day is None else (now - last_day).days
        last_day = now
        win = hist[hist["d"] >= now - pd.Timedelta(days=SEQ_LEN - 1)]
        rows.append({
            "step": i + 1, "event_time": r["event_time"], "event_type": r["event_type"],
            "gap_days": gap,
            "win_events": int(len(win)), "win_active_days": int(win["d"].nunique()),
            "win_purchase": int((win["event_type"] == "purchase").sum()),
            "churn_prob": round(p, 4), "risk": risk(p),
        })
    return pd.DataFrame(rows)


def daily_ticks(uid, df, model, scaler, end_day):
    """매 '하루'를 실시간 tick으로 평가 — 이벤트 없는 날(침묵)도 포함.
    침묵이 쌓이면 14일 창의 최근부가 0으로 비어 이탈확률이 오른다(핵심 신호)."""
    smu, ssd = scaler
    u = df[df["user_id"] == uid].copy()
    u["d"] = u["event_time"].dt.normalize()
    start_day = u["d"].min()
    rows = []; silent = 0
    for day in pd.date_range(start_day, end_day, freq="D"):
        hist = u[u["d"] <= day]
        X = seq_from(hist, day)
        Xs = (X - smu) / ssd
        with torch.no_grad():
            p = float(torch.sigmoid(model(torch.tensor(Xs, dtype=torch.float32))).item())
        today = u[u["d"] == day]
        silent = 0 if len(today) else silent + 1
        win = hist[hist["d"] >= day - pd.Timedelta(days=SEQ_LEN - 1)]
        rows.append({
            "day": day, "today_events": int(len(today)), "silent_days": silent,
            "win_events": int(len(win)), "win_active_days": int(win["d"].nunique()),
            "win_purchase": int((win["event_type"] == "purchase").sum()),
            "churn_prob": round(p, 4), "risk": risk(p),
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(RAW)
    df["event_time"] = pd.to_datetime(df["event_time"])
    model, scaler = load_model()

    ids = sys.argv[1:]
    if not ids:
        # 대비되는 두 유형 자동 선택: 끝까지 활발 vs 중간에 끊김
        g = df.groupby("user_id")
        last_day = g["event_time"].max()
        span = (g["event_time"].max() - g["event_time"].min()).dt.total_seconds() / 86400
        n = g.size()
        rich = n[(n >= 25) & (span >= 14)].index
        end = df["event_time"].max().normalize()
        # 끝까지 활발: 마지막 이벤트가 관측 끝 근처
        active = (last_day.loc[rich].dt.normalize() >= end - pd.Timedelta(days=2))
        quiet = (last_day.loc[rich].dt.normalize() <= end - pd.Timedelta(days=9))
        a = last_day.loc[rich][active].index
        q = last_day.loc[rich][quiet].index
        ids = [str(int(n.loc[a].idxmax()))] if len(a) else []
        ids += [str(int(n.loc[q].idxmax()))] if len(q) else []
        print(f"[자동선택] 끝까지 활발형={ids[0] if ids else '-'}, 중간이탈형={ids[1] if len(ids)>1 else '-'}")

    end_day = df["event_time"].max().normalize()      # 관측 종료일(실시간 '현재')
    md = ["# 실시간 이탈확률 리플레이 결과\n",
          "한 유저의 전체 로그를 실시간 스트림으로 가정. **매일(이벤트 없는 침묵일 포함)** 직전 14일 창으로 "
          "LSTM 이탈확률을 재계산 → 활동하면 ↓, 침묵이 쌓이면 ↑ 로 살아 움직이는 위험도.\n"]
    for uid in ids:
        uid_i = int(uid)
        traj = replay(uid_i, df, model, scaler)       # 이벤트 단위(세밀) — CSV로 저장
        if traj.empty:
            print(f"  {uid}: 이벤트 없음 — 스킵"); continue
        traj.to_csv(os.path.join(OUTDIR, f"{uid}_trajectory.csv"), index=False)
        dv = daily_ticks(uid_i, df, model, scaler, end_day)   # 일별 tick(침묵 포함)
        dv.to_csv(os.path.join(OUTDIR, f"{uid}_daily.csv"), index=False)
        png = os.path.join(OUTDIR, f"{uid}_daily.png")
        plot_daily(uid_i, dv, png)
        first, last = dv.iloc[0], dv.iloc[-1]
        peak = dv.loc[dv["churn_prob"].idxmax()]
        # 위험등급 전환점
        dv["risk_prev"] = dv["risk"].shift()
        trans = dv[dv["risk"] != dv["risk_prev"]].iloc[1:]

        print(f"\n=== 유저 {uid} | {dv.day.min().date()}~{dv.day.max().date()} (일별 tick) ===")
        print(f"  시작 {first.churn_prob:.2f}({first.risk}) → 현재 {last.churn_prob:.2f}({last.risk}) | 최고 {peak.churn_prob:.2f}@{peak.day.date()}")
        print("  [일별 실시간 위험도]  (※=침묵일)")
        for _, r in dv.iterrows():
            bar = "█" * int(r.churn_prob * 25)
            tag = f"※침묵{r.silent_days}일" if r.today_events == 0 else f"오늘 {r.today_events:3d}건"
            print(f"   {r.day.date()} | {tag:9s} | 창내{r.win_events:4d}/{r.win_active_days:2d}일 구매{r.win_purchase:2d} "
                  f"| {r.churn_prob:.2f} {r.risk:6s} {bar}")

        md.append(f"\n## 유저 {uid}\n")
        md.append(f"![{uid}](realtime/{uid}_daily.png)\n")
        md.append(f"- 관측 {dv.day.min().date()}~{dv.day.max().date()} · 마지막 활동 {traj.event_time.max().date()}")
        md.append(f"- **시작 {first.churn_prob:.2f}({first.risk}) → 현재 {last.churn_prob:.2f}({last.risk})** (최고 {peak.churn_prob:.2f}@{peak.day.date()})")
        if len(trans):
            md.append(f"- 위험등급 전환: " +
                      ", ".join(f"{r.day.date()} {r.risk_prev}→{r.risk}({r.churn_prob:.2f})"
                                for _, r in trans.iterrows()))
        md.append("\n| 날짜 | 오늘 | 침묵 | 창내 이벤트/활동일 | 구매 | 이탈확률 | 위험 |\n|---|---|---|---|---|---|---|")
        for _, r in dv.iterrows():
            md.append(f"| {r.day.date()} | {r.today_events} | {r.silent_days} | "
                      f"{r.win_events}/{r.win_active_days}일 | {r.win_purchase} | {r.churn_prob:.2f} | {r.risk} |")

    open(os.path.join(OUTDIR, "realtime_replay_summary.md"), "w", encoding="utf-8").write("\n".join(md))
    print(f"\n저장 → {os.path.relpath(OUTDIR, HERE)}/  (*_trajectory.csv, realtime_replay_summary.md)")


if __name__ == "__main__":
    main()

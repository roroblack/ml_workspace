# -*- coding: utf-8 -*-
"""제안 확장 전처리(이벤트/세션 단위 + 바운스 + Δt + 카테고리/브랜드 + 초단위 라벨)가
적용되면 데이터가 어떤 모습인지를 '엑셀 표'처럼 PNG로 시각화. 실제 유저 원본 이벤트로 계산.
산출: output/_proposed_eventlevel_preview.png
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
INSP = os.path.join(HERE, "preprocessing_project", "v2_5m", "output", "sample_user_inspect")
OUT = os.path.join(HERE, "preprocessing_project", "v4_model_prep", "output")
CSV = os.path.join(INSP, "user_396059467_원본이벤트_전체.csv")

SESS_GAP = 1800   # 30분 = 세션 경계/이탈 기준


def build(df):
    df = df.copy()
    df["event_time"] = pd.to_datetime(df["event_time"])
    df = df.sort_values("event_time").reset_index(drop=True)
    df["t"] = df["event_time"].astype("int64") // 10**9
    # 세션 재구성(user_session 사용)
    g = df.groupby("user_session", sort=False)
    df["step_in_session"] = g.cumcount() + 1
    df["dt_prev_sec"] = df.groupby("user_session")["t"].diff().fillna(0).astype(int)
    df["dt_next_sec"] = df.groupby("user_session")["t"].shift(-1) - df["t"]
    df["dt_next_sec"] = df["dt_next_sec"].fillna(SESS_GAP).clip(upper=SESS_GAP).astype(int)
    sess_n = g["t"].transform("size")
    df["is_bounce_session"] = (sess_n == 1).astype(int)          # 세션 내 단일 이벤트 = 바운스
    df["dwell_bucket"] = pd.cut(df["dt_next_sec"], [-1, 3, 30, SESS_GAP],
                                labels=["<3s(오클릭)", "3-30s(탐색)", ">30s(정독)"]).astype(str)
    # 초단위 세션 이탈 라벨: 이 이벤트 뒤 30분 내 추가 활동 없으면 1
    df["y_session_churn"] = (df["dt_next_sec"] >= SESS_GAP).astype(int)
    return df


def main():
    df = build(pd.read_csv(CSV))
    cols = ["event_time", "user_session", "step_in_session", "event_type", "category_id",
            "brand", "price", "dt_prev_sec", "dt_next_sec", "dwell_bucket",
            "is_bounce_session", "y_session_churn"]
    show = df[cols].head(14).copy()
    show["event_time"] = pd.to_datetime(show["event_time"]).dt.strftime("%m-%d %H:%M:%S")
    show["user_session"] = show["user_session"].astype(str).str.slice(0, 8) + "…"
    show["brand"] = show["brand"].astype(str).str.slice(0, 8)
    show["price"] = show["price"].round(2)
    hdr = ["event_time", "session", "step", "type", "cat_id", "brand", "price",
           "dt_prev_s", "dt_next_s", "dwell", "bounce", "y_churn"]

    n = len(show)
    fig, ax = plt.subplots(figsize=(15.5, 0.5 + 0.42 * (n + 2)))
    ax.axis("off")
    ax.set_title("Proposed v4-EXPANDED preprocessing  (event/session grain · Δt · bounce · second-level churn)\n"
                 "real events of user 396059467  —  spreadsheet view", fontsize=12, fontweight="bold", loc="left")
    tbl = ax.table(cellText=show.values, colLabels=hdr, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.2); tbl.scale(1, 1.5)
    ncol = len(hdr)
    for j in range(ncol):                                  # 헤더 스타일
        c = tbl[0, j]; c.set_facecolor("#1f6e43"); c.set_text_props(color="white", fontweight="bold")
    for i in range(1, n + 1):
        bounce = show.iloc[i - 1]["is_bounce_session"]
        ych = show.iloc[i - 1]["y_session_churn"]
        for j in range(ncol):
            cell = tbl[i, j]
            cell.set_facecolor("#f2f7f4" if i % 2 else "#ffffff")
            if hdr[j] == "y_churn" and ych == 1: cell.set_facecolor("#ffd9d2"); cell.set_text_props(fontweight="bold")
            if hdr[j] == "bounce" and bounce == 1: cell.set_facecolor("#fff3cc")
            if hdr[j] == "dwell" and "<3s" in str(show.iloc[i-1]["dwell_bucket"]): cell.set_facecolor("#ffe3e0")
    cap = ("NEW columns vs current v4(10 aggregated cols/user):  session, step_in_session, dt_prev/next_sec, "
           "dwell_bucket, is_bounce_session, y_session_churn(30min)  +  category_id/brand kept.\n"
           "→ 1 row per EVENT (not 1 per user) · full population (no recency<=7 cohort cut) · enables bounce + second-level churn + item/category recommendation.")
    fig.text(0.012, 0.02, cap, fontsize=8.6, color="#333")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    p = os.path.join(OUT, "_proposed_eventlevel_preview.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("saved:", p)
    # 세션 요약 미리보기도 콘솔로
    print(f"events={len(df)} sessions={df.user_session.nunique()} bounce_sessions={df[df.is_bounce_session==1].user_session.nunique()}")


if __name__ == "__main__":
    main()

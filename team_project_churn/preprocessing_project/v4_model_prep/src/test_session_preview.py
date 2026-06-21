# -*- coding: utf-8 -*-
"""데모: 제안한 '이벤트/세션-단위 전처리'를 실제 REES46(Nov)로 돌려 30행 미리보기 이미지 생성.
바운스/초단위 이탈/추천 신호가 어떻게 컬럼으로 나오는지 엑셀 화면처럼 표 이미지로 저장.
산출: output/_preview/session_event_preview.png (+ .csv)
"""
import os, sys, zipfile
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(HERE, "src")
OUT = os.path.join(HERE, "preprocessing_project", "v4_model_prep", "output", "_preview"); os.makedirs(OUT, exist_ok=True)
DT = {"event_type": "category", "category_id": "int64", "brand": "object", "price": "float32", "user_id": "int64", "user_session": "object"}


def main():
    # Nov 첫 청크만(데모용)
    with zipfile.ZipFile(os.path.join(SRC, "2019-Nov.csv.zip")) as zf:
        df = next(pd.read_csv(zf.open("2019-Nov.csv"), usecols=list(DT) + ["event_time"], dtype=DT, chunksize=2_000_000))
    df["event_time"] = pd.to_datetime(df["event_time"].str.slice(0, 19), format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["event_time"]).sort_values(["user_id", "event_time"])

    # 유저단위 Δt(초): 실시간 backend가 보는 신호
    df["t"] = df["event_time"].astype("int64") // 10**9
    g = df.groupby("user_id", sort=False)
    df["dt_prev_s"] = (df["t"] - g["t"].shift(1)).fillna(-1).astype("int64")
    df["dt_next_s"] = (g["t"].shift(-1) - df["t"]).fillna(-1).astype("int64")
    # 세션 집계
    ses = df.groupby("user_session")
    slen = ses["event_type"].transform("size")
    has_purch = ses["event_type"].transform(lambda s: (s == "purchase").any())
    has_cart = ses["event_type"].transform(lambda s: (s == "cart").any())
    df["sess_len"] = slen
    df["bounce"] = (slen == 1).astype(int)                                  # 바운스: 세션 1이벤트
    df["churn30"] = ((df["dt_next_s"] > 1800) | (df["dt_next_s"] < 0)).astype(int)  # 30분+ 무활동=세션이탈(초단위 신호)
    # 세션 유형 선별: 바운스 / 구매완료 / 장바구니이탈
    bounce_s = df[(df.bounce == 1) & (df.event_type == "view")]["user_session"].drop_duplicates().head(5).tolist()
    purch_s = df[has_purch & (slen.between(4, 9))]["user_session"].drop_duplicates().head(2).tolist()
    cartab_s = df[(~has_purch.astype(bool)) & has_cart.astype(bool) & (slen.between(3, 7))]["user_session"].drop_duplicates().head(2).tolist()
    pick = bounce_s + purch_s + cartab_s
    sub = df[df.user_session.isin(pick)].copy()
    # 보기 좋은 컬럼
    sub["idx"] = sub.groupby("user_session").cumcount() + 1
    sub["time"] = sub["event_time"].dt.strftime("%H:%M:%S")
    sub["user"] = sub["user_id"].astype(str).str[-5:]
    sub["session"] = sub["user_session"].str[:6]
    sub["brand"] = sub["brand"].fillna("UNK").astype(str).str[:8]
    sub["price"] = sub["price"].round(2)
    sub["dt_prev_s"] = sub["dt_prev_s"].replace(-1, np.nan)
    sub["dt_next_s"] = sub["dt_next_s"].replace(-1, np.nan)
    sub["stype"] = np.where(sub.user_session.isin(bounce_s), "BOUNCE",
                    np.where(sub.user_session.isin(purch_s), "PURCHASE", "CART-ABANDON"))
    cols = ["stype", "session", "idx", "time", "event_type", "category_id", "brand", "price",
            "dt_prev_s", "dt_next_s", "sess_len", "bounce", "churn30"]
    view = sub[cols].sort_values(["stype", "session", "idx"]).head(30)
    view.to_csv(os.path.join(OUT, "session_event_preview.csv"), index=False, encoding="utf-8-sig")

    # 엑셀 느낌 표 이미지
    fig, ax = plt.subplots(figsize=(15, 9)); ax.axis("off")
    hdr = ["sess_type", "session", "i", "time", "event", "cat_id", "brand", "price",
           "dt_prev(s)", "dt_next(s)", "sess_len", "bounce", "churn30"]
    cell = [[("" if pd.isna(v) else (f"{v:.0f}" if isinstance(v, float) and c in ("dt_prev(s)", "dt_next(s)") else v))
             for v, c in zip(r, hdr)] for r in view.values]
    tbl = ax.table(cellText=cell, colLabels=hdr, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.5)
    # 헤더/유형별 색
    for j in range(len(hdr)):
        tbl[0, j].set_facecolor("#2F5496"); tbl[0, j].get_text().set_color("white"); tbl[0, j].get_text().set_weight("bold")
    color = {"BOUNCE": "#FCE4D6", "PURCHASE": "#E2EFDA", "CART-ABANDON": "#FFF2CC"}
    for i, r in enumerate(view.values, start=1):
        c = color.get(r[0], "white")
        for j in range(len(hdr)): tbl[i, j].set_facecolor(c)
        # churn30/bounce 강조
        if r[cols.index("churn30")] == 1: tbl[i, hdr.index("churn30")].get_text().set_color("#C00000"); tbl[i, hdr.index("churn30")].get_text().set_weight("bold")
        if r[cols.index("bounce")] == 1: tbl[i, hdr.index("bounce")].get_text().set_color("#C00000"); tbl[i, hdr.index("bounce")].get_text().set_weight("bold")
    ax.set_title("제안 전처리(event/session-level) 결과 미리보기 — REES46 2019-Nov 실데이터 30행\n"
                 "bounce=세션1이벤트  churn30=다음활동까지 30분+(초단위 세션이탈)  dt_*=행동간 시간간격(초)",
                 fontsize=11, pad=14, fontproperties=plt.matplotlib.font_manager.FontProperties(family=["Malgun Gothic", "sans-serif"]))
    plt.tight_layout()
    p = os.path.join(OUT, "session_event_preview.png"); fig.savefig(p, dpi=130, bbox_inches="tight")
    print(f"[preview] 세션 {len(pick)}개 | 표 {len(view)}행 | 저장 {p}", flush=True)
    print(view.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

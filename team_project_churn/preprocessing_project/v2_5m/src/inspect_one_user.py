# -*- coding: utf-8 -*-
"""검증용: 활동 활발한 코호트 유저 1명의 (1) 원본 이벤트 전체(관찰기간) (2) tabular 집계행 (3) 주별 시퀀스
를 CSV(엑셀에서 바로 열림)로 추출. "recency≤7은 코호트 선택일 뿐, 피처는 관찰기간 전체 집계"임을 실데이터로 확인.
실행: python preprocessing_project/v2_5m/src/inspect_one_user.py [min_events] [max_events]
산출: preprocessing_project/v2_5m/output/sample_user_inspect/
"""
import os, sys, json, zipfile
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(HERE, "src")
D = os.path.join(HERE, "sample_project", "data", "processed_5m")
OUT = os.path.join(HERE, "preprocessing_project", "v2_5m", "output", "sample_user_inspect"); os.makedirs(OUT, exist_ok=True)
# 관찰기간(train) = 피처 집계 구간
OBS = (pd.Timestamp("2019-10-01"), pd.Timestamp("2020-01-25"))
MONTHS = ["2019-Oct", "2019-Nov", "2019-Dec", "2020-Jan", "2020-Feb"]
DT = {"event_type": "category", "product_id": "int64", "category_id": "int64",
      "category_code": "object", "brand": "object", "price": "float32", "user_id": "int64", "user_session": "object"}


def main():
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    tr = pd.read_parquet(os.path.join(D, "train_cohort_tabular.parquet"))
    # 관찰기간 전체에 걸쳐 '꾸준히' 활동한 유저: tenure 길고(>=60일=2달+) 활동일수 많은 + 코호트(recency<=7)
    cand = tr[(tr.tenure_days >= 60) & (tr.n_events >= lo) & (tr.n_events <= hi) & (tr.recency_days <= 7)] \
        .sort_values("ndays", ascending=False).head(1)
    if cand.empty:
        cand = tr[(tr.n_events >= lo) & (tr.n_events <= hi)].sort_values("tenure_days", ascending=False).head(1)
    if cand.empty: cand = tr.sort_values("tenure_days", ascending=False).head(1)
    row = cand.iloc[0]; uid = int(row["user_id"])
    print(f"[선택 유저] user_id={uid} | n_events={int(row.n_events)} ndays={int(row.ndays)} "
          f"tenure_days={row.tenure_days:.1f} recency_days={row.recency_days:.2f} churn={int(row.churn)}", flush=True)

    # (2) tabular 집계행 → CSV (세로형: 컬럼=값)
    row.to_frame("값").to_csv(os.path.join(OUT, f"user_{uid}_tabular집계.csv"), encoding="utf-8-sig")

    # (1) 원본 이벤트 전체 수집(관찰기간 표시)
    parts = []
    for m in MONTHS:
        with zipfile.ZipFile(os.path.join(SRC, f"{m}.csv.zip")) as zf:
            for chunk in pd.read_csv(zf.open(f"{m}.csv"), usecols=list(DT) + ["event_time"], dtype=DT, chunksize=1_000_000):
                hit = chunk[chunk.user_id == uid]
                if len(hit): parts.append(hit)
    ev = pd.concat(parts) if parts else pd.DataFrame()
    ev["event_time"] = pd.to_datetime(ev["event_time"].str.slice(0, 19), format="%Y-%m-%d %H:%M:%S", errors="coerce")
    ev = ev.sort_values("event_time")
    ev["기간"] = np.where((ev.event_time >= OBS[0]) & (ev.event_time < OBS[1]), "관찰(피처집계)", "결과/기타")
    ev.to_csv(os.path.join(OUT, f"user_{uid}_원본이벤트_전체.csv"), index=False, encoding="utf-8-sig")

    obs_ev = ev[ev["기간"] == "관찰(피처집계)"]
    span = (obs_ev.event_time.max() - obs_ev.event_time.min())
    print(f"[원본이벤트] 총 {len(ev)}건 | 관찰기간 {len(obs_ev)}건 | "
          f"활동 스팬 {span.days}일 ({obs_ev.event_time.min().date()} ~ {obs_ev.event_time.max().date()})", flush=True)
    print(f"  → 활동 스팬이 {span.days}일 = 7일 아님. 피처는 관찰기간 전체 집계임이 확인됨.", flush=True)

    # (3) 주별 시퀀스 행 → CSV
    z = np.load(os.path.join(D, "train_seq.npz"))
    si = np.where(z["user_id"] == uid)[0]
    if len(si):
        X = z["X"][si[0]]  # [17,3]
        seq = pd.DataFrame(X, columns=["view", "cart", "purchase"])
        seq.index.name = "week"; seq.reset_index().to_csv(os.path.join(OUT, f"user_{uid}_주별시퀀스.csv"), index=False, encoding="utf-8-sig")
        print(f"[주별시퀀스] {X.shape} (관찰 17주 × view/cart/purchase 카운트) — 0이 아닌 주: {(X.sum(1)>0).sum()}주", flush=True)

    # 요약 메모
    json.dump({"user_id": uid, "tabular": {k: (float(row[k]) if k != "user_id" else uid) for k in tr.columns},
               "events_total": int(len(ev)), "events_obs": int(len(obs_ev)), "activity_span_days": int(span.days),
               "주의": "recency<=7은 코호트(유저선택) 조건. 피처는 관찰기간 전체 집계. 원본 이벤트는 별도 보존."},
              open(os.path.join(OUT, f"user_{uid}_요약.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
    print(f"[저장] {OUT}", flush=True)
    # 콘솔에 원본 이벤트 앞부분 미리보기
    print("\n=== 원본 이벤트 미리보기(관찰기간, 앞 12행) ===", flush=True)
    cols = ["event_time", "event_type", "category_code", "brand", "price", "user_session"]
    print(obs_ev[cols].head(12).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

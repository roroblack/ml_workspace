# -*- coding: utf-8 -*-
"""실제 REES46 월별 CSV(zip 가능) → 샘플과 동일한 events.csv 형식으로 변환.
CPU/SQLite 부담을 위해 사용자 표본(USER_CAP)으로 자른다.
사용: python src/ingest_rees46.py [REES46_csv_or_zip]   (생략 시 기본 2019-Nov.zip)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import config

USECOLS = ["event_time", "event_type", "product_id", "category_id", "brand", "price", "user_id", "user_session"]
USER_CAP = 8000    # 표본 사용자 수(샘플 데모용; 예측 루프 속도 고려)
DEFAULT = os.path.join(os.path.dirname(config.HERE), "src", "2019-Nov.csv.zip")  # team_project_churn/src/...


def main(path=None):
    path = path or DEFAULT
    if not os.path.exists(path):
        print(f"[ingest] 파일 없음: {path}\n  team_project_churn/src/2019-Nov.csv.zip 를 확인하세요.")
        return False
    print(f"[ingest] 읽는 중: {os.path.basename(path)}")
    df = pd.read_csv(path, usecols=USECOLS)
    df["event_time"] = pd.to_datetime(df["event_time"].astype(str).str.replace(" UTC", "", regex=False),
                                      errors="coerce")
    df = df.dropna(subset=["event_time", "user_id"])
    df["user_id"] = df["user_id"].astype(str)

    # 사용자 표본 추출
    rng = np.random.RandomState(config.SEED)
    uids = df["user_id"].unique()
    if len(uids) > USER_CAP:
        keep = set(rng.choice(uids, USER_CAP, replace=False))
        df = df[df["user_id"].isin(keep)]

    df = df.rename(columns={"user_session": "session_id"})
    out = df[["user_id", "session_id", "event_type", "product_id", "category_id", "brand", "price", "event_time"]].copy()
    out["event_time"] = out["event_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    dst = os.path.join(config.RAW, "events.csv")
    out.sort_values(["user_id", "event_time"]).to_csv(dst, index=False)
    print(f"[ingest] REES46 {len(out):,}행 / 사용자 {out.user_id.nunique():,} → {dst}")
    return True


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

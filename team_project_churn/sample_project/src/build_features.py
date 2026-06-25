# -*- coding: utf-8 -*-
"""이벤트 → 정형 피처 + 일별 시퀀스 생성(파일) + DB 운영 데이터 적재.
관찰 OBS_DAYS / 결과 OUTCOME_DAYS, 결과기간 무활동=이탈(누수 차단)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import config
from db import db_client as db

FEAT_COLS = ["recency_days", "n_view", "n_cart", "n_purchase", "n_events", "active_days", "avg_price"]


def main():
    df = pd.read_csv(os.path.join(config.RAW, "events.csv"))
    df["event_time"] = pd.to_datetime(df["event_time"])
    BASE = df["event_time"].min().normalize()           # 데이터에서 기준일 자동 도출(실/합성 모두 대응)
    CUT = BASE + pd.Timedelta(days=config.OBS_DAYS)
    df["day"] = (df["event_time"].dt.normalize() - BASE).dt.days
    obs = df[df["event_time"] < CUT].copy()
    out = df[df["event_time"] >= CUT]

    users = np.sort(obs["user_id"].unique())
    out_users = set(out["user_id"].unique())
    churn = {u: (0 if u in out_users else 1) for u in users}

    g = obs.groupby("user_id")
    feat = pd.DataFrame(index=users); feat.index.name = "user_id"
    feat["recency_days"] = (CUT - g["event_time"].max()).dt.total_seconds() / 86400
    for et in config.EVENT_TYPES:
        feat[f"n_{et}"] = obs[obs.event_type == et].groupby("user_id").size().reindex(users).fillna(0).values
    feat["n_events"] = g.size().reindex(users).fillna(0)
    feat["active_days"] = g["day"].nunique().reindex(users).fillna(0)
    feat["avg_price"] = g["price"].mean().reindex(users).fillna(0)
    feat["churn"] = [churn[u] for u in users]
    feat.reset_index().to_csv(os.path.join(config.PROC, "features.csv"), index=False)

    # 일별 시퀀스 [N, OBS_DAYS, 3]
    uidx = {u: i for i, u in enumerate(users)}
    X = np.zeros((len(users), config.OBS_DAYS, len(config.EVENT_TYPES)), dtype=np.float32)
    daily = obs.groupby(["user_id", "day", "event_type"]).size()
    for (u, d, et), c in daily.items():
        if 0 <= d < config.OBS_DAYS and et in config.EVENT_TYPES:
            X[uidx[u], d, config.EVENT_TYPES.index(et)] = c
    seq_path = os.path.join(config.PROC, "sequences.npz")
    np.savez_compressed(seq_path, X=X, user_id=users)   # 시퀀스 96% 0 → 압축으로 약 40배 절감

    # ---- DB 적재 (운영 데이터) ----
    db.init_db()
    conn = db.get_conn(); cur = conn.cursor(); ph = "?"
    for t in ["realtime_event_log", "feature_user_snapshot", "sequence_snapshot", "face_user",
              "prediction_log", "retention_action_log", "model_registry"]:
        cur.execute(f"DELETE FROM {t}")
    ev = obs[["user_id", "session_id", "event_type", "product_id", "price", "event_time"]].copy()
    ev["event_time"] = ev["event_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    cur.executemany(
        f"INSERT INTO realtime_event_log(user_id,session_id,event_type,product_id,price,event_time,source) "
        f"VALUES({ph},{ph},{ph},{ph},{ph},{ph},'seed')",
        list(ev.itertuples(index=False, name=None)))
    cur.executemany(
        f"INSERT INTO feature_user_snapshot(user_id,snapshot_time,label,feature_json) VALUES({ph},{ph},{ph},{ph})",
        [(u, CUT.strftime("%Y-%m-%d"), int(churn[u]),
          json.dumps({c: float(feat.loc[u, c]) for c in FEAT_COLS})) for u in users])
    cur.executemany(
        f"INSERT INTO sequence_snapshot(user_id,seq_len,n_features,storage_format,artifact_path,row_index,label) "
        f"VALUES({ph},{ph},{ph},'npz',{ph},{ph},{ph})",
        [(u, config.OBS_DAYS, len(config.EVENT_TYPES), seq_path, uidx[u], int(churn[u])) for u in users])
    cur.executemany(f"INSERT INTO face_user(user_id,display_name,role) VALUES({ph},{ph},{ph})",
                    [(u, f"고객 {u}", "customer") for u in users])
    cur.execute(f"INSERT INTO face_user(user_id,display_name,role) VALUES({ph},{ph},{ph})", ("admin", "관리자", "admin"))
    conn.commit(); conn.close()

    print(f"[build] 사용자 {len(users)} | 이탈률 {feat.churn.mean()*100:.1f}% | "
          f"정형 {len(FEAT_COLS)}피처 | 시퀀스 {X.shape} | DB 이벤트 {len(ev):,} 적재")


if __name__ == "__main__":
    main()

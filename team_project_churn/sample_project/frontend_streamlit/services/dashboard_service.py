# -*- coding: utf-8 -*-
"""대시보드 조회 서비스 — DB에서 요약/그래프 데이터 생성."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
from db import db_client as db


def latest_per_user():
    """유저별 최신 예측 1건."""
    rows = db.query("SELECT * FROM prediction_log ORDER BY prediction_id DESC")
    seen, out = set(), []
    for r in rows:
        if r["user_id"] not in seen:
            seen.add(r["user_id"]); out.append(r)
    return pd.DataFrame(out)


def risk_summary():
    df = latest_per_user()
    if df.empty:
        return {"total": 0, "high": 0, "medium": 0, "low": 0, "avg_prob": 0.0}
    vc = df["risk_level"].value_counts().to_dict()
    return {"total": len(df), "high": int(vc.get("high", 0)), "medium": int(vc.get("medium", 0)),
            "low": int(vc.get("low", 0)), "avg_prob": round(float(df["churn_probability"].mean()), 3)}


def top_risky(n=20):
    df = latest_per_user()
    if df.empty:
        return df
    return df.sort_values("churn_probability", ascending=False).head(n)[
        ["user_id", "churn_probability", "risk_level", "recommended_action"]]


def funnel():
    rows = db.query("SELECT event_type, COUNT(*) AS c FROM realtime_event_log GROUP BY event_type")
    d = {r["event_type"]: r["c"] for r in rows}
    return {k: int(d.get(k, 0)) for k in ("view", "cart", "purchase")}


def model_performance():
    return db.query("SELECT model_name, model_type, metrics_json, is_active FROM model_registry ORDER BY model_id DESC")


def user_daily_behavior(user_id):
    rows = db.recent_events(user_id, limit=1000)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["event_time"]).dt.date
    piv = df.groupby(["date", "event_type"]).size().unstack(fill_value=0)
    for c in ("view", "cart", "purchase"):
        if c not in piv:
            piv[c] = 0
    return piv[["view", "cart", "purchase"]]

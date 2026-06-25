# -*- coding: utf-8 -*-
"""실시간 DL 시퀀스 예측 서비스 (로컬 torch).
DB 최근 이벤트 → 일별 시퀀스 → LSTM → 이탈확률·위험등급·추천액션.
모델/스케일러는 학습 때 저장한 파일을 transform만 한다(누수 방지)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd, torch, torch.nn as nn
import config
from db import db_client as db

_MODEL = None; _SCALER = None; _META = None


class LSTMClf(nn.Module):
    def __init__(self, f, h=32, p=0.3):
        super().__init__()
        self.lstm = nn.LSTM(f, h, batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(p), nn.Linear(h, 1))

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o[:, -1]).squeeze(1)


def _load():
    global _MODEL, _SCALER, _META
    if _MODEL is not None:
        return
    _META = json.load(open(os.path.join(config.MODELS, "lstm_meta.json"), encoding="utf-8"))
    m = LSTMClf(_META["n_features"], _META.get("hidden", 32))
    m.load_state_dict(torch.load(os.path.join(config.MODELS, "lstm.pth"), map_location="cpu"))
    m.eval(); _MODEL = m
    z = np.load(os.path.join(config.MODELS, "seq_scaler.npz"))
    _SCALER = (z["smu"], z["ssd"])


def build_sequence(events):
    """최근 SEQ_LEN일의 일별 [view,cart,purchase] 카운트 시퀀스 [1, SEQ_LEN, 3]."""
    L = config.SEQ_LEN; ets = config.EVENT_TYPES
    X = np.zeros((L, len(ets)), dtype=np.float32)
    if not events:
        return X[None], 0
    df = pd.DataFrame(events)
    df["event_time"] = pd.to_datetime(df["event_time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    last_day = df["event_time"].dt.normalize().max()
    start = last_day - pd.Timedelta(days=L - 1)
    df = df[df["event_time"].dt.normalize() >= start]
    df["di"] = (df["event_time"].dt.normalize() - start).dt.days.clip(0, L - 1)
    for (di, et), c in df.groupby(["di", "event_type"]).size().items():
        if et in ets:
            X[int(di), ets.index(et)] = c
    return X[None], len(df)


def risk_level(p):
    return "high" if p >= config.RISK_HIGH else ("medium" if p >= config.RISK_LOW else "low")


def recommend(p, events):
    if p >= config.RISK_HIGH:
        return "쿠폰 발송 + 재방문 알림 (고위험)"
    if p >= config.RISK_LOW:
        return "장바구니 리마인드 / 맞춤 추천 (중위험)"
    return "정상 유지 — 별도 조치 불필요"


def predict_user(user_id, log=True):
    """user_id의 최근 행동으로 이탈 확률 예측. dict 반환."""
    _load()
    events = db.recent_events(user_id, limit=500)
    X, n_recent = build_sequence(events)
    smu, ssd = _SCALER
    Xs = (X - smu) / ssd
    with torch.no_grad():
        prob = float(torch.sigmoid(_MODEL(torch.tensor(Xs, dtype=torch.float32))).item())
    risk = risk_level(prob)
    # 간단 top factors (해석용)
    df = pd.DataFrame(events)
    factors = {}
    if events:
        df["event_time"] = pd.to_datetime(df["event_time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        factors = {
            "recent_events": int(len(df)),
            "purchases": int((df.event_type == "purchase").sum()),
            "active_days": int(df["event_time"].dt.normalize().nunique()),
        }
    action = recommend(prob, events)
    pred_id = None
    if log:
        am = db.active_model("sequence")
        mid = am["model_id"] if am else None
        pred_id = db.log_prediction(mid, user_id, prob, risk, factors, action)
        if risk in ("high", "medium"):
            db.log_retention(pred_id, user_id, "auto", action)
    return {"user_id": user_id, "churn_probability": prob, "risk_level": risk,
            "factors": factors, "recommended_action": action, "prediction_id": pred_id}


def predict_all(log=True):
    ids = db.all_user_ids()
    return [predict_user(u, log=log) for u in ids]

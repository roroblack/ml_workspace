# -*- coding: utf-8 -*-
"""라이브 이벤트 시뮬레이터 (Vercel 시뮬 사이트 대역).
무작위 사용자에게 최근 이벤트를 DB에 추가하고 재예측한다."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import config
from db import db_client as db
from frontend_streamlit.services import predictor

BASE = pd.Timestamp("2024-01-01")
PRODUCTS = [f"P{1000+i}" for i in range(80)]


def simulate(n_users=10, seed=None):
    rng = np.random.RandomState(seed)
    uids = db.all_user_ids()
    if not uids:
        return []
    picks = rng.choice(uids, min(n_users, len(uids)), replace=False)
    results = []
    for uid in picks:
        day = config.OBS_DAYS + 6 + int(rng.randint(0, 2))      # 최근 활동(관찰 이후)
        t = BASE + pd.Timedelta(days=day, minutes=int(rng.randint(0, 1400)))
        sid = f"{uid}_live"
        n_view = 1 + rng.poisson(2)
        for _ in range(n_view):
            db.insert_event(str(uid), "view", PRODUCTS[rng.randint(len(PRODUCTS))],
                            round(float(rng.uniform(3, 120)), 2),
                            (t + pd.Timedelta(minutes=int(rng.randint(0, 60)))).strftime("%Y-%m-%d %H:%M:%S"),
                            sid, source="live")
        if rng.random() < 0.4:
            db.insert_event(str(uid), "cart", PRODUCTS[rng.randint(len(PRODUCTS))],
                            round(float(rng.uniform(3, 120)), 2), t.strftime("%Y-%m-%d %H:%M:%S"), sid, source="live")
        results.append(predictor.predict_user(str(uid), log=True))
    return results


def main():
    res = simulate(n_users=10, seed=config.SEED)
    for r in res:
        print(f"{r['user_id']}: 이탈 {r['churn_probability']*100:.1f}% ({r['risk_level']}) → {r['recommended_action']}")
    print(f"[simulate] {len(res)}명 라이브 이벤트 추가·재예측")


if __name__ == "__main__":
    main()

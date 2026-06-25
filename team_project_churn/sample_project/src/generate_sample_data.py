# -*- coding: utf-8 -*-
"""합성 이커머스 이벤트 생성 (REES46 대역). data/raw/events.csv 출력.
사용자별 engagement에 따라 활동량이 달라 recency 기반 이탈 신호가 자연 발생한다."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import config

N_USERS = 2000
N_DAYS = 21
BASE = pd.Timestamp("2024-01-01")
PRODUCTS = [f"P{1000+i}" for i in range(80)]


def main():
    rng = np.random.RandomState(config.SEED)
    rows = []
    for u in range(N_USERS):
        uid = f"U{u:04d}"
        eng = rng.beta(2, 2)                       # 0~1 참여도
        churn_early = rng.random() < (1 - eng) * 0.7   # 저참여일수록 조기 이탈 경향
        last_active_day = rng.randint(3, N_DAYS) if not churn_early else rng.randint(2, config.OBS_DAYS - 1)
        for d in range(N_DAYS):
            if d > last_active_day:
                continue
            if rng.random() > eng * 0.6 + 0.05:    # 그 날 활동 여부
                continue
            day = BASE + pd.Timedelta(days=d)
            sid = f"{uid}_s{d}"
            n_view = 1 + rng.poisson(eng * 4)
            n_cart = rng.binomial(n_view, 0.25 * eng)
            n_purchase = rng.binomial(max(n_cart, 0), 0.4)
            seq = ["view"] * n_view + ["cart"] * n_cart + ["purchase"] * n_purchase
            for k, et in enumerate(seq):
                t = day + pd.Timedelta(minutes=int(rng.randint(0, 1400)))
                rows.append((uid, sid, et, PRODUCTS[rng.randint(len(PRODUCTS))],
                             round(float(rng.uniform(3, 120)), 2), t.strftime("%Y-%m-%d %H:%M:%S")))
    df = pd.DataFrame(rows, columns=["user_id", "session_id", "event_type", "product_id", "price", "event_time"])
    df = df.sort_values(["user_id", "event_time"])
    out = os.path.join(config.RAW, "events.csv")
    df.to_csv(out, index=False)
    print(f"[generate] 이벤트 {len(df):,} / 사용자 {df.user_id.nunique()} → {out}")


if __name__ == "__main__":
    main()

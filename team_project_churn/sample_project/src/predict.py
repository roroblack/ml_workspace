# -*- coding: utf-8 -*-
"""단건 예측 smoke test:  python src/predict.py [user_id]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frontend_streamlit.services import predictor
from db import db_client as db


def main():
    uid = sys.argv[1] if len(sys.argv) > 1 else (db.all_user_ids() or ["U0000"])[0]
    r = predictor.predict_user(uid, log=False)
    print(f"사용자 {r['user_id']}")
    print(f"  이탈확률 {r['churn_probability']*100:.1f}% | 위험등급 {r['risk_level']}")
    print(f"  근거 {r['factors']}")
    print(f"  추천 {r['recommended_action']}")


if __name__ == "__main__":
    main()

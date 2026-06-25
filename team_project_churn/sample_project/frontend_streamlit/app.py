# -*- coding: utf-8 -*-
"""고객 이탈 예측 대시보드 (Streamlit, 로컬 실행).
얼굴 로그인(스텁) → 고객/관리자 대시보드. 실시간 예측은 로컬 torch(predictor).
실행: streamlit run frontend_streamlit/app.py   (먼저: python src/run_pipeline.py)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import streamlit as st
from frontend_streamlit.services import face_auth, dashboard_service as dash, predictor, db_service as dbs

st.set_page_config(page_title="고객 이탈 예측", layout="wide")
ss = st.session_state
ss.setdefault("user", None)
ss.setdefault("role", None)


def login_view():
    st.title("🔐 얼굴 로그인 (샘플 스텁)")
    st.caption("샘플은 카메라 없이 user_id로 로그인합니다. 실제는 InsightFace 얼굴 인증으로 교체.")
    uid = st.text_input("user_id", value="admin")
    st.caption("예: 관리자 = `admin`, 고객 = `U0000` 같은 ID (먼저 run_pipeline.py 실행)")
    if st.button("로그인", type="primary"):
        r = face_auth.verify(uid)
        if r["success"]:
            ss.user, ss.role = r["user_id"], r["role"]
            st.rerun()
        else:
            st.error(r["reason"])


def customer_view():
    st.title("👤 내 이탈 위험")
    uid = ss.user
    pred = dbs.latest_prediction_for(uid)
    if not pred:
        st.warning("예측 기록이 없습니다. 먼저 `python src/run_pipeline.py`를 실행하세요.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("이탈 확률", f"{pred['churn_probability']*100:.1f}%")
        c2.metric("위험 등급", pred["risk_level"].upper())
        c3.metric("추천 액션", "있음" if pred["recommended_action"] else "-")
        st.info(f"💡 개선안: {pred['recommended_action']}")
    st.subheader("최근 행동 (일별 view/cart/purchase)")
    beh = dash.user_daily_behavior(uid)
    if beh.empty:
        st.write("행동 데이터 없음")
    else:
        st.bar_chart(beh)
    if st.button("내 예측 새로고침(실시간)"):
        r = predictor.predict_user(uid, log=True)
        st.success(f"재예측: 이탈 {r['churn_probability']*100:.1f}% ({r['risk_level']})")
        st.rerun()


def admin_view():
    st.title("🛠 관리자 대시보드 — 이탈 현황")
    s = dash.risk_summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("평가 고객", s["total"])
    c2.metric("고위험", s["high"])
    c3.metric("중위험", s["medium"])
    c4.metric("평균 이탈확률", f"{s['avg_prob']*100:.1f}%")

    cols = st.columns(2)
    with cols[0]:
        st.subheader("이벤트 퍼널")
        fn = dash.funnel()
        st.bar_chart(pd.DataFrame({"count": fn}))
    with cols[1]:
        st.subheader("모델 성능 (model_registry)")
        st.dataframe(pd.DataFrame(dash.model_performance()), use_container_width=True)

    st.subheader("이탈 고위험 고객 Top 20")
    st.dataframe(dash.top_risky(20), use_container_width=True)

    st.divider()
    if st.button("🔄 라이브 이벤트 시뮬레이션 + 재예측", type="primary"):
        from src import simulate_events
        res = simulate_events.simulate(n_users=10)
        st.success(f"{len(res)}명 라이브 이벤트 추가·재예측 완료")
        st.rerun()


# ---- 라우팅 ----
if ss.user is None:
    login_view()
else:
    with st.sidebar:
        st.write(f"로그인: **{ss.user}** ({ss.role})")
        if st.button("로그아웃"):
            ss.user, ss.role = None, None
            st.rerun()
    if ss.role == "admin":
        admin_view()
    else:
        customer_view()

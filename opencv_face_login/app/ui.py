"""Streamlit 화면 구성에 필요한 UI 보조 함수를 모아 둔 모듈입니다."""

# Streamlit은 웹 화면을 구성하는 데 사용합니다.
import streamlit as st


def init_session_state() -> None:
    """로그인 상태 등 Streamlit 세션 변수를 초기화합니다."""

    # 로그인 여부가 세션에 없으면 기본값 False로 생성합니다.
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        # st.session_state.logged_in = True # 로그인 되어있는 상태로 만들어서 테스트용

    # 로그인 사용자 ID가 세션에 없으면 None으로 생성합니다.
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
        # st.session_state.user_id = 'user01 # 로그인 상태로 만들어서 테스트용

    # 얼굴 유사도 점수가 세션에 없으면 0.0으로 생성합니다.
    if "face_score" not in st.session_state:
        st.session_state.face_score = 0.0


def logout() -> None:
    """현재 로그인 세션을 초기화합니다."""

    # 로그인 여부를 False로 변경합니다.
    st.session_state.logged_in = False

    # 사용자 ID를 제거합니다.
    st.session_state.user_id = None

    # 얼굴 유사도 점수를 초기화합니다.
    st.session_state.face_score = 0.0

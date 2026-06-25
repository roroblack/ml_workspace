"""Streamlit 화면 구성에 필요한 UI 보조 함수를 모아 둔 모듈입니다."""

# Streamlit은 웹 화면을 구성하는 데 사용합니다.
import streamlit as st


def init_session_state() -> None:
    """로그인 상태 등 Streamlit 세션 변수를 초기화합니다."""

    # 로그인 여부가 세션에 없으면 기본값 False로 생성합니다.
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # 로그인 사용자 ID가 세션에 없으면 None으로 생성합니다.
    if "user_id" not in st.session_state:
        st.session_state.user_id = None

    # 로그인 사용자 이름이 세션에 없으면 None으로 생성합니다.
    if "user_name" not in st.session_state:
        st.session_state.user_name = None

    # 얼굴 유사도 점수가 세션에 없으면 0.0으로 생성합니다.
    if "face_score" not in st.session_state:
        st.session_state.face_score = 0.0

    # 아이디/암호 1차 인증을 통과한 사용자 ID를 저장합니다.
    if "pending_face_user_id" not in st.session_state:
        st.session_state.pending_face_user_id = None

    # 아이디/암호 1차 인증을 통과한 사용자 이름을 저장합니다.
    if "pending_face_user_name" not in st.session_state:
        st.session_state.pending_face_user_name = None


def reset_pending_face_auth() -> None:
    """얼굴 2차 인증 대기 상태를 초기화합니다."""

    # 2차 인증 대기 사용자 ID를 제거합니다.
    st.session_state.pending_face_user_id = None

    # 2차 인증 대기 사용자 이름을 제거합니다.
    st.session_state.pending_face_user_name = None


def logout() -> None:
    """현재 로그인 세션을 초기화합니다."""

    # 로그인 여부를 False로 변경합니다.
    st.session_state.logged_in = False

    # 사용자 ID를 제거합니다.
    st.session_state.user_id = None

    # 사용자 이름을 제거합니다.
    st.session_state.user_name = None

    # 얼굴 유사도 점수를 초기화합니다.
    st.session_state.face_score = 0.0

    # 얼굴 인증 대기 상태도 초기화합니다.
    reset_pending_face_auth()

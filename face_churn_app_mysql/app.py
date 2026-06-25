"""Streamlit + OpenCV + InsightFace 얼굴 로그인 + 고객 이탈 예측 메인 앱입니다."""

# OpenCV는 얼굴 사각형 표시 이미지의 색상 변환에 사용합니다.
import cv2

# Streamlit은 웹 애플리케이션 화면을 구성하는 프레임워크입니다.
import streamlit as st

# DB 초기화와 아이디/암호 확인 함수를 가져옵니다.
from app.db import init_db, verify_user_password

# 얼굴 등록, 얼굴 검출 표시, 얼굴 2차 인증 함수를 가져옵니다.
from app.face_auth import (
    DEFAULT_SIMILARITY_THRESHOLD,
    draw_face_box,
    read_camera_image,
    register_face,
    verify_face_for_user,
)

# 고객 이탈 예측 함수를 가져옵니다.
from app.churn_service import predict_churn

# 세션 초기화, 로그아웃, 2차 인증 대기 초기화 함수를 가져옵니다.
from app.ui import init_session_state, logout, reset_pending_face_auth


# Streamlit 페이지 제목, 아이콘, 화면 폭을 설정합니다.
st.set_page_config(page_title="Face Login + Churn Prediction", page_icon="🔐", layout="wide")

# 앱 시작 시 세션 상태를 초기화합니다.
init_session_state()

# MySQL DB와 사용자 테이블을 초기화합니다.
# MySQL 서버가 실행 중이어야 하며 접속 정보는 환경 변수 또는 app/db.py 기본값을 사용합니다.
try:
    init_db()
except Exception as e:
    st.error("MySQL DB 연결 또는 초기화에 실패했습니다.")
    st.exception(e)
    st.stop()

# camera_input 위젯의 화면 크기를 기본보다 작게 보이도록 CSS를 적용합니다.
# width:70%는 현재 camera_input 영역을 약 70% 크기로 줄여 표시합니다.
st.markdown(
    """
    <style>
    div[data-testid="stCameraInput"] {
        width: 70% !important;
        max-width: 520px !important;
    }
    div[data-testid="stCameraInput"] video {
        width: 100% !important;
    }
    div[data-testid="stCameraInput"] img {
        width: 100% !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 메인 제목을 출력합니다.
st.title("🔐 얼굴 2차 인증 로그인 + 고객 이탈 예측 서비스")

# 앱의 핵심 동작을 간단히 설명합니다.
st.caption("회원 등록 → 아이디/암호 1차 확인 → 얼굴 2차 인증 → 고객 이탈 예측 기능 사용")


# 사이드바에는 슬라이더 없이 로그인 상태만 표시합니다.
with st.sidebar:
    # 사이드바 제목을 출력합니다.
    st.header("사용자 상태")

    # 현재 로그인 상태라면 사용자 ID와 얼굴 유사도를 표시합니다.
    if st.session_state.logged_in:
        st.success(f"로그인 사용자: {st.session_state.user_id}")
        if st.session_state.user_name:
            st.info(f"이름: {st.session_state.user_name}")
        st.info(f"얼굴 유사도: {st.session_state.face_score:.3f}")

        # 로그아웃 버튼을 누르면 세션을 초기화합니다.
        if st.button("로그아웃"):
            logout()
            st.rerun()
    else:
        # 로그인 전에는 안내 메시지를 표시합니다.
        st.warning("현재 로그인되지 않았습니다.")


# 로그인 전에는 얼굴 등록과 로그인 탭을 제공합니다.
if not st.session_state.logged_in:
    # 얼굴 등록 탭과 로그인 탭을 생성합니다.
    register_tab, login_tab = st.tabs(["1. 얼굴 등록", "2. 로그인"])

    # 얼굴 등록 탭 화면을 구성합니다.
    with register_tab:
        # 등록 섹션 제목을 출력합니다.
        st.subheader("회원 정보 + 얼굴 등록")

        # 사용자 ID를 입력받습니다.
        register_user_id = st.text_input("아이디", placeholder="예: user01", key="register_user_id")

        # 비밀번호를 입력받습니다. type=password는 화면에 비밀번호를 숨겨 표시합니다.
        register_password = st.text_input("암호", type="password", key="register_password")

        # 사용자 이름을 입력받습니다.
        register_name = st.text_input("이름", placeholder="예: 홍길동", key="register_name")

        # camera_input으로 등록 얼굴을 촬영합니다. 파일 업로더는 사용하지 않습니다.
        register_camera_file = st.camera_input("등록할 얼굴을 촬영하세요.", key="register_camera")

        # 촬영 이미지가 있으면 OpenCV 이미지로 변환합니다.
        register_image_bgr = read_camera_image(register_camera_file)

        # 촬영된 얼굴에 사각형을 표시하여 얼굴 검출 여부를 확인시킵니다.
        if register_image_bgr is not None:
            annotated_bgr, face_found, face_message = draw_face_box(register_image_bgr)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption=face_message, width=420)
            if not face_found:
                st.warning(face_message)

        # 등록 버튼을 누르면 사용자 정보와 얼굴 정보를 MySQL에 저장합니다.
        if st.button("회원 및 얼굴 등록 실행", type="primary"):
            # 이미지가 없으면 경고를 표시합니다.
            if register_image_bgr is None:
                st.warning("등록할 얼굴 이미지를 카메라로 촬영하세요.")
            else:
                try:
                    # 회원 정보와 얼굴 임베딩을 등록합니다.
                    ok, message = register_face(
                        register_user_id,
                        register_password,
                        register_name,
                        register_image_bgr,
                    )

                    # 등록 성공 시 성공 메시지를 표시합니다.
                    if ok:
                        st.success(message)
                    else:
                        # 등록 실패 시 오류 메시지를 표시합니다.
                        st.error(message)
                except Exception as e:
                    st.error("회원/얼굴 등록 중 오류가 발생했습니다.")
                    st.exception(e)

    # 로그인 탭 화면을 구성합니다.
    with login_tab:
        # 로그인 섹션 제목을 출력합니다.
        st.subheader("아이디/암호 로그인 후 얼굴 2차 인증")

        # 로그인할 아이디를 입력받습니다.
        login_user_id = st.text_input("아이디", key="login_user_id")

        # 로그인할 암호를 입력받습니다.
        login_password = st.text_input("암호", type="password", key="login_password")

        # 1차 인증 버튼입니다.
        if st.button("1차 아이디/암호 확인", type="primary"):
            # 기존 2차 인증 대기 상태를 초기화합니다.
            reset_pending_face_auth()

            # 아이디와 암호 입력 여부를 먼저 확인합니다.
            if not login_user_id.strip() or not login_password.strip():
                st.warning("아이디와 암호를 모두 입력하세요.")
            else:
                try:
                    # MySQL에 저장된 사용자 정보와 비밀번호 해시를 확인합니다.
                    ok, user_name, message = verify_user_password(login_user_id.strip(), login_password)

                    # 아이디/암호가 맞으면 2차 얼굴 인증 대기 상태로 전환합니다.
                    if ok:
                        st.session_state.pending_face_user_id = login_user_id.strip()
                        st.session_state.pending_face_user_name = user_name
                        st.success(message)
                    else:
                        st.error(message)
                except Exception as e:
                    st.error("아이디/암호 확인 중 오류가 발생했습니다.")
                    st.exception(e)

        # 1차 인증을 통과한 경우에만 얼굴 2차 인증 화면을 표시합니다.
        if st.session_state.pending_face_user_id:
            st.divider()
            st.info(f"{st.session_state.pending_face_user_id} 계정의 얼굴 2차 인증을 진행하세요.")

            # 로그인용 얼굴을 camera_input으로 촬영합니다. 파일 업로더는 사용하지 않습니다.
            login_camera_file = st.camera_input("로그인할 얼굴을 촬영하세요.", key="login_camera")

            # 촬영 이미지를 OpenCV BGR 이미지로 변환합니다.
            login_image_bgr = read_camera_image(login_camera_file)

            # 촬영된 얼굴에 사각형 테두리를 표시합니다.
            if login_image_bgr is not None:
                annotated_bgr, face_found, face_message = draw_face_box(login_image_bgr)
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, caption=face_message, width=420)
                if not face_found:
                    st.warning(face_message)

            # 2차 얼굴 인증 버튼입니다.
            if st.button("2차 얼굴 인증 실행", type="primary"):
                if login_image_bgr is None:
                    st.warning("로그인할 얼굴 이미지를 카메라로 촬영하세요.")
                else:
                    try:
                        # 1차 인증을 통과한 user_id의 등록 얼굴과 현재 촬영 얼굴을 비교합니다.
                        ok, score, message = verify_face_for_user(
                            st.session_state.pending_face_user_id,
                            login_image_bgr,
                            threshold=DEFAULT_SIMILARITY_THRESHOLD,
                        )

                        # 얼굴 인증 성공 시 최종 로그인 세션을 저장합니다.
                        if ok:
                            st.session_state.logged_in = True
                            st.session_state.user_id = st.session_state.pending_face_user_id
                            st.session_state.user_name = st.session_state.pending_face_user_name
                            st.session_state.face_score = score
                            reset_pending_face_auth()
                            st.success(f"{message} 유사도: {score:.3f}")
                            st.rerun()
                        else:
                            st.error(f"{message} 유사도: {score:.3f}")
                    except Exception as e:
                        st.error("얼굴 2차 인증 중 오류가 발생했습니다.")
                        st.exception(e)

# 로그인 후에는 고객 이탈 예측 서비스를 표시합니다.
else:
    # 고객 이탈 예측 제목을 출력합니다.
    st.subheader("📊 고객 이탈 예측")

    # 입력 폼을 사용하여 한 번에 고객 정보를 입력받습니다.
    with st.form("churn_form"):
        # 좌우 두 컬럼을 만들어 입력 항목을 보기 좋게 배치합니다.
        col1, col2 = st.columns(2)

        # 첫 번째 컬럼에는 기본 고객 정보와 계정 정보를 배치합니다.
        with col1:
            gender = st.selectbox("성별", ["Male", "Female"])
            senior = st.selectbox("고령 고객 여부", ["0", "1"])
            partner = st.selectbox("배우자 여부", ["Yes", "No"])
            dependents = st.selectbox("부양가족 여부", ["Yes", "No"])
            tenure = st.number_input("가입 기간(개월)", min_value=0, max_value=100, value=12)
            contract = st.selectbox("계약 유형", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("전자 청구서 사용", ["Yes", "No"])
            payment = st.selectbox(
                "결제 방식",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        # 두 번째 컬럼에는 서비스 이용 정보를 배치합니다.
        with col2:
            phone = st.selectbox("전화 서비스", ["Yes", "No"])
            multiple = st.selectbox("복수 회선", ["Yes", "No", "No phone service"])
            internet = st.selectbox("인터넷 서비스", ["DSL", "Fiber optic", "No"])
            security = st.selectbox("온라인 보안", ["Yes", "No", "No internet service"])
            backup = st.selectbox("온라인 백업", ["Yes", "No", "No internet service"])
            protection = st.selectbox("기기 보호", ["Yes", "No", "No internet service"])
            tech = st.selectbox("기술 지원", ["Yes", "No", "No internet service"])
            tv = st.selectbox("스트리밍 TV", ["Yes", "No", "No internet service"])
            movies = st.selectbox("스트리밍 영화", ["Yes", "No", "No internet service"])
            monthly = st.number_input("월 요금", min_value=0.0, max_value=300.0, value=75.0, step=1.0)
            total = st.number_input("총 요금", min_value=0.0, max_value=20000.0, value=900.0, step=10.0)

        # 예측 실행 버튼을 생성합니다.
        submitted = st.form_submit_button("이탈 예측 실행", type="primary")

    # 사용자가 예측 버튼을 누르면 모델 입력값을 만들고 예측을 수행합니다.
    if submitted:
        # 모델 입력 컬럼명은 학습 때 사용한 컬럼명과 동일해야 합니다.
        values = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": tech,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }

        # 고객 이탈 예측 서비스를 호출합니다.
        result = predict_churn(values)

        # 예측 확률을 퍼센트로 변환합니다.
        churn_pct = result["churn_probability"] * 100

        # 예측 결과 라벨을 크게 표시합니다.
        st.metric("예측 결과", result["label"], f"이탈 확률 {churn_pct:.1f}%")

        # 이탈 확률을 진행 막대로 표시합니다.
        st.progress(result["churn_probability"])

        # 이탈 위험이 높으면 관리 전략을 안내합니다.
        if result["prediction"] == 1:
            st.warning("이 고객은 이탈 가능성이 높습니다. 장기 계약 할인, 기술 지원 강화, 요금제 재설계를 검토하세요.")
        else:
            st.success("이 고객은 현재 잔류 가능성이 높습니다. 만족도 유지와 추가 서비스 제안을 검토하세요.")

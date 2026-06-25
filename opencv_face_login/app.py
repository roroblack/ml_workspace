"""Streamlit + OpenCV + InsightFace 얼굴 로그인 + 고객 이탈 예측 메인 앱입니다."""

# OpenCV는 업로드 이미지 디코딩과 색상 변환에 사용합니다.
import cv2

# Streamlit은 웹 애플리케이션 화면을 구성하는 프레임워크입니다.
import streamlit as st

# 얼굴 등록, 얼굴 인증, 업로드 이미지 변환 함수를 가져옵니다.
from app.face_auth import read_uploaded_image, register_face, verify_face, DEFAULT_SIMILARITY_THRESHOLD

# 고객 이탈 예측 함수를 가져옵니다.
from app.churn_service import predict_churn

# 세션 초기화와 로그아웃 함수를 가져옵니다.
from app.ui import init_session_state, logout


# Streamlit 페이지 제목, 아이콘, 화면 폭을 설정합니다.
st.set_page_config(page_title="Face Login + Churn Prediction", page_icon="🔐", layout="wide")

# 앱 시작 시 세션 상태를 초기화합니다.
init_session_state()

# 메인 제목을 출력합니다.
st.title("🔐 InsightFace 얼굴 로그인 + 고객 이탈 예측 서비스")

# 앱의 핵심 동작을 간단히 설명합니다.
st.caption("얼굴 등록 → 얼굴 로그인 → 로그인 후 고객 이탈 예측 기능 사용")


# 사이드바에 로그인 상태를 표시합니다.
with st.sidebar:
    # 사이드바 제목을 출력합니다.
    st.header("사용자 상태")

    # 현재 로그인 상태라면 사용자 ID와 얼굴 유사도를 표시합니다.
    if st.session_state.logged_in:
        st.success(f"로그인 사용자: {st.session_state.user_id}")
        st.info(f"얼굴 유사도: {st.session_state.face_score:.3f}")

        # 로그아웃 버튼을 누르면 세션을 초기화합니다.
        if st.button("로그아웃"):
            logout()
            st.rerun()
    else:
        # 로그인 전에는 안내 메시지를 표시합니다.
        st.warning("현재 로그인되지 않았습니다.")

    # 얼굴 인증 임계값을 사용자가 조정할 수 있게 합니다.
    threshold = st.slider(
        "얼굴 인증 임계값",
        min_value=0.20,
        max_value=0.80,
        value=DEFAULT_SIMILARITY_THRESHOLD,
        step=0.01,
        help="값을 높이면 보안은 강해지지만 로그인 실패가 늘 수 있습니다.",
    )


# 로그인 전에는 얼굴 등록과 얼굴 로그인 탭을 제공합니다.
if not st.session_state.logged_in:
    # 얼굴 등록 탭과 얼굴 로그인 탭을 생성합니다.
    register_tab, login_tab = st.tabs(["1. 얼굴 등록", "2. 얼굴 로그인"])

    # 얼굴 등록 탭 화면을 구성합니다.
    with register_tab:
        # 등록 섹션 제목을 출력합니다.
        st.subheader("얼굴 등록")

        # 등록할 사용자 ID를 입력받습니다.
        user_id = st.text_input("사용자 ID", placeholder="예: user01")

        # 카메라 촬영 이미지를 입력받습니다.
        camera_image = st.camera_input("등록할 얼굴을 촬영하세요.")

        # 파일 업로드 방식도 함께 제공합니다.
        uploaded_image = st.file_uploader("또는 얼굴 이미지 파일 업로드", type=["jpg", "jpeg", "png"], key="register_upload")

        # 촬영 이미지가 있으면 우선 사용하고, 없으면 업로드 이미지를 사용합니다.
        selected_file = camera_image if camera_image is not None else uploaded_image

        # 등록 버튼을 누르면 얼굴 등록 로직을 실행합니다.
        if st.button("얼굴 등록 실행", type="primary"):
            # 업로드 또는 촬영 파일을 OpenCV 이미지로 변환합니다.
            image_bgr = read_uploaded_image(selected_file)

            # 이미지가 없으면 경고를 표시합니다.
            if image_bgr is None:
                st.warning("등록할 얼굴 이미지를 촬영하거나 업로드하세요.")
            else:
                # 얼굴 등록 함수를 호출합니다.
                ok, message = register_face(user_id, image_bgr)

                # 등록 성공 시 성공 메시지를 표시합니다.
                if ok:
                    st.success(message)
                else:
                    # 등록 실패 시 오류 메시지를 표시합니다.
                    st.error(message)

    # 얼굴 로그인 탭 화면을 구성합니다.
    with login_tab:
        # 로그인 섹션 제목을 출력합니다.
        st.subheader("얼굴 로그인")

        # 로그인용 카메라 촬영 이미지를 입력받습니다.
        login_camera_image = st.camera_input("로그인할 얼굴을 촬영하세요.", key="login_camera")

        # 로그인용 파일 업로드도 제공합니다.
        login_uploaded_image = st.file_uploader("또는 로그인 얼굴 이미지 업로드", type=["jpg", "jpeg", "png"], key="login_upload")

        # 촬영 이미지가 있으면 우선 사용하고, 없으면 업로드 이미지를 사용합니다.
        login_selected_file = login_camera_image if login_camera_image is not None else login_uploaded_image

        # 로그인 버튼을 누르면 얼굴 비교 로직을 실행합니다.
        if st.button("얼굴 로그인 실행", type="primary"):
            # 입력 파일을 OpenCV BGR 이미지로 변환합니다.
            image_bgr = read_uploaded_image(login_selected_file)

            # 이미지가 없으면 경고를 표시합니다.
            if image_bgr is None:
                st.warning("로그인할 얼굴 이미지를 촬영하거나 업로드하세요.")
            else:
                # 등록 얼굴 DB와 입력 얼굴을 비교합니다.
                ok, matched_user_id, score, message = verify_face(image_bgr, threshold=threshold)

                # 인증 성공 시 로그인 세션을 저장합니다.
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.user_id = matched_user_id
                    st.session_state.face_score = score
                    st.success(f"{message} 사용자: {matched_user_id}, 유사도: {score:.3f}")
                    st.rerun()
                else:
                    # 인증 실패 시 가장 가까운 후보와 점수를 함께 표시합니다.
                    st.error(f"{message} 가장 가까운 사용자: {matched_user_id}, 유사도: {score:.3f}")

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

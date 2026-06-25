# Streamlit 화면 구성에 필요한 ui 관련 사용자 정의  함수를 모아둔 모듈

import streamlit as st
# 웹 화면을 구성하는데 사용하는 파이썬 웹 애플리케이션 프레임워크



def init_session_state() -> None:
    """로그인 상태에 대한 세션 변수를 초기화 하는 함수
        streamlit 은 버튼 클릭이나 입력 변경이 발생시
        스크립트를 위에서 다시 실행"""

    # logged_in 키가 세션에 없다면, 로그인 여부의 기본값을 False 로 지정
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False # 비로그인

    # user_id 키가 세션에 없다면
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None

    # face_score 키가 아직 세션에 없다면, 얼굴 유사도 점수를 저장할 공간을 세션에 생성함
    if 'face_score' not in st.session_state:
        st.session_state.face_score = 0.0

    return
#---------------------------------------------------

def logout() -> None:

    '''현재 로그인 세션의 정보를 비로그인 상태로 초기화함
        로그아웃 버튼 눌렀을 때 호출할 것임'''
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.face_score = 0.0
    return
#---------------------------------------------------

def render_page_header() -> None:
    '''앱 상단 제목과 설명 문구를 출력하는 함수'''
    st.title('insightFace 얼굴 로그인 + 고객 이탈 예측 서비스') # Streamlit 가장 큰 제목을 출력
    # 제목 아래에 작은 안내 문구를 출력
    st.caption('얼굴 등록 -> 얼굴 로그인 -> 로그인 후 고객 이탈 예측 기능 사용')
    return
#---------------------------------------------------

def render_sidebar(default_threshold: float ) -> float:
    '''사이드바 ui 를 출력하고 얼굴 인증 임계값을 반환
        Args:
            default_threshold : 얼굴 인증에 사용할 기본 유사도 임계값임
        Returns:
            사용자가 사이드바 슬라이더에서 선택한 얼굴 인증 임계값임
    '''

    # with st.sidebar: 블록 안에 작성하면 왼쪽 사이드바에 표시됨
    with st.sidebar:
        # 사이드바의 섹션 제목 출력
        st.header('사용자 상태')

        if st.session_state.logged_in:
            # 로그인 한 사용자 아이디와 로그인 성공 메세지를 출력
            st.success(f'로그인 사용자 : {st.session_state.user_id}')

            # 얼굴 로그인 시 계산된 유사도 점수를 소수점 3자리까지 출력
            st.info(f'얼굴 유사도 : {st.session_state.face_score:.3f}')

            # 로그아웃 버튼 생성. 클릭시 True 가 반환
            if st.button('로그아웃'):
                logout()
                st.rerun() # 변경된 세션 상태를 화면에 반영함 (비로그인 화면 상태로 바꿈)
        else: # 비로그인 상태라면
            st.warning('현재 로그인되지 않았습니다.')

            # 얼굴 인증 기준값을 사용자가 조정할 수 있도록 슬라이더를 생성
            threshold = st.slider(
                '얼굴 인증 임계값',
                min_value=0.20,
                max_value=0.80,
                value=default_threshold,
                step=0.01,
                help='값을 높이면 보안이 강해지고, 로그인 실패 확률이 높아집니다',
            )
    # with---------------------------

    return threshold


#---------------------------------------------------

def render_register_ui():
    '''얼굴 등록 화면을 출력하고, 사용자 입력값을 반환하는 함수
        Returns :
            clicked : 얼굴 등록 실행 버튼 클릭 여부
            user_id : 등록할 사용자 아이디
            selected_file : 카메라 촬영 이미지 또는 업로드한 이미지 파일
    '''

    # 얼굴 등록 영역의 소제목 출력
    st.subheader('얼굴 등록')

    # 등록할 사용자 아이디 입력받음
    user_id = st.text_input(
        '사용자 아이디',                      # 입력창에 표시할 라벨 문구
        placeholder='예: user01',             # 입력전 안내 문구

    )

    # 웹캠으로 얼굴을 촬영할 수 있는 입력 컴포넌트
    camera_image = st.camera_input(
        '등록할 얼굴을 촬영하세요.',
        type=['jpg', 'jpeg', 'png'],
        key='register_camera',
    )
    uploaded_image = st.file_uploader(
        '등록할 얼굴 이미지 파일 업로드',
        type=['jpg', 'jpeg', 'png'],
        key='register_upload',
    )

    selected_file = camera_image if camera_image is not None else uploaded_image

    # 얼굴 등록 로직을 실행할 버튼
    clicked = st.button('얼굴 등록 실행', type='primary')

    return clicked, user_id, selected_file
#---------------------------------------------------

def render_login_ui() :
    '''얼굴 로그인 화면을 출력하고, 사용자의 입력값을 받아서 리턴하는 함수
        Returns :
            clicked         : 얼굴 로그인 실행 버튼 클릭 여부
            selected_file   : 카메라 촬영 이미지 또는 업로드 된 이미지 파일
    '''

    # 얼굴 로그인 영역의 소제목을 출력함
    st.subheader('얼굴 로그인')

    # 웹캠으로 로그인할 얼굴 사진을 촬영할 수 있는 입력 컴포넌트
    login_camera_image = st.camera_input(
        '로그인할 얼굴을 촬영하세요.',                  # 카메라 입력 컴포넌트에 표시할 라벨 문구
        key='login_camera',                             # 등록 탭의 카메라 입력과 충돌나지 않게 고유키를 지정
    )

    # 등록에 사용할 이미지 파일 업로드 할 수 있는 파일 업로더 제공
    login_uploaded_image = st.file_uploader(
        '로그인 시 비교할 얼굴 이미지 파일 업로드',     # 파일 업로드 컴포넌트에 표시할 라벨 문구
        type=['jpg', 'jpeg', 'png'],                    # 허용할 이미지 확장자 목록
        key='login_upload',                             # 등록 업로드 컴포넌트와 충돌나지 않도록 고유 이름 지정
    )

    # 카메라 촬영 이미지가 있으면 카메라 이미지를 우선 사용하고, 없으면 업로드 이미지 사용
    selected_file = login_camera_image if login_camera_image is not None else login_uploaded_image

    # 얼굴 로그인 로직 실행할 버튼
    clicked = selected_file('얼굴 로그인 실행', type='primary')

    # app.py 에서 실제 로그인 로직 처리를 할 수 있도록 입력값을 반환
    return clicked, selected_file
#---------------------------------------------------

def render_churn_form():
    pass
#---------------------------------------------------

def render_churn_result(result: dict):
    pass


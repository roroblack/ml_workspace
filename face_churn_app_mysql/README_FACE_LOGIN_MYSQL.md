# 얼굴 로그인 + 고객 이탈 예측 수정 버전

## 수정 반영 내용

1. 사이드바의 얼굴 인증 임계값 슬라이더를 제거했습니다.
2. `st.camera_input` 화면 크기를 CSS로 약 70% 크기로 줄였습니다.
3. 얼굴 등록과 얼굴 로그인 촬영 후 얼굴 영역에 초록색 사각형 테두리가 표시되도록 수정했습니다.
4. 기존 파일 기반 얼굴 DB(`face_db.npy`) 중심 구조를 MySQL 저장 구조로 변경했습니다.
5. 로그인은 `아이디/암호 1차 인증 → 얼굴 2차 인증` 순서로 동작합니다.
6. 얼굴 등록 시 아이디, 암호, 이름, 얼굴 임베딩이 MySQL에 저장됩니다.
7. 로그인 시 MySQL에 저장된 아이디/암호를 먼저 확인한 후 얼굴 인증으로 넘어갑니다.
8. 얼굴 등록과 로그인 화면에서 파일 업로더를 제거했습니다.

## MySQL 설정

기본 접속 정보는 `app/db.py`에 다음과 같이 설정되어 있습니다.

```python
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=1234
MYSQL_DATABASE=face_churn_db
```

환경이 다르면 실행 전에 환경 변수를 설정하거나 `app/db.py`의 기본값을 수정하세요.

Windows PowerShell 예시:

```powershell
$env:MYSQL_HOST="localhost"
$env:MYSQL_PORT="3306"
$env:MYSQL_USER="root"
$env:MYSQL_PASSWORD="본인암호"
$env:MYSQL_DATABASE="face_churn_db"
streamlit run app.py
```

## 테이블

앱 실행 시 아래 테이블이 자동 생성됩니다.

```sql
CREATE TABLE IF NOT EXISTS face_users (
    user_id VARCHAR(100) PRIMARY KEY,
    user_name VARCHAR(100) NOT NULL,
    password_salt VARCHAR(64) NOT NULL,
    password_hash VARCHAR(128) NOT NULL,
    face_embedding LONGBLOB NOT NULL,
    face_image_path VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 주의 사항

- MySQL 서버가 먼저 실행 중이어야 합니다.
- 얼굴 인증은 조명, 카메라 각도, 얼굴 크기에 영향을 받습니다.
- 등록과 로그인 모두 정면 얼굴이 선명하게 보여야 합니다.
- `insightface` 모델팩은 최초 실행 시 다운로드가 필요할 수 있습니다.

"""InsightFace + ArcFace 기반 얼굴 등록/로그인 기능을 담당하는 모듈입니다."""

# 운영체제 경로 처리를 안전하게 하기 위해 pathlib의 Path 클래스를 가져옵니다.
from pathlib import Path

# 타입 힌트를 사용하여 함수 입력과 반환값을 명확하게 표현하기 위해 가져옵니다.
from typing import Optional, Tuple

# OpenCV는 웹캠 프레임과 이미지 파일을 읽고 BGR/RGB 변환을 수행하는 데 사용합니다.
import cv2

# InsightFace의 FaceAnalysis는 얼굴 검출, 정렬, 임베딩 추출을 한 번에 수행하는 고수준 API입니다.
from insightface.app import FaceAnalysis

# NumPy는 얼굴 임베딩 벡터 저장, 정규화, 코사인 유사도 계산에 사용합니다.
import numpy as np

# MySQL 사용자 저장/조회 함수들을 가져옵니다.
from app.db import create_or_update_user, get_user_face_embedding


# 등록된 얼굴 원본 이미지가 저장될 기본 폴더를 정의합니다.
FACE_IMAGE_DIR = Path("registered_faces")

# 동일 인물 여부를 판단하기 위한 기본 코사인 유사도 임계값입니다.
# 슬라이더를 제거하기 위해 앱에서는 이 고정값을 사용합니다.
DEFAULT_SIMILARITY_THRESHOLD = 0.45

# InsightFace 모델 객체를 앱 전체에서 한 번만 초기화하여 반복 로딩 비용을 줄이기 위한 전역 캐시 변수입니다.
_FACE_APP: Optional[FaceAnalysis] = None


def get_face_app() -> FaceAnalysis:
    """InsightFace FaceAnalysis 객체를 초기화하고 반환합니다."""

    # 전역 캐시 변수를 함수 내부에서 수정하기 위해 global 키워드를 사용합니다.
    global _FACE_APP

    # 이미 모델이 초기화되어 있다면 다시 로딩하지 않고 기존 객체를 반환합니다.
    if _FACE_APP is not None:
        return _FACE_APP

    # buffalo_l 모델팩은 얼굴 검출과 얼굴 임베딩 모델을 포함합니다.
    _FACE_APP = FaceAnalysis(name="buffalo_l")

    # ctx_id=-1은 CPU 실행을 의미합니다.
    # det_size는 얼굴 검출 입력 크기이며, 클수록 작은 얼굴 검출에 유리하지만 속도는 느려질 수 있습니다.
    _FACE_APP.prepare(ctx_id=-1, det_size=(640, 640))

    # 초기화된 모델 객체를 반환합니다.
    return _FACE_APP


def ensure_dirs() -> None:
    """얼굴 이미지 저장 폴더를 생성합니다."""

    # parents=True는 상위 폴더가 없어도 함께 만들도록 하고, exist_ok=True는 이미 있어도 오류를 내지 않습니다.
    FACE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """OpenCV BGR 이미지를 RGB 이미지로 변환합니다."""

    # OpenCV는 기본적으로 BGR 색상 순서를 사용하므로 RGB로 변환합니다.
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    """RGB 이미지를 OpenCV BGR 이미지로 변환합니다."""

    # Streamlit 화면 표시용 RGB 이미지를 OpenCV 처리용 BGR 이미지로 변환합니다.
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def read_camera_image(camera_file) -> Optional[np.ndarray]:
    """Streamlit camera_input 결과를 OpenCV BGR 이미지로 변환합니다."""

    # 촬영된 이미지가 없으면 None을 반환합니다.
    if camera_file is None:
        return None

    # camera_input은 UploadedFile과 유사한 객체를 반환하므로 바이트 값을 읽습니다.
    file_bytes = np.frombuffer(camera_file.getvalue(), dtype=np.uint8)

    # OpenCV imdecode로 이미지 바이트를 BGR 배열로 변환합니다.
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 변환된 이미지를 반환합니다.
    return image_bgr


def detect_largest_face(image_bgr: np.ndarray):
    """이미지에서 가장 큰 얼굴 객체를 검출하여 반환합니다."""

    # InsightFace 모델 객체를 가져옵니다.
    app = get_face_app()

    # 입력 이미지가 비어 있으면 얼굴 검출을 수행할 수 없습니다.
    if image_bgr is None or image_bgr.size == 0:
        return None, "이미지를 읽을 수 없습니다."

    # InsightFace 입력에 맞게 BGR 이미지를 RGB로 변환합니다.
    image_rgb = bgr_to_rgb(image_bgr)

    # 이미지에서 얼굴 후보 목록을 검출합니다.
    faces = app.get(image_rgb)

    # 얼굴이 없으면 실패 메시지를 반환합니다.
    if len(faces) == 0:
        return None, "얼굴을 찾지 못했습니다. 정면 얼굴이 잘 보이도록 다시 촬영하세요."

    # 여러 얼굴이 검출되면 가장 큰 얼굴 하나를 선택합니다.
    largest_face = max(
        faces,
        key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]),
    )

    # 가장 큰 얼굴 객체와 성공 메시지를 반환합니다.
    return largest_face, "얼굴 검출 성공"


def draw_face_box(image_bgr: np.ndarray) -> Tuple[np.ndarray, bool, str]:
    """얼굴 주변에 사각형 테두리를 그린 이미지를 반환합니다."""

    # 원본 이미지를 수정하지 않기 위해 복사본을 만듭니다.
    annotated = image_bgr.copy()

    # 가장 큰 얼굴을 검출합니다.
    face, message = detect_largest_face(image_bgr)

    # 얼굴이 없으면 원본 복사본과 실패 상태를 반환합니다.
    if face is None:
        return annotated, False, message

    # bbox는 [x1, y1, x2, y2] 형식의 얼굴 영역 좌표입니다.
    x1, y1, x2, y2 = face.bbox.astype(int)

    # 이미지 범위를 벗어나지 않도록 좌표를 보정합니다.
    height, width = annotated.shape[:2]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    # 얼굴 영역에 초록색 사각형 테두리를 그립니다.
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # 사각형 위에 안내 텍스트를 표시합니다.
    cv2.putText(
        annotated,
        "FACE",
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return annotated, True, message


def extract_embedding(image_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    """이미지에서 가장 큰 얼굴 하나의 임베딩 벡터를 추출합니다."""

    # 가장 큰 얼굴 객체를 검출합니다.
    face, message = detect_largest_face(image_bgr)

    # 얼굴 검출에 실패하면 임베딩을 만들 수 없습니다.
    if face is None:
        return None, message

    # InsightFace가 제공하는 embedding은 일반적으로 512차원 얼굴 특징 벡터입니다.
    embedding = face.embedding.astype(np.float32)

    # 벡터 크기를 계산합니다.
    norm = np.linalg.norm(embedding)

    # norm이 0이면 유효하지 않은 임베딩으로 판단합니다.
    if norm == 0:
        return None, "얼굴 특징 벡터를 생성하지 못했습니다."

    # 코사인 유사도 비교를 안정적으로 하기 위해 L2 정규화를 수행합니다.
    normalized_embedding = embedding / norm

    # 정상 임베딩과 성공 메시지를 반환합니다.
    return normalized_embedding, "얼굴 특징 추출 성공"


def save_registered_face_image(user_id: str, image_bgr: np.ndarray) -> Path:
    """등록용 얼굴 이미지를 파일로 저장합니다."""

    # 저장 폴더가 없으면 생성합니다.
    ensure_dirs()

    # 사용자 ID에 파일 경로 특수문자가 들어오는 것을 피하기 위해 안전한 문자만 남깁니다.
    safe_user_id = "".join(ch for ch in user_id if ch.isalnum() or ch in ("_", "-"))

    # 저장할 이미지 파일 경로를 만듭니다.
    image_path = FACE_IMAGE_DIR / f"{safe_user_id}.jpg"

    # OpenCV imwrite로 BGR 이미지를 jpg 파일로 저장합니다.
    cv2.imwrite(str(image_path), image_bgr)

    # 저장된 경로를 반환합니다.
    return image_path


def register_face(user_id: str, password: str, user_name: str, image_bgr: np.ndarray) -> Tuple[bool, str]:
    """사용자 정보와 얼굴 이미지를 MySQL에 등록합니다."""

    # 사용자 ID 공백 입력을 방지합니다.
    if not user_id or not user_id.strip():
        return False, "아이디를 입력하세요."

    # 암호 공백 입력을 방지합니다.
    if not password or not password.strip():
        return False, "암호를 입력하세요."

    # 이름 공백 입력을 방지합니다.
    if not user_name or not user_name.strip():
        return False, "이름을 입력하세요."

    # 입력 문자열 앞뒤 공백을 제거합니다.
    user_id = user_id.strip()
    user_name = user_name.strip()

    # 이미지에서 얼굴 임베딩을 추출합니다.
    embedding, message = extract_embedding(image_bgr)

    # 임베딩 추출 실패 시 실패 메시지를 반환합니다.
    if embedding is None:
        return False, message

    # 등록 이미지 원본도 별도 폴더에 저장하여 추후 확인할 수 있게 합니다.
    image_path = save_registered_face_image(user_id, image_bgr)

    # 사용자 정보, 암호 해시, 얼굴 임베딩을 MySQL에 저장합니다.
    create_or_update_user(user_id, password, user_name, embedding, str(image_path))

    # 성공 여부와 안내 메시지를 반환합니다.
    return True, f"{user_id} 사용자의 계정과 얼굴 등록이 완료되었습니다."


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """두 정규화된 얼굴 임베딩 벡터의 코사인 유사도를 계산합니다."""

    # 두 벡터가 이미 L2 정규화되어 있으므로 내적 값이 코사인 유사도가 됩니다.
    return float(np.dot(a, b))


def verify_face_for_user(
    user_id: str,
    image_bgr: np.ndarray,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> Tuple[bool, float, str]:
    """로그인 중인 특정 사용자 ID의 등록 얼굴과 촬영 얼굴을 비교합니다."""

    # MySQL에서 해당 사용자의 등록 얼굴 임베딩을 읽습니다.
    registered_embedding = get_user_face_embedding(user_id)

    # 등록 얼굴이 없으면 실패 처리합니다.
    if registered_embedding is None:
        return False, 0.0, "등록된 얼굴 정보가 없습니다. 먼저 얼굴을 등록하세요."

    # 로그인 이미지에서 얼굴 임베딩을 추출합니다.
    login_embedding, message = extract_embedding(image_bgr)

    # 임베딩 추출에 실패하면 실패 메시지를 반환합니다.
    if login_embedding is None:
        return False, 0.0, message

    # 등록 얼굴과 로그인 얼굴의 코사인 유사도를 계산합니다.
    score = cosine_similarity(login_embedding, registered_embedding)

    # 유사도가 임계값 이상이면 2차 얼굴 인증 성공입니다.
    if score >= threshold:
        return True, score, "2차 얼굴 인증이 성공했습니다."

    # 임계값 미만이면 얼굴 인증 실패입니다.
    return False, score, "등록된 얼굴과 충분히 일치하지 않습니다."

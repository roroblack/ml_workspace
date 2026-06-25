"""InsightFace + ArcFace 기반 얼굴 등록/로그인 기능을 담당하는 모듈입니다."""

# 운영체제 경로 처리를 안전하게 하기 위해 pathlib의 Path 클래스를 가져옵니다.
from pathlib import Path

# 타입 힌트를 사용하여 함수 입력과 반환값을 명확하게 표현하기 위해 가져옵니다.
from typing import Dict, Optional, Tuple

# OpenCV는 웹캠 프레임과 이미지 파일을 읽고 BGR/RGB 변환을 수행하는 데 사용합니다.
import cv2

# InsightFace의 FaceAnalysis는 얼굴 검출, 정렬, 임베딩 추출을 한 번에 수행하는 고수준 API입니다.
from insightface.app import FaceAnalysis

# NumPy는 얼굴 임베딩 벡터 저장, 정규화, 코사인 유사도 계산에 사용합니다.
import numpy as np


# 얼굴 데이터베이스 파일이 저장될 기본 경로를 정의합니다.
FACE_DB_PATH = Path("face_db.npy")

# 등록된 얼굴 원본 이미지가 저장될 기본 폴더를 정의합니다.
FACE_IMAGE_DIR = Path("registered_faces")

# 동일 인물 여부를 판단하기 위한 기본 코사인 유사도 임계값입니다.
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

    # buffalo_l 모델팩은 RetinaFace 기반 얼굴 검출과 ArcFace 계열 얼굴 임베딩 모델을 포함합니다.
    _FACE_APP = FaceAnalysis(name="buffalo_l")

    # ctx_id=-1은 CPU 실행을 의미합니다. GPU 환경이면 ctx_id=0으로 변경할 수 있습니다.
    # det_size는 얼굴 검출 입력 크기이며, 클수록 작은 얼굴 검출에 유리하지만 속도는 느려질 수 있습니다.
    _FACE_APP.prepare(ctx_id=-1, det_size=(640, 640))

    # 초기화된 모델 객체를 반환합니다.
    return _FACE_APP


def ensure_dirs() -> None:
    """얼굴 이미지 저장 폴더를 생성합니다."""

    # parents=True는 상위 폴더가 없어도 함께 만들도록 하고, exist_ok=True는 이미 있어도 오류를 내지 않습니다.
    FACE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def load_face_db() -> Dict[str, np.ndarray]:
    """저장된 얼굴 임베딩 데이터베이스를 불러옵니다."""

    # 얼굴 DB 파일이 없으면 빈 딕셔너리를 반환하여 최초 실행도 오류 없이 처리합니다.
    if not FACE_DB_PATH.exists():
        return {}

    # allow_pickle=True는 딕셔너리 형태로 저장한 NumPy 객체를 다시 읽기 위해 필요합니다.
    data = np.load(FACE_DB_PATH, allow_pickle=True).item()

    # 저장 파일이 손상되었거나 딕셔너리가 아니면 빈 DB로 안전하게 처리합니다.
    if not isinstance(data, dict):
        return {}

    # 정상적으로 읽은 사용자별 임베딩 딕셔너리를 반환합니다.
    return data


def save_face_db(face_db: Dict[str, np.ndarray]) -> None:
    """사용자별 얼굴 임베딩 데이터베이스를 파일로 저장합니다."""

    # NumPy save는 딕셔너리 객체를 npy 파일로 저장할 수 있습니다.
    np.save(FACE_DB_PATH, face_db)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """OpenCV BGR 이미지를 InsightFace 입력에 적합한 RGB 이미지로 변환합니다."""

    # OpenCV는 기본적으로 BGR 색상 순서를 사용하므로 RGB로 변환합니다.
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def extract_embedding(image_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    """이미지에서 가장 큰 얼굴 하나의 임베딩 벡터를 추출합니다."""

    # InsightFace 모델 객체를 가져옵니다.
    app = get_face_app()

    # OpenCV 이미지가 비어 있으면 얼굴 추출을 진행할 수 없으므로 오류 메시지를 반환합니다.
    if image_bgr is None or image_bgr.size == 0:
        return None, "이미지를 읽을 수 없습니다."

    # InsightFace는 RGB 이미지를 기준으로 얼굴을 분석하므로 색상 순서를 변환합니다.
    image_rgb = bgr_to_rgb(image_bgr)

    # 이미지에서 얼굴 후보 목록을 추출합니다.
    faces = app.get(image_rgb)

    # 검출된 얼굴이 없으면 사용자에게 재촬영을 안내할 수 있도록 메시지를 반환합니다.
    if len(faces) == 0:
        return None, "얼굴을 찾지 못했습니다. 정면 얼굴이 잘 보이는 이미지로 다시 시도하세요."

    # 여러 얼굴이 검출되면 로그인 보안을 위해 가장 큰 얼굴 하나를 기준으로 사용합니다.
    largest_face = max(
        faces,
        key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]),
    )

    # InsightFace가 제공하는 embedding은 일반적으로 512차원 얼굴 특징 벡터입니다.
    embedding = largest_face.embedding.astype(np.float32)

    # 벡터 크기를 계산합니다. 0이면 정규화할 수 없으므로 실패 처리합니다.
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


def register_face(user_id: str, image_bgr: np.ndarray) -> Tuple[bool, str]:
    """사용자 ID와 얼굴 이미지를 이용해 얼굴 로그인을 등록합니다."""

    # 공백만 있는 사용자 ID는 허용하지 않습니다.
    if not user_id or not user_id.strip():
        return False, "사용자 ID를 입력하세요."

    # 입력된 사용자 ID의 앞뒤 공백을 제거합니다.
    user_id = user_id.strip()

    # 이미지에서 얼굴 임베딩을 추출합니다.
    embedding, message = extract_embedding(image_bgr)

    # 임베딩 추출에 실패하면 실패 메시지를 그대로 반환합니다.
    if embedding is None:
        return False, message

    # 기존 얼굴 데이터베이스를 불러옵니다.
    face_db = load_face_db()

    # 사용자 ID를 키로 하여 얼굴 임베딩을 저장합니다.
    face_db[user_id] = embedding

    # 갱신된 얼굴 데이터베이스를 파일에 저장합니다.
    save_face_db(face_db)

    # 등록 이미지 원본도 별도 폴더에 저장하여 추후 확인할 수 있게 합니다.
    save_registered_face_image(user_id, image_bgr)

    # 성공 여부와 안내 메시지를 반환합니다.
    return True, f"{user_id} 사용자의 얼굴 등록이 완료되었습니다."


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """두 정규화된 얼굴 임베딩 벡터의 코사인 유사도를 계산합니다."""

    # 두 벡터가 이미 L2 정규화되어 있으므로 내적 값이 코사인 유사도가 됩니다.
    return float(np.dot(a, b))


def verify_face(image_bgr: np.ndarray, threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> Tuple[bool, Optional[str], float, str]:
    """로그인 시 촬영한 얼굴과 등록된 얼굴 DB를 비교합니다."""

    # 저장된 얼굴 데이터베이스를 불러옵니다.
    face_db = load_face_db()

    # 등록된 얼굴이 하나도 없으면 로그인할 수 없으므로 실패 처리합니다.
    if len(face_db) == 0:
        return False, None, 0.0, "등록된 얼굴이 없습니다. 먼저 얼굴을 등록하세요."

    # 로그인 이미지에서 얼굴 임베딩을 추출합니다.
    embedding, message = extract_embedding(image_bgr)

    # 임베딩 추출에 실패하면 실패 메시지를 반환합니다.
    if embedding is None:
        return False, None, 0.0, message

    # 최고 유사도 사용자의 ID를 저장할 변수를 초기화합니다.
    best_user_id: Optional[str] = None

    # 최고 유사도 값을 저장할 변수를 매우 낮은 값으로 초기화합니다.
    best_score = -1.0

    # 등록된 모든 사용자 얼굴과 입력 얼굴을 비교합니다.
    for user_id, registered_embedding in face_db.items():
        # 현재 사용자와의 코사인 유사도를 계산합니다.
        score = cosine_similarity(embedding, registered_embedding)

        # 현재 점수가 최고 점수보다 높으면 최고 후보를 갱신합니다.
        if score > best_score:
            best_score = score
            best_user_id = user_id

    # 최고 유사도가 임계값 이상이면 동일 인물로 판단합니다.
    if best_score >= threshold:
        return True, best_user_id, best_score, "얼굴 로그인이 성공했습니다."

    # 임계값 미만이면 등록된 사용자와 일치하지 않는 것으로 판단합니다.
    return False, best_user_id, best_score, "등록된 얼굴과 충분히 일치하지 않습니다."


def read_uploaded_image(uploaded_file) -> Optional[np.ndarray]:
    """Streamlit 업로드 파일을 OpenCV BGR 이미지로 변환합니다."""

    # 업로드 파일이 없으면 None을 반환합니다.
    if uploaded_file is None:
        return None

    # 업로드된 바이트 데이터를 NumPy 배열로 변환합니다.
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)

    # OpenCV imdecode로 이미지 파일 바이트를 BGR 이미지 배열로 변환합니다.
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 변환된 이미지를 반환합니다.
    return image_bgr

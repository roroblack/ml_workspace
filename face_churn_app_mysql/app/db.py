"""MySQL 사용자 계정과 얼굴 임베딩 저장 기능을 담당하는 모듈입니다."""

# 환경 변수에서 DB 접속 정보를 읽기 위해 os 모듈을 사용합니다.
import os

from pathlib import Path

from dotenv import load_dotenv

# 비밀번호를 평문으로 저장하지 않기 위해 안전한 해시 계산에 hashlib를 사용합니다.
import hashlib

# 비밀번호 해시용 임의 salt를 생성하기 위해 secrets를 사용합니다.
import secrets

# 타입 힌트를 위해 Optional, Tuple을 사용합니다.
from typing import Optional, Tuple

# NumPy 배열인 얼굴 임베딩을 bytes로 변환하고 복원하기 위해 numpy를 사용합니다.
import numpy as np

# MySQL 접속을 위해 mysql-connector-python 패키지를 사용합니다.
import mysql.connector


# 사용자 테이블 이름을 상수로 정의합니다.
USER_TABLE = "face_users"

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


# DB 접속 기본값을 환경 변수에서 읽습니다.
# 실제 운영 환경에서는 아래 환경 변수를 사용자의 MySQL 환경에 맞게 설정하면 됩니다.
DB_HOST = os.getenv("MYSQL_HOST", "localhost")
DB_PORT = int(os.getenv("MYSQL_PORT", "3306"))
DB_USER = os.getenv("MYSQL_USER", "root")
DB_PASSWORD = os.getenv("MYSQL_PASSWORD", "1234")
DB_NAME = os.getenv("MYSQL_DATABASE", "face_churn_db")


def get_connection(database: Optional[str] = DB_NAME):
    """MySQL 연결 객체를 생성하여 반환합니다."""

    # mysql.connector.connect는 MySQL 서버에 접속하는 함수입니다.
    # database=None이면 특정 DB 선택 없이 서버에만 접속합니다.
    return mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=database,
        charset="utf8mb4",
        use_unicode=True,
    )


def init_db() -> None:
    """DB와 사용자 테이블이 없으면 자동으로 생성합니다."""

    # 먼저 database 없이 MySQL 서버에 접속합니다.
    conn = get_connection(database=None)

    # cursor는 SQL 문장을 실행하는 객체입니다.
    cur = conn.cursor()

    # 지정한 데이터베이스가 없으면 생성합니다.
    cur.execute(
        f"CREATE DATABASE IF NOT EXISTS {DB_NAME} "
        "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    )

    # DB 생성 명령을 실제 반영합니다.
    conn.commit()

    # cursor와 연결을 닫아 자원을 해제합니다.
    cur.close()
    conn.close()

    # 생성된 데이터베이스에 다시 접속합니다.
    conn = get_connection(database=DB_NAME)
    cur = conn.cursor()

    # 사용자 계정, 비밀번호 해시, 이름, 얼굴 임베딩을 저장할 테이블을 생성합니다.
    # face_embedding은 512차원 float32 벡터를 bytes로 변환한 값을 저장합니다.
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {USER_TABLE} (
            user_id VARCHAR(100) PRIMARY KEY,
            user_name VARCHAR(100) NOT NULL,
            password_salt VARCHAR(64) NOT NULL,
            password_hash VARCHAR(128) NOT NULL,
            face_embedding LONGBLOB NOT NULL,
            face_image_path VARCHAR(500),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
    )

    conn.commit()
    cur.close()
    conn.close()


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """비밀번호를 PBKDF2 방식으로 해시 처리합니다."""

    # salt가 없으면 32자리 난수 문자열을 새로 생성합니다.
    if salt is None:
        salt = secrets.token_hex(16)

    # pbkdf2_hmac은 반복 해시를 수행하여 무차별 대입 공격을 어렵게 합니다.
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000,
    )

    # bytes 결과를 16진수 문자열로 변환하여 DB에 저장하기 쉽게 만듭니다.
    return salt, digest.hex()


def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """얼굴 임베딩 NumPy 배열을 MySQL BLOB 저장용 bytes로 변환합니다."""

    # float32 타입으로 고정해야 저장/복원 시 크기와 값이 안정적입니다.
    return embedding.astype(np.float32).tobytes()


def bytes_to_embedding(blob: bytes) -> np.ndarray:
    """MySQL BLOB에서 읽은 bytes를 얼굴 임베딩 NumPy 배열로 복원합니다."""

    # frombuffer는 bytes 데이터를 float32 배열로 해석합니다.
    embedding = np.frombuffer(blob, dtype=np.float32)

    # 코사인 유사도 계산 안정성을 위해 다시 정규화합니다.
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def user_exists(user_id: str) -> bool:
    """사용자 ID가 이미 등록되어 있는지 확인합니다."""

    init_db()
    conn = get_connection()
    cur = conn.cursor()

    # COUNT(*)로 해당 ID의 존재 여부를 확인합니다.
    cur.execute(f"SELECT COUNT(*) FROM {USER_TABLE} WHERE user_id = %s", (user_id,))
    count = cur.fetchone()[0]

    cur.close()
    conn.close()

    return count > 0


def create_or_update_user(user_id: str, password: str, user_name: str, embedding: np.ndarray, image_path: str) -> None:
    """사용자 정보와 얼굴 임베딩을 MySQL에 저장합니다."""

    init_db()

    # 입력 비밀번호를 salt + hash 형태로 변환합니다.
    salt, password_hash = hash_password(password)

    # 얼굴 임베딩 배열을 BLOB 저장용 bytes로 변환합니다.
    embedding_blob = embedding_to_bytes(embedding)

    conn = get_connection()
    cur = conn.cursor()

    # 같은 user_id가 있으면 사용자 정보와 얼굴 정보를 갱신합니다.
    cur.execute(
        f"""
        INSERT INTO {USER_TABLE}
            (user_id, user_name, password_salt, password_hash, face_embedding, face_image_path)
        VALUES
            (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            user_name = VALUES(user_name),
            password_salt = VALUES(password_salt),
            password_hash = VALUES(password_hash),
            face_embedding = VALUES(face_embedding),
            face_image_path = VALUES(face_image_path)
        """,
        (user_id, user_name, salt, password_hash, embedding_blob, image_path),
    )

    conn.commit()
    cur.close()
    conn.close()


def verify_user_password(user_id: str, password: str) -> Tuple[bool, Optional[str], str]:
    """아이디와 비밀번호가 MySQL에 저장된 값과 일치하는지 확인합니다."""

    init_db()
    conn = get_connection()
    cur = conn.cursor(dictionary=True)

    # 입력된 user_id에 해당하는 사용자 정보를 조회합니다.
    cur.execute(
        f"SELECT user_id, user_name, password_salt, password_hash FROM {USER_TABLE} WHERE user_id = %s",
        (user_id,),
    )
    row = cur.fetchone()

    cur.close()
    conn.close()

    # 사용자가 없으면 실패를 반환합니다.
    if row is None:
        return False, None, "등록되지 않은 아이디입니다."

    # DB에 저장된 salt로 입력 비밀번호를 다시 해시합니다.
    _, input_hash = hash_password(password, salt=row["password_salt"])

    # secrets.compare_digest는 문자열 비교 시 타이밍 공격 위험을 줄입니다.
    if not secrets.compare_digest(input_hash, row["password_hash"]):
        return False, None, "암호가 일치하지 않습니다."

    return True, row["user_name"], "아이디와 암호 확인이 완료되었습니다."


def get_user_face_embedding(user_id: str) -> Optional[np.ndarray]:
    """특정 사용자 ID의 등록 얼굴 임베딩을 MySQL에서 읽어옵니다."""

    init_db()
    conn = get_connection()
    cur = conn.cursor()

    # user_id에 해당하는 얼굴 임베딩 BLOB을 조회합니다.
    cur.execute(f"SELECT face_embedding FROM {USER_TABLE} WHERE user_id = %s", (user_id,))
    row = cur.fetchone()

    cur.close()
    conn.close()

    # 조회 결과가 없으면 None을 반환합니다.
    if row is None or row[0] is None:
        return None

    # BLOB 데이터를 NumPy 임베딩 배열로 복원합니다.
    return bytes_to_embedding(row[0])

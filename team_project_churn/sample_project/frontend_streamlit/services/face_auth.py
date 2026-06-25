# -*- coding: utf-8 -*-
"""얼굴 로그인 서비스 (샘플 스텁).
실제 InsightFace 구현은 이 인터페이스(register/verify)에 주입한다.
샘플에서는 카메라/생체인식 없이 user_id 존재 여부로 로그인하고, 결과를 face_login_log에 남긴다.
(opencv_face_login 연동 시 verify(image)->user_id,similarity 로 교체)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from db import db_client as db

# 실제 연동 시 교체: InsightFace FaceAnalysis("buffalo_l"), 512d, cosine >= 0.45
FACE_AVAILABLE = False


def verify(user_id, image=None):
    """샘플 스텁: face_user에 존재하면 로그인 성공.
    실제 구현: image에서 임베딩 추출 → 후보와 cosine similarity → 임계 비교."""
    u = db.get_face_user(user_id)
    if u is None:
        db.log_login(user_id, False, reason="unknown_user")
        return {"success": False, "user_id": user_id, "role": None, "similarity": None,
                "reason": "등록되지 않은 사용자"}
    db.log_login(user_id, True, similarity=1.0)
    return {"success": True, "user_id": user_id, "role": u["role"], "similarity": 1.0, "reason": None}


def register(user_id, display_name, role="customer", image=None):
    db.upsert_face_user(user_id, display_name, role)
    return {"success": True, "user_id": user_id}

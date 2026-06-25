# -*- coding: utf-8 -*-
"""DB 접근 얇은 래퍼 — 프론트는 이 모듈만 import.
샘플은 로컬 SQLite(db_client)를 직접 호출. (실배포 시 Vercel API 호출로 교체 가능)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from db import db_client as db

get_face_user = db.get_face_user
all_user_ids = db.all_user_ids
recent_events = db.recent_events
latest_prediction_for = db.latest_prediction_for
query = db.query

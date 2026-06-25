# -*- coding: utf-8 -*-
"""샘플 프로젝트 공통 설정 (경로·파라미터)."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RAW = os.path.join(DATA, "raw")
PROC = os.path.join(DATA, "processed")
MODELS = os.path.join(HERE, "models")
DB_PATH = os.path.join(DATA, "churn.db")          # SQLite 기본
for d in (RAW, PROC, MODELS):
    os.makedirs(d, exist_ok=True)

# DATABASE_URL(postgres://...) 가 있으면 Postgres, 없으면 SQLite
DATABASE_URL = os.environ.get("DATABASE_URL", "")

SEED = 42
EVENT_TYPES = ["view", "cart", "purchase"]

# 라벨링/시퀀스
OBS_DAYS = 14            # 관찰 기간
OUTCOME_DAYS = 7         # 결과 기간(이 기간 무활동=이탈)
SEQ_LEN = OBS_DAYS       # 일별 시퀀스 길이
N_SEQ_FEATURES = len(EVENT_TYPES)

# 위험 등급 임계
RISK_LOW = 0.35
RISK_HIGH = 0.65

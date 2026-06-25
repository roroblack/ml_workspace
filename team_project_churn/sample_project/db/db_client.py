# -*- coding: utf-8 -*-
"""DB 접속·초기화·조회 공통 함수.
기본은 SQLite(data/churn.db). 환경변수 DATABASE_URL(postgres) 있으면 Postgres(psycopg) 시도.
v5의 '운영 데이터는 DB / 모델·시퀀스는 파일(경로만)' 원칙을 따른다.
"""
import os, sys, sqlite3, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

IS_PG = config.DATABASE_URL.startswith("postgres")


def get_conn():
    if IS_PG:
        import psycopg  # 선택 의존성
        return psycopg.connect(config.DATABASE_URL)
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS realtime_event_log (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL, session_id TEXT, event_type TEXT NOT NULL,
    product_id TEXT, price REAL, event_time TEXT NOT NULL,
    source TEXT DEFAULT 'sim', value_json TEXT,
    received_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS feature_user_snapshot (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL, snapshot_time TEXT, label INTEGER,
    feature_json TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS sequence_snapshot (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL, seq_len INTEGER, n_features INTEGER,
    storage_format TEXT DEFAULT 'npz', artifact_path TEXT, row_index INTEGER,
    label INTEGER, seq_type TEXT DEFAULT 'daily',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS model_registry (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL, model_type TEXT NOT NULL, artifact_path TEXT NOT NULL,
    metrics_json TEXT, is_active INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS prediction_log (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER, user_id TEXT NOT NULL, session_id TEXT,
    churn_probability REAL NOT NULL, risk_level TEXT NOT NULL,
    top_factors_json TEXT, recommended_action TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS face_user (
    user_id TEXT PRIMARY KEY, display_name TEXT, role TEXT DEFAULT 'customer',
    is_active INTEGER DEFAULT 1, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS face_login_log (
    login_id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT,
    success INTEGER NOT NULL, similarity REAL, failure_reason TEXT,
    login_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS retention_action_log (
    action_id INTEGER PRIMARY KEY AUTOINCREMENT, prediction_id INTEGER,
    user_id TEXT NOT NULL, action_type TEXT, action_message TEXT,
    status TEXT DEFAULT 'suggested', created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE INDEX IF NOT EXISTS idx_evt_user ON realtime_event_log(user_id);
CREATE INDEX IF NOT EXISTS idx_pred_user ON prediction_log(user_id);
"""


def init_db():
    if IS_PG:
        raise NotImplementedError("샘플은 SQLite 기준. Postgres(Neon)는 reports/05-5의 DDL을 사용하세요.")
    conn = get_conn()
    conn.executescript(SCHEMA_SQLITE)
    conn.commit(); conn.close()


def _ph(n):
    return ",".join(["%s" if IS_PG else "?"] * n)


def execute(sql, params=()):
    conn = get_conn(); cur = conn.cursor()
    cur.execute(sql, params); conn.commit()
    last = None if IS_PG else cur.lastrowid
    conn.close(); return last


def query(sql, params=()):
    conn = get_conn(); cur = conn.cursor()
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    conn.close(); return rows


def insert_event(user_id, event_type, product_id, price, event_time, session_id, source="sim"):
    return execute(
        "INSERT INTO realtime_event_log(user_id,session_id,event_type,product_id,price,event_time,source) "
        f"VALUES({_ph(7)})",
        (user_id, session_id, event_type, product_id, price, event_time, source))


def recent_events(user_id, limit=500):
    rows = query(
        "SELECT user_id,event_type,product_id,price,event_time FROM realtime_event_log "
        f"WHERE user_id={_ph(1)} ORDER BY event_time", (user_id,))
    return rows[-limit:]


def all_user_ids():
    return [r["user_id"] for r in query("SELECT DISTINCT user_id FROM realtime_event_log")]


def register_model(name, mtype, path, metrics, active=True):
    if active:
        execute(f"UPDATE model_registry SET is_active=0 WHERE model_type={_ph(1)}", (mtype,))
    return execute(
        "INSERT INTO model_registry(model_name,model_type,artifact_path,metrics_json,is_active) "
        f"VALUES({_ph(5)})", (name, mtype, path, json.dumps(metrics), 1 if active else 0))


def active_model(mtype):
    r = query(f"SELECT * FROM model_registry WHERE model_type={_ph(1)} AND is_active=1 ORDER BY model_id DESC", (mtype,))
    return r[0] if r else None


def log_prediction(model_id, user_id, prob, risk, factors, action):
    return execute(
        "INSERT INTO prediction_log(model_id,user_id,churn_probability,risk_level,top_factors_json,recommended_action) "
        f"VALUES({_ph(6)})", (model_id, user_id, prob, risk, json.dumps(factors, ensure_ascii=False), action))


def latest_predictions(limit=200):
    return query("SELECT * FROM prediction_log ORDER BY prediction_id DESC")[:limit]


def latest_prediction_for(user_id):
    r = query(f"SELECT * FROM prediction_log WHERE user_id={_ph(1)} ORDER BY prediction_id DESC", (user_id,))
    return r[0] if r else None


def log_retention(prediction_id, user_id, action_type, msg):
    return execute(
        "INSERT INTO retention_action_log(prediction_id,user_id,action_type,action_message) "
        f"VALUES({_ph(4)})", (prediction_id, user_id, action_type, msg))


def upsert_face_user(user_id, display_name, role="customer"):
    if not query(f"SELECT 1 FROM face_user WHERE user_id={_ph(1)}", (user_id,)):
        execute(f"INSERT INTO face_user(user_id,display_name,role) VALUES({_ph(3)})", (user_id, display_name, role))


def get_face_user(user_id):
    r = query(f"SELECT * FROM face_user WHERE user_id={_ph(1)}", (user_id,))
    return r[0] if r else None


def log_login(user_id, success, similarity=None, reason=None):
    return execute(
        f"INSERT INTO face_login_log(user_id,success,similarity,failure_reason) VALUES({_ph(4)})",
        (user_id, 1 if success else 0, similarity, reason))

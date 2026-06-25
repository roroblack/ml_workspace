-- 샘플 프로젝트 DB 스키마 (SQLite 기준)
-- db_client.init_db()가 동일 내용을 생성한다. Postgres(Neon)는 reports/05-5의 DDL 사용.
-- 원칙: 운영 데이터는 DB / 모델·시퀀스 원본은 파일(경로만 저장).

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

-- 시퀀스 원본은 파일(npz), DB엔 경로/메타만
CREATE TABLE IF NOT EXISTS sequence_snapshot (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL, seq_len INTEGER, n_features INTEGER,
    storage_format TEXT DEFAULT 'npz', artifact_path TEXT, row_index INTEGER,
    label INTEGER, seq_type TEXT DEFAULT 'daily',   -- v3 추가형: 'daily'|'event' (기존행 자동 daily)
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

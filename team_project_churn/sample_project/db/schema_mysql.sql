-- ============================================================
-- 운영 DB(MySQL 8) DDL — churn_ops
-- 원칙: 운영 데이터만 DB. 1.27M 유저 피처/시퀀스/모델바이너리는 파일(경로만).
-- 기준: reports/05-6 §4, reports/19 §3, reports/17-6-1
-- ============================================================
-- CREATE DATABASE IF NOT EXISTS churn_ops CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
-- USE churn_ops;

-- ---- 모델 레지스트리 (17-6-1 확장: preprocessing_config / dataset_path / metric_cv,oot / model_type) ----
CREATE TABLE IF NOT EXISTS model_registry (
  model_id               BIGINT AUTO_INCREMENT PRIMARY KEY,
  model_name             VARCHAR(128) NOT NULL,
  model_type             VARCHAR(32)  NOT NULL,            -- tree | linear | sequence | ensemble
  feature_schema_version VARCHAR(16)  DEFAULT 'v2',
  preprocessing_config   JSON,                              -- 서빙 전처리 분기 근거(scale/scaler_path/label/seq...)
  dataset_path           TEXT,                              -- 학습 데이터셋 파일 경로(DB 밖)
  artifact_path          TEXT NOT NULL,                     -- 모델 바이너리 파일 경로(DB 밖)
  train_period           VARCHAR(64),
  metric_cv              DOUBLE,
  metric_oot             DOUBLE,
  metrics_json           JSON,
  is_active              TINYINT(1) DEFAULT 0,
  created_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uq_model_name (model_name),
  KEY idx_model_active (model_type, is_active)
) ENGINE=InnoDB;

-- ---- 예측 로그 ----
CREATE TABLE IF NOT EXISTS prediction_log (
  prediction_id     BIGINT AUTO_INCREMENT PRIMARY KEY,
  model_id          BIGINT,
  user_id           VARCHAR(64) NOT NULL,
  session_id        VARCHAR(128),
  churn_probability DOUBLE NOT NULL,
  risk_level        VARCHAR(16) NOT NULL,
  top_factors_json  JSON,
  recommended_action TEXT,
  created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  KEY idx_pred_user (user_id),
  CONSTRAINT fk_pred_model FOREIGN KEY (model_id) REFERENCES model_registry(model_id)
) ENGINE=InnoDB;

-- ---- 피처 스냅샷 (17-6-1: cohort_flag, churn, churn_no_purchase, obs/outcome_period) ----
-- 1.27M 전체는 파일. DB엔 운영 대상 유저 요약만.
CREATE TABLE IF NOT EXISTS feature_user_snapshot (
  snapshot_id      BIGINT AUTO_INCREMENT PRIMARY KEY,
  user_id          VARCHAR(64) NOT NULL,
  snapshot_time    TIMESTAMP NULL,
  cohort_flag      TINYINT(1),
  churn            TINYINT(1),
  churn_no_purchase TINYINT(1),
  obs_period       VARCHAR(64),
  outcome_period   VARCHAR(64),
  feature_json     JSON,                                    -- 10피처 canonical(ndays...)
  created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  KEY idx_feat_user (user_id)
) ENGINE=InnoDB;

-- ---- 시퀀스 스냅샷 (원본 npz는 파일. 경로/메타만) ----
CREATE TABLE IF NOT EXISTS sequence_snapshot (
  snapshot_id    BIGINT AUTO_INCREMENT PRIMARY KEY,
  user_id        VARCHAR(64) NOT NULL,
  dataset_tag    VARCHAR(32),                               -- 다중 시퀀스 데이터셋 구분
  seq_len        INT,
  n_features     INT,
  storage_format VARCHAR(16) DEFAULT 'npz',
  artifact_path  TEXT,                                      -- DB 밖 파일
  row_index      INT,
  label          INT,
  created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  KEY idx_seq_user (user_id)
) ENGINE=InnoDB;

-- ---- 추천 (17-6-1 신규) ----
CREATE TABLE IF NOT EXISTS user_interest (
  user_id         VARCHAR(64) PRIMARY KEY,
  top_category_id VARCHAR(64),
  top_brand       VARCHAR(128),
  interest_json   JSON,
  updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS recommendation (
  rec_id              BIGINT AUTO_INCREMENT PRIMARY KEY,
  user_id             VARCHAR(64) NOT NULL,
  model_id            BIGINT,
  rec_items_json      JSON,
  rec_categories_json JSON,
  created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  KEY idx_rec_user (user_id),
  CONSTRAINT fk_rec_model FOREIGN KEY (model_id) REFERENCES model_registry(model_id)
) ENGINE=InnoDB;

-- ---- 앙상블 ----
CREATE TABLE IF NOT EXISTS ensemble_result (
  ensemble_id      BIGINT AUTO_INCREMENT PRIMARY KEY,
  user_id          VARCHAR(64) NOT NULL,
  prob_ensemble    DOUBLE NOT NULL,
  risk_level       VARCHAR(16),
  improvement_json JSON,                                    -- 개선점(어느 모델이 무엇을 보완)
  created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  KEY idx_ens_user (user_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS ensemble_member (
  member_id   BIGINT AUTO_INCREMENT PRIMARY KEY,
  ensemble_id BIGINT NOT NULL,
  model_id    BIGINT,
  weight      DOUBLE,
  prob        DOUBLE,
  CONSTRAINT fk_em_ens FOREIGN KEY (ensemble_id) REFERENCES ensemble_result(ensemble_id)
) ENGINE=InnoDB;

-- ---- 리텐션 액션 ----
CREATE TABLE IF NOT EXISTS retention_action_log (
  action_id      BIGINT AUTO_INCREMENT PRIMARY KEY,
  prediction_id  BIGINT,
  user_id        VARCHAR(64) NOT NULL,
  action_type    VARCHAR(64),
  action_message TEXT,
  status         VARCHAR(32) DEFAULT 'suggested',
  created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_act_pred FOREIGN KEY (prediction_id) REFERENCES prediction_log(prediction_id)
) ENGINE=InnoDB;

-- ---- 얼굴 로그인(인증/권한만, 학습 미사용) ----
CREATE TABLE IF NOT EXISTS face_user (
  user_id      VARCHAR(64) PRIMARY KEY,
  display_name VARCHAR(100),
  role         VARCHAR(32) DEFAULT 'customer',
  is_active    TINYINT(1) DEFAULT 1,
  created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS face_login_log (
  login_id       BIGINT AUTO_INCREMENT PRIMARY KEY,
  user_id        VARCHAR(64),
  success        TINYINT(1) NOT NULL,
  similarity     DOUBLE,
  failure_reason TEXT,
  login_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- ---- (옵션) Neon pull cursor 영속화 ----
CREATE TABLE IF NOT EXISTS pull_cursor (
  source     VARCHAR(64) PRIMARY KEY,
  last_ts    TIMESTAMP NULL,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;

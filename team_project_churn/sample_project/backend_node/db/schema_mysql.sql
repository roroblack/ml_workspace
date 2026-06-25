-- 가지마 운영 DB (MySQL 8) — 19-1 설계 반영. 대용량(parquet/npz/model)은 파일, DB엔 경로.
-- CREATE DATABASE IF NOT EXISTS gajima CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci; USE gajima;

CREATE TABLE IF NOT EXISTS model_registry (
  model_id BIGINT AUTO_INCREMENT PRIMARY KEY,
  model_name VARCHAR(128) NOT NULL UNIQUE,
  model_type VARCHAR(32) NOT NULL,                 -- tree|linear|sequence|ensemble
  feature_schema_version VARCHAR(16) DEFAULT 'v2',
  label_name VARCHAR(32) DEFAULT 'churn',
  horizon_days INT DEFAULT 7,
  preprocessing_config JSON,
  dataset_path TEXT,
  artifact_path TEXT NOT NULL,
  metrics_json JSON,
  is_active TINYINT(1) DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  KEY idx_active (model_type, is_active)
) ENGINE=InnoDB;

-- 19-1 §6.3: 모델 평가/차트 산출물
CREATE TABLE IF NOT EXISTS model_evaluation (
  eval_id BIGINT AUTO_INCREMENT PRIMARY KEY,
  model_id BIGINT NOT NULL,
  dataset_tag VARCHAR(64) NOT NULL,
  split_name VARCHAR(32) DEFAULT 'test',
  horizon_days INT DEFAULT 7,
  label_name VARCHAR(32) DEFAULT 'churn',
  n_samples BIGINT, positive_rate DOUBLE,
  roc_auc DOUBLE, pr_auc DOUBLE, best_threshold DOUBLE, best_f1 DOUBLE,
  confusion_json JSON, threshold_curve_json JSON, calibration_json JSON,
  lift_curve_json JSON, score_distribution_json JSON, training_history_json JSON,
  eval_predictions_path TEXT, shap_summary_path TEXT, business_value_path TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_eval_model FOREIGN KEY (model_id) REFERENCES model_registry(model_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS prediction_log (
  prediction_id BIGINT AUTO_INCREMENT PRIMARY KEY,
  model_id BIGINT, user_id VARCHAR(64) NOT NULL, session_id VARCHAR(128),
  churn_probability DOUBLE NOT NULL, risk_level VARCHAR(16) NOT NULL,
  horizon_days INT DEFAULT 7, top_factors_json JSON, recommended_action TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, KEY idx_pred_user (user_id),
  CONSTRAINT fk_pred_model FOREIGN KEY (model_id) REFERENCES model_registry(model_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS feature_user_snapshot (
  snapshot_id BIGINT AUTO_INCREMENT PRIMARY KEY, user_id VARCHAR(64) NOT NULL,
  snapshot_time TIMESTAMP NULL, cohort_flag TINYINT(1), churn TINYINT(1),
  obs_period VARCHAR(64), outcome_period VARCHAR(64), feature_json JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, KEY idx_feat_user (user_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS sequence_snapshot (
  snapshot_id BIGINT AUTO_INCREMENT PRIMARY KEY, user_id VARCHAR(64) NOT NULL,
  dataset_tag VARCHAR(32), seq_type VARCHAR(16) DEFAULT 'weekly',
  seq_len INT, n_features INT, storage_format VARCHAR(16) DEFAULT 'npz',
  artifact_path TEXT, row_index INT, label INT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, KEY idx_seq_user (user_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS user_interest (
  user_id VARCHAR(64) PRIMARY KEY, top_category_id VARCHAR(64), top_brand VARCHAR(128),
  interest_json JSON, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS recommendation (
  rec_id BIGINT AUTO_INCREMENT PRIMARY KEY, user_id VARCHAR(64) NOT NULL, model_id BIGINT,
  rec_items_json JSON, rec_categories_json JSON, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  KEY idx_rec_user (user_id), CONSTRAINT fk_rec_model FOREIGN KEY (model_id) REFERENCES model_registry(model_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS retention_action_log (
  action_id BIGINT AUTO_INCREMENT PRIMARY KEY, prediction_id BIGINT, user_id VARCHAR(64) NOT NULL,
  action_type VARCHAR(64), action_message TEXT, status VARCHAR(32) DEFAULT 'suggested',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_act_pred FOREIGN KEY (prediction_id) REFERENCES prediction_log(prediction_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS ensemble_result (
  ensemble_id BIGINT AUTO_INCREMENT PRIMARY KEY, user_id VARCHAR(64) NOT NULL,
  prob_ensemble DOUBLE NOT NULL, risk_level VARCHAR(16), improvement_json JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, KEY idx_ens_user (user_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS ensemble_member (
  member_id BIGINT AUTO_INCREMENT PRIMARY KEY, ensemble_id BIGINT NOT NULL,
  model_id BIGINT, weight DOUBLE, prob DOUBLE,
  CONSTRAINT fk_em_ens FOREIGN KEY (ensemble_id) REFERENCES ensemble_result(ensemble_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS face_user (
  user_id VARCHAR(64) PRIMARY KEY, display_name VARCHAR(100), role VARCHAR(32) DEFAULT 'customer',
  embedding LONGBLOB, is_active TINYINT(1) DEFAULT 1, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS face_login_log (
  login_id BIGINT AUTO_INCREMENT PRIMARY KEY, user_id VARCHAR(64), success TINYINT(1) NOT NULL,
  similarity DOUBLE, failure_reason TEXT, login_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

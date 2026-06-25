-- 데모 시드 (운영 DB). 실제 모델 바이너리는 파일, 여기엔 경로/메타만.
INSERT INTO face_user (user_id, display_name, role) VALUES
  ('admin',  '관리자',   'admin'),
  ('demo01', '데모고객1', 'customer'),
  ('demo02', '데모고객2', 'customer')
ON DUPLICATE KEY UPDATE display_name=VALUES(display_name);

INSERT INTO model_registry
  (model_name, model_type, feature_schema_version, preprocessing_config, dataset_path, artifact_path, train_period, metric_cv, metric_oot, is_active)
VALUES
  ('CatBoost_ES_Feb', 'tree', 'v2',
   JSON_OBJECT('scale','none','label','churn'),
   'sample_project/data/processed_5m/models7/CatBoost_train.parquet',
   'models/tabular/catboost_es_feb.cbm', '2019-10-01..2020-01-31', 0.7902, 0.7902, 1),
  ('LogReg_5m_opt', 'linear', 'v2',
   JSON_OBJECT('scale','log1p+minmax','scaler_path','models/preprocessors/ML_opt_scaler.joblib','label','churn'),
   'sample_project/data/processed_5m/models7/LogReg_train.parquet',
   'models/tabular/logreg_5m_opt.joblib', '2019-10-01..2020-01-31', 0.7856, 0.7856, 0),
  ('Transformer_5m', 'sequence', 'v2',
   JSON_OBJECT('scale','raw','seq_granularity','weekly','seq_len',17,'mask',true,'label','churn'),
   'sample_project/data/processed_5m/models7/Transformer_train_seq.npz',
   'models/sequence/transformer_5m.pth', '2019-10-01..2020-01-31', 0.7878, 0.7878, 0)
ON DUPLICATE KEY UPDATE metric_oot=VALUES(metric_oot);

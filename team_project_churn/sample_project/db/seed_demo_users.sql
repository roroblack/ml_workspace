-- 데모 사용자/권한 seed (참고용)
-- 샘플에서는 build_features.py가 face_user를 자동 시드한다:
--   고객 = 생성된 U#### (role=customer), 관리자 = 'admin' (role=admin)
-- 수동 추가 예시:
INSERT OR IGNORE INTO face_user(user_id, display_name, role) VALUES
  ('admin', '관리자', 'admin'),
  ('demo_customer', '데모 고객', 'customer');

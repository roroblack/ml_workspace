-- 자주 쓰는 조회 (대시보드/검증용)

-- 유저별 최신 예측
SELECT p.* FROM prediction_log p
JOIN (SELECT user_id, MAX(prediction_id) mid FROM prediction_log GROUP BY user_id) t
  ON p.prediction_id = t.mid;

-- 위험등급 분포
SELECT risk_level, COUNT(*) FROM (
  SELECT user_id, risk_level FROM prediction_log p
  WHERE prediction_id=(SELECT MAX(prediction_id) FROM prediction_log WHERE user_id=p.user_id)
) GROUP BY risk_level;

-- 이벤트 퍼널
SELECT event_type, COUNT(*) FROM realtime_event_log GROUP BY event_type;

-- 활성 모델
SELECT * FROM model_registry WHERE is_active=1;

-- 고위험 Top 20
SELECT user_id, churn_probability, risk_level, recommended_action
FROM prediction_log ORDER BY churn_probability DESC LIMIT 20;

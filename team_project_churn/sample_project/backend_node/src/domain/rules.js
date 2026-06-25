'use strict';
// domain 계층 — 순수 규칙(위험등급·리텐션·앙상블). DB/HTTP 의존 없음.
const { RISK_HIGH, RISK_LOW } = require('../config');

const riskLevel = (p) => (p >= RISK_HIGH ? 'high' : p >= RISK_LOW ? 'medium' : 'low');

const retentionAction = (p) => {
  const r = riskLevel(p);
  if (r === 'high') return { action_type: 'coupon', action_message: '쿠폰 발송 + 재방문 알림(고위험)' };
  if (r === 'medium') return { action_type: 'remind', action_message: '장바구니 리마인드/맞춤추천(중위험)' };
  return { action_type: 'none', action_message: '정상 유지' };
};

const ensemble = (members) => {
  const tw = members.reduce((s, m) => s + (m.weight || 1), 0) || 1;
  const prob = members.reduce((s, m) => s + m.prob * (m.weight || 1), 0) / tw;
  const spread = Math.max(...members.map((m) => m.prob)) - Math.min(...members.map((m) => m.prob));
  return { prob_ensemble: prob, risk_level: riskLevel(prob),
    improvement: spread > 0.2 ? '모델 간 편차 큼 → 보정·임계값 재조정·약한모델 가중↓' : '모델 합의 양호', members };
};

module.exports = { riskLevel, retentionAction, ensemble };

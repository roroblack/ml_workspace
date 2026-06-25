// 운영 백엔드 서버 (Node.js, 의존성 0 — 내장 http). 추론 코드 없음: 모델파트가 파일로 학습/예측 → 결과 변수만 제출.
// 역할: 모델 결과 저장(registry) · 실시간 예측 조합/앙상블 · 추천/리텐션 · Streamlit에 자료 제공.
// 운영 권장: Fastify(JSON Schema 검증). 본 스켈레톤은 무설치 실행을 위해 내장 http 사용.
'use strict';
const http = require('http');
const db = require('./db');
try { require('dotenv').config(); } catch {}

const API_KEY = process.env.API_KEY || 'dev-key';
const PORT = process.env.PORT || 8080;
const RISK = { low: 0.35, high: 0.65 };
const riskLevel = p => (p >= RISK.high ? 'high' : p >= RISK.low ? 'medium' : 'low');

// ── 공유 변수(계약): 모델파트가 /models/submit 으로 보내는 형태 검증 ──
function validateSubmit(b) {
  const need = ['model_name', 'model_type', 'artifact_path'];
  for (const k of need) if (!b?.[k]) return `필수 누락: ${k}`;
  if (!['tree', 'linear', 'sequence', 'ensemble'].includes(b.model_type)) return 'model_type 허용값 아님';
  if (b.predictions && !Array.isArray(b.predictions)) return 'predictions 는 배열';
  return null;
}

// 리텐션 액션(risk별)
function retentionAction(prob) {
  const r = riskLevel(prob);
  if (r === 'high') return { action_type: 'coupon', action_message: '쿠폰 발송 + 재방문 알림(고위험)' };
  if (r === 'medium') return { action_type: 'remind', action_message: '장바구니 리마인드/맞춤추천(중위험)' };
  return { action_type: 'none', action_message: '정상 유지' };
}
// 추천 가중(view1/cart3/purchase5)
function recommend(interest) {
  const w = { view: 1, cart: 3, purchase: 5 };
  const score = {};
  for (const e of interest?.events || []) score[e.category_id] = (score[e.category_id] || 0) + (w[e.event_type] || 0);
  const top = Object.entries(score).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([c]) => c);
  return { rec_categories: top };
}
// 앙상블(가중평균) + 개선점
function ensemble(members) {
  const tw = members.reduce((s, m) => s + (m.weight || 1), 0) || 1;
  const prob = members.reduce((s, m) => s + (m.prob * (m.weight || 1)), 0) / tw;
  const spread = Math.max(...members.map(m => m.prob)) - Math.min(...members.map(m => m.prob));
  const improvement = spread > 0.2
    ? '모델 간 편차 큼 → calibration/임계값 재조정·약한 모델 가중 하향 권고'
    : '모델 합의 양호';
  return { prob_ensemble: prob, risk_level: riskLevel(prob), improvement, members };
}

// ── 라우팅 ──
const routes = {
  'GET /health': async () => ({ ok: true, mode: db.isLive() ? 'mysql' : 'skeleton(memory)' }),

  // 모델파트 결과 제출(공유 변수 계약)
  'POST /models/submit': async (b) => {
    const err = validateSubmit(b); if (err) return { _status: 400, error: err };
    const reg = await db.registerModel(b);
    let logged = 0;
    for (const p of b.predictions || []) {
      await db.logPrediction({ model_id: reg.model_id, ...p, recommended_action: retentionAction(p.churn_probability).action_message });
      logged++;
    }
    return { registered: reg, predictions_logged: logged };
  },
  'GET /models': async () => ({ models: await db.listModels() }),

  // 실시간 예측: 모델파트가 보낸 배치예측을 조합해 응답(백엔드는 추론 X)
  'POST /predict': async (b) => {
    const prob = b.churn_probability; // 모델파트 산출값
    return { user_id: b.user_id, churn_probability: prob, risk_level: riskLevel(prob),
      recommended_action: retentionAction(prob).action_message };
  },
  // Neon 시뮬 로그 pull → (모델파트 예측 결과와 결합) 표시용
  'POST /predict/realtime': async (b) => {
    const events = await db.pullSimEvents(b.since, b.limit || 1000);
    return { pulled: events.length, note: '모델파트 예측 결과(/models/submit)와 user_id로 조인해 Streamlit에 표시' };
  },
  'POST /recommend': async (b) => recommend(b.interest),
  'POST /retention-action': async (b) => {
    const act = retentionAction(b.churn_probability);
    if (b.push_to_neon) await db.pushRetention({ user_id: b.user_id, ...act });
    return { user_id: b.user_id, ...act, pushed: !!b.push_to_neon };
  },
  'POST /ensemble': async (b) => ensemble(b.members || []),
};

function send(res, code, obj) { res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8' }); res.end(JSON.stringify(obj)); }

const server = http.createServer((req, res) => {
  const url = new URL(req.url, 'http://x');
  const key = `${req.method} ${url.pathname}`;
  const handler = routes[key];
  if (!handler) return send(res, 404, { error: 'not found', route: key });
  if (url.pathname !== '/health' && (req.headers['x-api-key'] || '') !== API_KEY)
    return send(res, 401, { error: 'invalid api key' });
  let body = '';
  req.on('data', c => (body += c));
  req.on('end', async () => {
    try {
      const b = body ? JSON.parse(body) : {};
      const out = await handler(b);
      send(res, out?._status || 200, out);
    } catch (e) { send(res, 500, { error: String(e.message || e) }); }
  });
});

if (require.main === module) server.listen(PORT, () => console.log(`[backend] http://localhost:${PORT} (mode=${db.isLive() ? 'mysql' : 'skeleton'})`));
module.exports = { server, routes, riskLevel, ensemble, recommend, retentionAction, validateSubmit };

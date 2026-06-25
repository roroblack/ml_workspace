// 운영 DB(MySQL) + 시뮬 로그 DB(Neon Postgres) 연결.
// 의존성(mysql2/pg)이 없거나 env 미설정이면 '스켈레톤 모드'(메모리)로 동작 — npm install 없이도 서버가 선다.
'use strict';
let mysqlPool = null, pgPool = null;
const mem = { models: [], predictions: [], ensembles: [] }; // 스켈레톤 모드 저장소

function tryRequire(name) { try { return require(name); } catch { return null; } }

function initMySQL() {
  if (mysqlPool || !process.env.MYSQL_URL) return mysqlPool;
  const mysql = tryRequire('mysql2/promise');
  if (!mysql) { console.warn('[db] mysql2 미설치 → 스켈레톤 모드'); return null; }
  mysqlPool = mysql.createPool(process.env.MYSQL_URL);
  return mysqlPool;
}
function initNeon() {
  if (pgPool || !process.env.NEON_URL) return pgPool;
  const pg = tryRequire('pg');
  if (!pg) { console.warn('[db] pg 미설치 → Neon pull 비활성'); return null; }
  pgPool = new pg.Pool({ connectionString: process.env.NEON_URL, ssl: { rejectUnauthorized: false } });
  return pgPool;
}

const isLive = () => !!initMySQL();

// 모델 레지스트리 등록(모델파트 제출 저장)
async function registerModel(m) {
  if (isLive()) {
    const [r] = await mysqlPool.execute(
      `INSERT INTO model_registry(model_name,model_type,feature_schema_version,preprocessing_config,dataset_path,artifact_path,train_period,metric_cv,metric_oot,metrics_json,is_active)
       VALUES(?,?,?,?,?,?,?,?,?,?,?)
       ON DUPLICATE KEY UPDATE model_type=VALUES(model_type),preprocessing_config=VALUES(preprocessing_config),
         dataset_path=VALUES(dataset_path),artifact_path=VALUES(artifact_path),metric_cv=VALUES(metric_cv),
         metric_oot=VALUES(metric_oot),metrics_json=VALUES(metrics_json),is_active=VALUES(is_active)`,
      [m.model_name, m.model_type, m.feature_schema_version || 'v2',
       JSON.stringify(m.preprocessing_config || {}), m.dataset_path || null, m.artifact_path,
       m.train_period || null, m.metrics?.cv_pr_auc ?? null, m.metrics?.oot_pr_auc ?? null,
       JSON.stringify(m.metrics || {}), m.set_active ? 1 : 0]);
    if (m.set_active) await mysqlPool.execute(
      'UPDATE model_registry SET is_active=0 WHERE model_type=? AND model_name<>?', [m.model_type, m.model_name]);
    return { model_id: r.insertId, mode: 'mysql' };
  }
  const id = mem.models.length + 1;
  mem.models = mem.models.filter(x => !(m.set_active && x.model_type === m.model_type)).map(x => ({ ...x, is_active: 0 }));
  mem.models.push({ model_id: id, ...m, is_active: m.set_active ? 1 : 0 });
  return { model_id: id, mode: 'memory' };
}

async function listModels() {
  if (isLive()) { const [rows] = await mysqlPool.query('SELECT * FROM model_registry ORDER BY model_id DESC'); return rows; }
  return mem.models;
}
async function activeModels() {
  if (isLive()) { const [rows] = await mysqlPool.query('SELECT * FROM model_registry WHERE is_active=1'); return rows; }
  return mem.models.filter(m => m.is_active);
}
async function logPrediction(p) {
  if (isLive()) { await mysqlPool.execute(
    'INSERT INTO prediction_log(model_id,user_id,session_id,churn_probability,risk_level,top_factors_json,recommended_action) VALUES(?,?,?,?,?,?,?)',
    [p.model_id ?? null, p.user_id, p.session_id ?? null, p.churn_probability, p.risk_level, JSON.stringify(p.top_factors || {}), p.recommended_action ?? null]); return; }
  mem.predictions.push(p);
}
// Neon: 시뮬 로그 pull (최근 이벤트), 결과 push(쿠폰/추천)
async function pullSimEvents(sinceTs, limit = 1000) {
  const pool = initNeon(); if (!pool) return [];
  const { rows } = await pool.query(
    'SELECT user_id,event_type,product_id,category_id,price,event_time FROM sim_event_log WHERE event_time > $1 ORDER BY event_time LIMIT $2',
    [sinceTs || '1970-01-01', limit]);
  return rows;
}
async function pushRetention(neonAction) {
  const pool = initNeon(); if (!pool) return false;
  await pool.query('INSERT INTO retention_push(user_id,action_type,payload_json) VALUES($1,$2,$3)',
    [neonAction.user_id, neonAction.action_type, JSON.stringify(neonAction.payload || {})]);
  return true;
}

module.exports = { isLive, registerModel, listModels, activeModels, logPrediction, pullSimEvents, pushRetention, _mem: mem };

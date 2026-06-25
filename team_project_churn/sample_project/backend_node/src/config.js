'use strict';
// env 로딩·검증 (19-1: config 계층). .env 있으면 읽고, MySQL은 옵션(미설정 시 데모 모드).
try { require('dotenv').config(); } catch {}
const path = require('path');
const ROOT = path.resolve(__dirname, '..', '..');          // 가지마 루트
module.exports = {
  PORT: process.env.PORT || 8090,
  API_KEY: process.env.API_KEY || 'dev-key',
  DATA_DIR: path.join(ROOT, 'data', 'processed'),
  EVAL_DIR: path.join(ROOT, 'data', 'processed', 'evaluation'),
  REC_DIR: path.join(ROOT, 'data', 'processed', 'recommendation'),
  mysql: process.env.MYSQL_HOST ? {
    host: process.env.MYSQL_HOST, port: +(process.env.MYSQL_PORT || 3306),
    user: process.env.MYSQL_USER || 'root', password: process.env.MYSQL_PASSWORD || '',
    database: process.env.MYSQL_DATABASE || 'gajima',
  } : null,
  neon: process.env.NEON_URL || null,    // 시뮬 로그 전용(외부)
  RISK_HIGH: 0.65, RISK_LOW: 0.35,
};

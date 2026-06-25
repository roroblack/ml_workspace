'use strict';
// infrastructure/files — eval 산출물(차트 원천) 파일 reader. 백엔드는 재학습 안 함.
const fs = require('fs');
const path = require('path');
const { EVAL_DIR, REC_DIR } = require('../../config');

const _json = (p) => (fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, 'utf-8')) : null);

module.exports = {
  metrics: () => _json(path.join(EVAL_DIR, 'metrics_summary.json')) || {},
  curves: () => _json(path.join(EVAL_DIR, 'curves.json')) || {},
  shap: () => _json(path.join(EVAL_DIR, 'shap_summary.json')) || {},
  // 모델별 차트 JSON (PR/ROC/threshold/calibration/shap)
  chart: (model, name) => {
    const c = _json(path.join(EVAL_DIR, 'curves.json')) || {};
    if (name === 'shap') return (_json(path.join(EVAL_DIR, 'shap_summary.json')) || {})[model] || null;
    return (c[model] || {})[name] || null;
  },
  evalExists: () => fs.existsSync(path.join(EVAL_DIR, 'metrics_summary.json')),
};

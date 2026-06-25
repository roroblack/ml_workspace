'use strict';
// validators — 제출 payload 검증(경량). 19-1 modelSubmit/evaluation 스키마.
function validateModelSubmit(b) {
  if (!b || typeof b !== 'object') return 'body 없음';
  for (const k of ['model_name', 'model_type', 'artifact_path']) if (!b[k]) return `필수 누락: ${k}`;
  if (!['tree', 'linear', 'sequence', 'ensemble'].includes(b.model_type)) return 'model_type 허용값 아님';
  if (b.metrics && typeof b.metrics !== 'object') return 'metrics 는 객체';
  if (b.evaluation && typeof b.evaluation !== 'object') return 'evaluation 은 객체';
  return null;
}
module.exports = { validateModelSubmit };

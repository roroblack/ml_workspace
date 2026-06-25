'use strict';
// interfaces/http — 라우트 → usecase 바인딩(얇게). SQL/파일 직접 접근 안 함.
const uc = require('../../application/usecases');
const art = require('../../infrastructure/files/artifactStore');

// key: "METHOD /path"  (·:param 은 server에서 매칭)
module.exports = {
  'GET /health': async () => ({ ok: true, eval: art.evalExists() }),
  'POST /models/submit': async (b) => uc.submitModel(b),
  'GET /models': async () => uc.listModels(),
  'GET /dashboard/summary': async () => uc.getDashboardSummary(),
  // /models/:model/charts/:name
  'GET /models/:model/charts/:name': async (_b, p) => uc.getModelCharts(p.model, p.name),
  'POST /predict': async (b) => uc.predictFromScore(b.user_id, b.churn_probability, b.model_id),
  'POST /ensemble/run': async (b) => uc.runEnsemble(b.members),
};

'use strict';
// application 계층 — usecase 오케스트레이션(도메인 규칙 + repository/artifact 포트 조합). SQL/HTTP 직접 안 함.
const { riskLevel, retentionAction, ensemble } = require('../domain/rules');
const { modelRepository, evaluationRepository, predictionRepository, mode } = require('../infrastructure/mysql/pool');
const art = require('../infrastructure/files/artifactStore');
const { validateModelSubmit } = require('../validators/schemas');

async function submitModel(body) {                                  // 9.1
  const err = validateModelSubmit(body);
  if (err) return { _status: 400, error: err };
  const reg = await modelRepository.upsert(body);
  let eval_id = null;
  if (body.evaluation || body.metrics) {
    const m = body.metrics || {};
    ({ eval_id } = await evaluationRepository.insert(reg.model_id, {
      dataset_tag: body.label_name || 'churn', roc_auc: m.roc_auc, pr_auc: m.pr_auc,
      best_threshold: m.best_threshold, best_f1: m.best_f1,
      eval_predictions_path: body.evaluation?.eval_predictions_path,
      shap_summary_path: body.evaluation?.shap_summary_path }));
  }
  return { model_id: reg.model_id, eval_id, mode: reg.mode };
}

async function listModels() { return { models: (await modelRepository.list()) || [], mode: mode() }; }

function getDashboardSummary() {                                    // 9.x 관리자 요약(파일 원천)
  const m = art.metrics();
  const tab = Object.entries(m).filter(([, v]) => v && v.auc != null)
    .map(([k, v]) => ({ model: k, roc_auc: v.auc, pr_auc: v.pr_auc, brier: v.brier, ece: v.ece, f1: v.f1, threshold: v.threshold }))
    .sort((a, b) => b.roc_auc - a.roc_auc);
  const best = tab[0] || null;
  return { best_model: best?.model, best_auc: best?.roc_auc, models: tab, label: 'churn', horizon_days: 7 };
}

function getModelCharts(model, name) {                              // 9.3 차트 원천=학습 산출물
  const data = art.chart(model, name);
  return data ? { model, chart: name, data } : { _status: 404, error: `chart 없음: ${model}/${name}` };
}

function predictFromScore(user_id, churn_probability, model_id) {   // 9.2 (점수는 모델파트/사이드카가 제공)
  const r = riskLevel(churn_probability);
  const act = retentionAction(churn_probability);
  predictionRepository.log({ model_id, user_id, churn_probability, risk_level: r, recommended_action: act.action_message });
  return { user_id, churn_probability, risk_level: r, recommended_action: act.action_message, horizon_days: 7 };
}

const runEnsemble = (members) => ensemble(members || []);

module.exports = { submitModel, listModels, getDashboardSummary, getModelCharts, predictFromScore, runEnsemble };

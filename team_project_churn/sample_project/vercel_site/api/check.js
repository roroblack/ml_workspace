// 무설치 스모크: 모듈 로드 + 핵심 로직 단언 (DB/npm 없이 실행).
'use strict';
const assert = require('assert');
const { riskLevel, ensemble, recommend, retentionAction, validateSubmit } = require('./server');

assert.strictEqual(riskLevel(0.9), 'high');
assert.strictEqual(riskLevel(0.5), 'medium');
assert.strictEqual(riskLevel(0.1), 'low');

assert.strictEqual(validateSubmit({ model_name: 'm', model_type: 'tree', artifact_path: 'a' }), null);
assert.ok(validateSubmit({ model_name: 'm' }));               // 누락 감지
assert.ok(validateSubmit({ model_name: 'm', model_type: 'x', artifact_path: 'a' })); // 잘못된 type

const e = ensemble([{ prob: 0.8, weight: 2 }, { prob: 0.4, weight: 1 }]);
assert.ok(Math.abs(e.prob_ensemble - (0.8 * 2 + 0.4) / 3) < 1e-9);
assert.strictEqual(e.risk_level, 'high');

const r = recommend({ events: [{ category_id: 'c1', event_type: 'purchase' }, { category_id: 'c2', event_type: 'view' }] });
assert.deepStrictEqual(r.rec_categories[0], 'c1');

assert.strictEqual(retentionAction(0.9).action_type, 'coupon');

console.log('[check] 모든 단언 통과 ✅ (무설치 스켈레톤 정상)');

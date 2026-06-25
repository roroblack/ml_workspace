# BUG-004 — 대시보드 모델 진단 탭 크래시 (`/models/active` 응답 언랩 누락)

| 항목 | 값 |
| --- | --- |
| **ID** | BUG-004 |
| **제목** | 02_dashboard.py 모델 진단 탭에서 `'str' object has no attribute 'get'` |
| **심각도** | High (모델 진단 탭 렌더 전체 크래시) |
| **상태** | ✅ Resolved (2026-06-22) |
| **영역** | `dashboard_streamlit/pages/02_dashboard.py` (프론트) — 백엔드는 정상 |
| **보고일** | 2026-06-22 (사용자 트레이스백 제보) |
| **관련 문서** | [19-7-1](19-7-1_대시보드_IO계약요소_리포트.md) §5.1, [BUG-001](BUG-001_얼굴인증_계약불일치.md)(형식) |

## 1. 요약
백엔드 `GET /models/active`는 `{ok, data:{"models":[{model_id, model_name,...}]}}`(봉투 안 dict)를 주는데, 대시보드가 `data`(dict)를 바로 순회해 dict **키 문자열**(`"models"`)에 `.get()`을 호출 → 크래시. `data["models"]`를 안 깐 프론트 버그.

## 2. 증상
```
AttributeError: 'str' object has no attribute 'get'
02_dashboard.py:197  model_options = [m.get("model_id") for m in models_resp["data"]]
```

## 3. 발생 위치 (Where)
- `dashboard_streamlit/pages/02_dashboard.py` 모델 진단 탭, `model_options` 생성부.

## 4. 근본 원인 (Why)
- 백엔드 응답 `data = {"models": [...]}` 인데 `for m in models_resp["data"]` 가 dict를 순회 → `m = "models"`(str) → `m.get(...)` 실패.
- 추가로 원 코드가 `model_id`(정수 14)를 쓰려 했는데, 차트 엔드포인트 `/models/{model}/charts`는 **model_name/key로 해석**(`eval_artifacts._key`)하므로 숫자 id로는 차트도 안 됐을 것.

## 5. 해결 (2026-06-22) — 프론트 최소수정
```python
data = models_resp["data"]
rows = data.get("models", []) if isinstance(data, dict) else data   # {"models":[...]} 언랩
for m in rows:
    if isinstance(m, dict):
        model_options.append(m.get("model_name") or m.get("model_id"))  # id 아닌 name
    elif isinstance(m, str):
        model_options.append(m)
if not model_options:
    model_options = ["CatBoost_Churn_v2", "XGBoost_Baseline"]
```
- ① `data["models"]` 언랩(크래시 해결) ② `model_name` 사용(차트 호출 정상) ③ dict/list/str 모두 방어.

## 6. 검증
| 항목 | 결과 |
| --- | --- |
| 백엔드 `/models/active` 응답 | `data={"models":[{model_id:14, model_name:"LightGBM_Churn_v2",...}]}` (계약대로 정상) |
| 수정 후 model_options | `["LightGBM_Churn_v2", ...]`(이름) — 셀렉트박스 표시·차트 호출 정상 |
| 차트 엔드포인트 model 해석 | `eval_artifacts._key`가 이름/키/등록명 모두 허용 → name으로 매칭 |

## 7. 재발 방지
- 봉투(`{ok,data}`) **내부 컨테이너(`data.models`)를 항상 명시적으로 언랩**. 리스트로 가정 금지.
- 프론트 서비스가 받는 응답 shape를 19-7-1 계약(§5.1)과 대조하는 체크리스트.
- 백엔드는 계약대로 정상이었음 — 프론트 단독 버그.

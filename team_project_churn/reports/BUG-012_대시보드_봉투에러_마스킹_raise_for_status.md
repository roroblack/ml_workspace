# BUG-012 — 대시보드가 백엔드 비즈니스 에러를 "API 호출 실패"로 가림 (raise_for_status 마스킹)

- 작성일: 2026-06-23
- 심각도: Medium (기능은 정상이나 **에러 원인 오인** 유발 — 디버깅·UX 저해)
- 상태: **FIXED**
- 영역: `dashboard_streamlit/services/api_client.py`

## 증상
- 얼굴 로그인 시 대시보드: **"백엔드 API 호출 실패: 404 Client Error: Not Found for url .../auth/face/login"**.
- 마치 서버 다운/라우트 없음처럼 보이나, 실제 백엔드 응답은 정상 봉투:
  `{"ok":false, "error":{"code":"HTTP_404","message":"등록되지 않은 사용자 ID: sltko"}}` (HTTP 404).

## 원인
1. **직접 원인**: 입력 ID `sltko`가 **미등록**(현 등록 8명: admin·test1·real_face_1 …). 앞선 등록이 422(얼굴 임베딩 실패)로 **저장되지 않았음** → 로그인 404는 정상 동작.
2. **마스킹 버그**: `request_json()`이 `response.raise_for_status()`를 **본문 파싱 전에** 호출 → 4xx면 즉시 `HTTPError` → `except`에서 `"백엔드 API 호출 실패: {exc}"` 로 치환. 그 결과 백엔드의 친절한 에러 메시지(`error.message`)가 **버려짐**.
   - 영향: login 404(미등록), register 422(얼굴미검출)·409(중복) 등 **모든 비즈니스 4xx가 동일하게 "API 호출 실패"로 가려짐** → 사용자가 원인 구분 불가.

## 수정 (`api_client.request_json`)
봉투(`ok`/`error`) 응답은 **4xx여도 그대로 반환**하도록 순서 변경:
```python
try:
    payload = response.json()
except ValueError:
    payload = None
if isinstance(payload, dict) and "ok" in payload:
    return payload                # 비즈니스 메시지 보존(2xx/4xx 무관)
response.raise_for_status()       # 봉투 아닌 응답이 4xx/5xx면 진짜 실패
```
- 결과: 로그인 실패 시 대시보드가 **"등록되지 않은 사용자 ID: sltko"** 를 그대로 표시 → 사용자가 "등록 먼저" 라고 인지.

## 검증
- `auth_service.login_face("sltko", ...)` → `ok:False`, **message="등록되지 않은 사용자 ID: sltko"** (기존엔 "백엔드 API 호출 실패: 404").
- 컴파일 통과. (대시보드는 파일 변경 후 새로고침/재실행 시 반영)

## 사용자 조치
- `sltko`로 로그인하려면 **먼저 얼굴 등록을 성공**시켜야 함. 현 머신은 `/health.face_ready:true`(insightface 정상)라 **실제 얼굴이 보이게 촬영**하면 등록됨. 등록 성공 후 동일 ID로 로그인.

## 사례 — test4 "가입했는데 로그인 안 됨"
- 증거: `face_login_log`에 test4 로그인 시도 다수 전부 `user_not_found`, `face_user`에 test4 없음. (반면 test1·test3는 로그인 성공 → 파이프라인 정상)
- 해석: **test4 등록이 얼굴 미검출(422)로 미저장**됐는데, 마스킹 때문에 사용자가 등록 실패를 인지 못 함. 본 수정 후엔 등록 화면에 **"얼굴 임베딩 실패(얼굴 미검출 또는 insightface 모델 미준비)"** 가 그대로 떠 재촬영을 유도.
- 검증: no-face 이미지로 `register_face("test4",...)` → `ok:False`, message="얼굴 임베딩 실패(…)" (기존엔 "API 호출 실패: 422"로 가려짐).

## 파일
- `dashboard_streamlit/services/api_client.py` (봉투 4xx 보존)

# BUG-009 — 배포 PC에서 얼굴 등록/로그인 422 (insightface 모델 미준비)

- 작성일: 2026-06-23
- 심각도: **High** (배포본에서 얼굴 인증 기능 전면 불가)
- 상태: **FIXED**(코드) + 배포 절차 보완 필요
- 영역: `backend/app/infrastructure/face/embedder.py` · 배포 패키징

## 증상
- **배포본을 받은 다른 PC**에서 `POST /auth/face/register`(및 `/login`) → **422**.
  ```
  POST /auth/face/register 422 Unprocessable Content
  대시보드: "백엔드 API 호출 실패: 422 ... /auth/face/register"
  ```
- **개발 머신에선 정상** → 환경 차이 = 이식성 문제.

## 원인 (root cause)
422 본문은 요청검증 오류가 아니라 **비즈니스 오류**: `auth_usecase` → "얼굴 임베딩 실패(얼굴 미검출 또는 insightface 모델 미준비)". 즉 `embedder.embed_from_image_bytes()`가 **None**을 반환.

왜 배포 PC만 None인가:
1. `embedder._get_app()`이 `FaceAnalysis(name="buffalo_l")`를 **`root=` 없이** 호출 → insightface가 **기본 경로 `~/.insightface/models/buffalo_l`** 를 찾고, 없으면 **런타임 자동 다운로드**(~280MB).
2. 개발 머신엔 예전에 자동 다운로드된 캐시(`~/.insightface/...`, 06-18자)가 있어 동작. **배포 PC엔 캐시 없음** + 오프라인/방화벽/최초실행이면 **다운로드 실패**.
3. 실패를 `except Exception: _APP=None` 으로 **조용히 삼켜** → 원인 로그도 없이 전부 None → 모든 얼굴 인증 422.
4. 프로젝트에 번들(`models/buffalo_l/*.onnx` + `buffalo_l.zip`)이 있으나 **embedder가 가리키지 않았고**, `models/buffalo_l` 는 **`.gitignore` 대상**이라 git 기반 배포 시 **누락**될 수 있었음.

## 수정 (`embedder.py`)
1. **번들 우선 사용**: `models/buffalo_l` 을 insightface `root`로 지정 → `FaceAnalysis(name="buffalo_l", root=<GAJIMA_ROOT>, providers=["CPUExecutionProvider"])`. insightface가 `<root>/models/buffalo_l/*.onnx` 를 **다운로드 없이** 로드(오프라인 OK). 번들 없으면 기존 기본경로(다운로드)로 폴백.
2. **zip 자동 추출**: `models/buffalo_l` 에 onnx 없고 `buffalo_l.zip` 있으면 자동 추출(배포 시 zip만 동봉해도 동작).
3. **CPU provider 명시**: 배포 PC GPU/CUDA 부재 대비 `CPUExecutionProvider`.
4. **에러 노출(silent 금지)**: 실패 사유를 `print`+`traceback`+`embedder.last_error()` 로 보존.
5. **`/health` 진단 필드**: `face_ready`(bool)·`face_error`(사유) 추가 → 배포 PC에서 즉시 원인 확인.

## 배포 절차 보완 (필수)
- `models/buffalo_l` 는 gitignore라 **git push로는 안 따라감**. 배포 패키지에 **`models/buffalo_l/`(또는 `models/buffalo_l.zip`)을 반드시 포함**.
- 대상 PC에서 `pip install -r backend/requirements.txt` (insightface 0.7.3 · onnxruntime 1.20.1 · opencv-python 4.10) 설치.
- 첫 기동 후 `GET /health` 의 **`face_ready: true`** 확인. false면 `face_error` 로 원인 파악(모델 누락/onnxruntime 미설치 등).

## 검증
- insightface root 해석: `<GAJIMA_ROOT>/models/buffalo_l` 존재 확인 → 번들 직사용(다운로드 불필요).
- zip 구조 확인: 루트에 onnx 5개(det_10g·w600k_r50 등) → `models/buffalo_l/` 로 추출.
- `py_compile` 통과. (개발 머신은 번들·캐시 모두 보유 → 회귀 없음. 적용엔 백엔드 재기동 권장 → `/health.face_ready` 확인)

## 파일
- `backend/app/infrastructure/face/embedder.py` (번들 root·zip 추출·CPU provider·에러 노출)
- `backend/app/interfaces/http/health_router.py` (`face_ready`·`face_error`)

## 후속(선택)
- 422 응답 메시지에 `embedder.last_error()` 동봉(대시보드에서 "모델 미준비 vs 얼굴 미검출" 구분 표시).
- 배포 산출물 빌드 스크립트에 `models/buffalo_l` 포함 체크 추가.

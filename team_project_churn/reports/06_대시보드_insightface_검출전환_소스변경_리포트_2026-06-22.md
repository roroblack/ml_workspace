# 대시보드 얼굴검출 insightface 전환 — 소스 변경 리포트

- 작성일: 2026-06-22
- 대상 저장소: `SKN32-2nd_GAJIMA_Dev` (branch: `feature/backend`)
- 작업: 대시보드 얼굴 로그인 검출기 **OpenCV Haar Cascade → insightface(buffalo_l)** 전환
- 근거: 실제 `git diff` 검증 결과(추정 아님)

---

## 1. 배경
- 동원판 대시보드의 얼굴 검출은 `face_utils.py`에서 **Haar Cascade**(`haarcascade_frontalface_default.xml`, `minSize=(80,80)`, `minNeighbors=5`) 사용.
- Haar는 조명·각도·거리에 취약 → "얼굴을 찾지 못했습니다. 밝은 곳에서…" 메시지가 빈발.
- 환경: insightface 0.7.3 / onnxruntime 1.20.1 설치됨, `buffalo_l` 모델 캐시 완료(`~/.insightface/models`).

## 2. 작업 중 발견된 방향 전환 (중요)
작업 도중, **백엔드가 동시에 재작성**되어 있었음을 확인:
- 새 구조 = "**프론트는 검출만, 얼굴 이미지를 multipart로 전송 → 백엔드가 insightface 임베딩 수행**" (19-4 §1).
- 관련(=내가 작성하지 않은, 동시 작업으로 들어온) 백엔드 파일:
  - `backend/app/interfaces/http/auth_router.py` — `/auth/face/register|login`을 `UploadFile`(multipart)로 수신, `/auth/face/check-id`(GET) 추가.
  - `backend/app/application/auth_usecase.py` — `check_id()`, `register_face_image()`, `login_face_image()` 추가.
  - `backend/app/infrastructure/face/embedder.py` — 신규. 이미지 bytes → buffalo_l/ArcFace 512d 정규화 임베딩.
- 따라서 초기에 시도했던 "프론트 임베딩 추출 → JSON 전송" 안은 **방향이 반대라 충돌** → 되돌리고 프론트를 백엔드 계약(multipart)에 정렬.

---

## 3. 실제 변경된 소스 — `dashboard_streamlit/services/face_utils.py` (1개 파일)

> **순(net) 소스 변경은 이 파일 하나뿐.** Haar → insightface **검출 전용**(임베딩은 백엔드 담당이므로 recognition 모듈 미로드).

### 3.1 변경 요약
| 위치 | 변경 내용 |
|---|---|
| 파일 상단 docstring | insightface 검출 전용 설명 **추가** |
| import | `import threading` **추가** |
| `FaceDetectionResult` dataclass | 필드(`detected`/`bbox`/`preview_bytes`) **유지**, `bbox`에 `(x,y,w,h)` 주석만 추가 |
| 모듈 전역 | `_app=None`, `_lock` + **`_get_app()` 싱글톤 신규** |
| `_decode_image()` | 본문 동일, `# BGR(insightface 입력 규격)` 주석만 |
| `_fail()` | 미검출/디코드 실패 공통 처리 헬퍼 **신규** |
| `detect_largest_face()` 본문 | Haar 검출 블록 → insightface 검출 블록 **교체** |

### 3.2 신규 추가된 insightface 로더(싱글톤, lazy)
```python
_app = None
_lock = threading.Lock()

def _get_app():
    """FaceAnalysis 싱글톤(detection 만, CPU)."""
    global _app
    if _app is None:
        with _lock:
            if _app is None:
                from insightface.app import FaceAnalysis
                app = FaceAnalysis(
                    name="buffalo_l",
                    allowed_modules=["detection"],     # 검출만 (임베딩은 백엔드)
                    providers=["CPUExecutionProvider"],
                )
                app.prepare(ctx_id=0, det_size=(640, 640))
                _app = app
    return _app
```

### 3.3 `detect_largest_face()` 핵심 교체부
**제거된 Haar 검출:**
```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
if len(faces) == 0:
    ok, encoded = cv2.imencode(".jpg", image)
    return FaceDetectionResult(False, (0, 0, 0, 0), encoded.tobytes() if ok else image_bytes)
x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
preview = image.copy()
cv2.rectangle(preview, (x, y), (x + w, y + h), (118, 217, 87), 4)
```

**추가된 insightface 검출:**
```python
try:
    faces = _get_app().get(image)
except Exception:
    return _fail(image, image_bytes)
if not faces:
    return _fail(image, image_bytes)

f = max(faces, key=lambda fc: (fc.bbox[2] - fc.bbox[0]) * (fc.bbox[3] - fc.bbox[1]))
x1, y1, x2, y2 = (int(v) for v in f.bbox)         # insightface bbox = (x1,y1,x2,y2)
preview = image.copy()
cv2.rectangle(preview, (x1, y1), (x2, y2), (118, 217, 87), 4)
ok, encoded = cv2.imencode(".jpg", preview)
return FaceDetectionResult(
    True, (x1, y1, max(x2 - x1, 0), max(y2 - y1, 0)),  # (x,y,w,h) 인터페이스 유지
    encoded.tobytes() if ok else image_bytes,
)
```

> `FaceDetectionResult`의 외부 인터페이스(detected/bbox/preview_bytes)는 동일하게 유지 → 호출부(`01_face_login.py`, `auth_service.py`) 수정 불필요.

---

## 4. 수정했다가 **되돌린** 파일 (현재 순변경 0)
초기 "프론트 임베딩→JSON" 시도분을 `git checkout`으로 동원 원본(multipart)으로 복원. 검증: `git diff --quiet` 통과.
- `dashboard_streamlit/services/auth_service.py` — 변경 없음(multipart 전송 원본 유지)
- `dashboard_streamlit/pages/01_face_login.py` — 변경 없음(`image_bytes` + `face.bbox` 전송 원본 유지)

## 5. 본 작업에서 **건드리지 않은** 파일 (동시 작업분)
- `backend/app/application/auth_usecase.py`
- `backend/app/interfaces/http/auth_router.py`
- `backend/app/infrastructure/face/embedder.py`

## 6. 관련 보조 파일 (소스 아님, `.gitignore` 처리)
- `dashboard_streamlit/.env` — UI↔백엔드 라이브 연동(포트 8090, `anchor-dev-key`, `USE_MOCK=false`)
- `plans/feature-backend-faceauth-insightface_swap.md` — 작업 계획서

---

## 7. 검증 결과 (서버 재기동 후)
| 항목 | 결과 |
|---|---|
| 변경 파일 `py_compile` | OK |
| 백엔드 `/health` | OK |
| `GET /auth/face/check-id?user_id=` | 200 `{available:true}` |
| `POST /auth/face/register` (multipart, 노이즈 이미지) | 422 "얼굴 미검출" → 백엔드 insightface 동작 확인 |
| 프론트 `face_utils` 검출 모델 로드 | `det_10g.onnx`(detection)만 로드, 노이즈→`detected=False`, 무크래시 |
| Streamlit `_stcore/health` | OK |
| 실제 얼굴 등록/로그인 라운드트립 | **웹캠 사용자 검증 필요(미완)** |

## 8. 최종 구조
```
프론트(Streamlit)                         백엔드(FastAPI)
─────────────────                        ─────────────────
face_utils.detect_largest_face()         /auth/face/register (multipart)
  = insightface 검출(det only)   ──img──►   embedder.embed_from_image_bytes()
  + 미리보기/bbox                            = insightface 512d 임베딩
                                            → 저장/매칭/로깅
```
- 프론트: insightface 검출(견고) + 미리보기 + 이미지 전송
- 백엔드: insightface 임베딩 + 매칭 + 저장

## 9. 후속(미완) 항목
1. 실제 웹캠으로 등록→로그인 라운드트립 검증.
2. (선택) 프론트/백엔드가 각각 insightface 로드 → 메모리 중복. 프론트 검출 생략하고 백엔드 검증에만 의존하도록 단순화 가능.
3. 대시보드 본문(`02_dashboard.py`)은 현재 "API 연결 대기" 플레이스홀더 — 별도 연동 작업 필요.

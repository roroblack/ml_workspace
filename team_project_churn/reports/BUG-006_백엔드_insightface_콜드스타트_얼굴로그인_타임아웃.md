# BUG-006 — 백엔드 insightface 콜드스타트로 얼굴 로그인 Read timeout

| 항목 | 값 |
| --- | --- |
| **ID** | BUG-006 |
| **제목** | 첫 얼굴 등록/로그인 시 백엔드 insightface 모델 로드(6.8s)가 단일 워커를 막아 대시보드 10s Read timeout |
| **심각도** | Medium (첫 얼굴 요청 실패 — 재시도하면 동작, 단 사용자 경험 저해) |
| **상태** | ✅ FIXED (2026-06-23) |
| **영역** | backend `app/infrastructure/face/embedder.py`, `app/main.py` / dashboard `.env` |
| **보고일** | 2026-06-23 |
| **관련** | [BUG-001](BUG-001_얼굴인증_계약불일치.md)(백엔드 임베딩), [BUG-005 얼굴로그인](BUG-005_얼굴로그인_흐름점검.md) M1 |

---

## 1. 증상
대시보드 얼굴 로그인/등록 시:
```
백엔드 API 호출 실패: HTTPConnectionPool(host='127.0.0.1', port=8090): Read timed out. (read timeout=10.0)
```
연결은 되는데(서버 살아있음) 응답이 10초 넘게 안 옴.

## 2. 원인
- 전 GET 엔드포인트는 빠름(측정: health 0.01s, summary 0.04s, user 1.3s, charts <0.1s).
- 느린 것은 **얼굴 등록/로그인(POST multipart) → 백엔드 insightface 임베딩**.
- `embedder._get_app()`가 **첫 호출 때 buffalo_l 5개 모델(det/recognition/landmark_3d_68/landmark_2d_106/genderage) 전체 로드 = 6.8s**. 임베딩 계산 자체는 0.16s로 빠름.
- 백엔드 uvicorn이 **단일 워커**라 이 6.8s 로드가 진행되는 동안 모든 요청이 블록 → 대시보드 10s 타임아웃 초과.

## 3. 수정 (FIXED)
| 파일 | 변경 |
| --- | --- |
| `backend/app/main.py` | `lifespan`에 **insightface 백그라운드 사전로드(prewarm)** 추가 — 기동 시 `asyncio.to_thread(embedder._get_app)`로 미리 로드. 첫 사용자 요청이 cold-load를 안 겪음(핵심) |
| `backend/app/infrastructure/face/embedder.py` | `FaceAnalysis(name="buffalo_l", allowed_modules=["detection","recognition"])` — 임베딩에 불필요한 landmark_3d_68·landmark_2d_106·genderage 제외(로드·추론 단축) |
| `dashboard_streamlit/.env` | `DASHBOARD_TIMEOUT_SEC` 10 → 30 (여유) |

## 4. 검증
- 재기동(사전로드 포함) 후 `POST /auth/face/register`(노이즈 이미지) → **0.20s** HTTP 422(얼굴 미검출, 정상) = **warm 상태**. (이전 cold 6.8s → 타임아웃)
- 전 파일 `py_compile` 통과.

## 5. 비고
- 프론트(BUG-005 M1)는 insightface를 **제거**(OpenCV 검출만), 임베딩은 백엔드 일원화 상태. 따라서 무거운 insightface 로드는 **백엔드 1곳뿐** → 그 1곳을 사전로드로 워밍하면 첫 요청도 빠름.
- 단일 워커 한계 자체를 더 줄이려면 추후 `--workers N` 또는 임베딩 워커 분리 고려(현 데모 규모엔 prewarm으로 충분).

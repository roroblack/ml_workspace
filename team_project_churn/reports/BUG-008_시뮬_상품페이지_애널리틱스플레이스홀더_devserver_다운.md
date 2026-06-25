# BUG-008 — 시뮬 사이트 상품페이지 진입 시 dev 서버 다운(애널리틱스 플레이스홀더)

- 작성일: 2026-06-23
- 심각도: **High** (페이지 진입 시 사이트 전체 다운)
- 상태: **FIXED**
- 영역: `simulation_site` (Vite + Express dev 서버 / index.html)

## 증상
- (Admin 등) 상품페이지(`/products`, `/product/:id`) 진입 시 사이트가 "꺼짐".

## 원인 (root cause)
`simulation_site/client/index.html`에 umami 애널리틱스 정적 스크립트:
```html
<script defer src="%VITE_ANALYTICS_ENDPOINT%/umami" data-website-id="%VITE_ANALYTICS_WEBSITE_ID%"></script>
```
- `VITE_ANALYTICS_ENDPOINT` env가 **미설정** → Vite가 `%VITE_ANALYTICS_ENDPOINT%`를 치환 못 하고 **리터럴로 방치**.
- 브라우저가 `src="%VITE_ANALYTICS_ENDPOINT%/umami"`를 **상대경로**로 해석 → 현재 경로가 `/product/sku_10002`면 `/product/%VITE_ANALYTICS_ENDPOINT%/umami` 요청.
- Express dev 서버가 이 URL을 라우팅하며 `decodeURIComponent('%VI...')` → **URIError: Failed to decode param** 을 `setImmediate` 콜백에서 던짐 → **미처리 예외 → Node dev 서버 프로세스 종료** → 사이트 전체 다운.
- 홈(`/`)에선 상대경로가 `/%VITE_..%/umami`라도 비슷하게 터졌고, 특히 path 세그먼트가 있는 `/product/...`에서 재현이 쉬움.

서버 로그 증거:
```
Malformed URI sequence in request URL: /product/%VITE_ANALYTICS_ENDPOINT%/umami
URIError: Failed to decode param '/product/%VITE_ANALYTICS_ENDPOINT%/umami'
    at decodeURIComponent (<anonymous>) ... at process.processImmediate
```

## 수정
1. **정적 스크립트 제거** — `client/index.html`에서 미치환 플레이스홀더 스크립트 삭제(주석으로 사유 기록).
2. **안전 동적 주입** — `client/src/main.tsx`에서 `VITE_ANALYTICS_ENDPOINT`가 **실제 http(s) URL일 때만** umami 스크립트를 주입(`/^https?:\/\//` 검증). 미설정/플레이스홀더면 스킵 → 깨진 URL 자체가 생성 안 됨. (애널리틱스 기능은 env 설정 시 그대로 동작)
3. **방어 심화(서버)** — `server/_core/index.ts` 최상단 미들웨어: `decodeURIComponent(req.path)` 실패 시 **400 반환**(프로세스 다운 대신). 깨진 URL 한 건이 dev 서버 전체를 죽이지 못하게 함.

## 검증
- `/product/sku_10002`, `/products` → **200**, served HTML에 `%VITE` 플레이스홀더 **0**.
- 과거 다운 유발 URL `/product/%VITE_ANALYTICS_ENDPOINT%/umami` → **400**(가드), 직후 상품페이지 다시 **200**(서버 생존).
- 서버 로그에 신규 `URIError`/`Malformed`/`VITE_ANALYTICS` 경고 **없음**.
- `tsc --noEmit` 통과(client+server).

## 파일
- `simulation_site/client/index.html`
- `simulation_site/client/src/main.tsx`
- `simulation_site/server/_core/index.ts`

## 후속(선택)
- 운영에서 애널리틱스가 필요하면 `.env`에 `VITE_ANALYTICS_ENDPOINT`(http(s) 전체 URL), `VITE_ANALYTICS_WEBSITE_ID` 지정 → main.tsx가 자동 주입.

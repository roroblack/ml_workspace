# BUG-002 — 시뮬레이션 사이트 Home 렌더 크래시 (getLoginUrl Invalid URL · OAuth 미설정)

| 항목 | 값 |
| --- | --- |
| **ID** | BUG-002 |
| **제목** | 시뮬 사이트가 흰 화면/에러로 안 뜸 — `getLoginUrl`이 미설정 OAuth env로 `new URL()` throw |
| **심각도** | High (메인 Home 및 `useAuth` 사용 화면 전체 렌더 실패 → 사이트 사용 불가) |
| **상태** | ✅ Resolved (2026-06-22) |
| **영역** | `simulation_site/client/src/const.ts`(`getLoginUrl`) · 호출처 `useAuth`/`main`/`DashboardLayout` |
| **보고일** | 2026-06-22 (사용자 브라우저 콘솔 스택트레이스 제보) |
| **유입 시점** | simulation_site(Manus export) 교체 시([26-15](26-15_시뮬사이트_교체_및_FastAPI어댑터.md)) — Manus OAuth 템플릿이 우리 미사용 env에 의존 |
| **관련 문서** | [BUG-001](BUG-001_얼굴인증_계약불일치.md)(형식 기준), 26-15 |

---

## 1. 요약 (TL;DR)
시뮬 프론트가 `http://localhost:3000/`에서 **HTTP 200(HTML 셸)은 뜨지만 React가 렌더 중 크래시**. 원인은 Manus 템플릿의 `getLoginUrl()`이 미설정 `VITE_OAUTH_PORTAL_URL`로 `new URL("undefined/app-auth")`를 만들어 `TypeError: Invalid URL`을 던지고, 이 함수가 `useAuth`의 **기본 인자에서 매 렌더 호출**되기 때문. 우리 시뮬은 Manus OAuth를 쓰지 않으므로 **미설정 시 `"/"`로 폴백**하도록 가드를 추가해 해결.

## 2. 증상 (What)
- 브라우저 콘솔:
  ```
  TypeError: Invalid URL
    at getLoginUrl (http://localhost:3000/src/const.ts:7:15)
    at useAuth (http://localhost:3000/src/_core/hooks/useAuth.ts:6:61)
    at Home (http://localhost:3000/src/pages/Home.tsx:10:31)
  ```
- 결과: 메인 페이지 렌더 실패(에러 오버레이/흰 화면). `curl`로는 200이라 **서버는 정상으로 오인**되기 쉬움.

## 3. 발생 위치 (Where)
- `simulation_site/client/src/const.ts:13` — `const url = new URL(`${oauthPortalUrl}/app-auth`);` 에서 `oauthPortalUrl = import.meta.env.VITE_OAUTH_PORTAL_URL = undefined`.
- 호출처(모두 throw 전파):
  - `client/src/_core/hooks/useAuth.ts:12` — `redirectPath = getLoginUrl()` (**기본 인자 → 인증 안 써도 매 렌더 실행**)
  - `client/src/main.tsx:21`, `client/src/components/DashboardLayout.tsx:73` — 미인증 리다이렉트 시 호출

## 4. 근본 원인 (Why)
- Manus export 템플릿은 **Manus OAuth 포털 로그인 URL**을 런타임에 빌드한다(`VITE_OAUTH_PORTAL_URL`, `VITE_APP_ID`).
- 우리 시뮬은 Manus OAuth를 **사용하지 않으며** 해당 env를 설정하지 않는다 → `oauthPortalUrl`이 `undefined` → `new URL("undefined/app-auth")`는 절대 URL이 아니라 **throw**.
- 게다가 `useAuth`가 옵션 기본값으로 `getLoginUrl()`을 **무조건 평가**하므로, 로그인 기능을 안 써도 컴포넌트가 마운트되는 순간 크래시.

## 5. 영향 (Impact)
- `useAuth`를 쓰는 Home 등 메인 화면 렌더 불가 → 시뮬 사이트 사실상 사용 불가.
- 서버 응답(200)만 보면 정상으로 보여 **오진 위험**(본 세션에서 실제로 "동작함"으로 1차 오판 → 정정).

## 6. 진단 방법 (How detected)
1. 사용자 콘솔 스택트레이스로 진입점 특정: `getLoginUrl → useAuth → Home`.
2. `const.ts` 확인 → `new URL(`${oauthPortalUrl}/app-auth`)`, env 미주입으로 `undefined`.
3. `grep getLoginUrl` → `useAuth.ts:12` 기본 인자에서 매 렌더 호출 확인(인증 비사용에도 실행되는 이유).
4. `grep "new URL("` → 유사 크래시 지점은 이 한 곳뿐 확인.

## 7. 해결 (How / Method / When)
**방법: `getLoginUrl`에 미설정 가드 추가(throw 제거, `"/"` 폴백). 최소 diff.** 적용일 2026-06-22.

### 7.1 코드 수정
| 파일 | 변경 |
| --- | --- |
| `simulation_site/client/src/const.ts` | `getLoginUrl` 진입부에 **`if (!oauthPortalUrl || !appId) return "/";`** 가드 추가 |

대표 diff:
```ts
// BEFORE  const.ts
export const getLoginUrl = () => {
  const oauthPortalUrl = import.meta.env.VITE_OAUTH_PORTAL_URL;
  const appId = import.meta.env.VITE_APP_ID;
  const redirectUri = `${window.location.origin}/api/oauth/callback`;
  ...
  const url = new URL(`${oauthPortalUrl}/app-auth`);   // ← undefined → Invalid URL throw
  ...
};

// AFTER  const.ts
export const getLoginUrl = () => {
  const oauthPortalUrl = import.meta.env.VITE_OAUTH_PORTAL_URL;
  const appId = import.meta.env.VITE_APP_ID;
  // 본 시뮬은 Manus OAuth 미사용. 포털/앱ID 미설정 시 crash 대신 홈으로 폴백.
  if (!oauthPortalUrl || !appId) return "/";
  const redirectUri = `${window.location.origin}/api/oauth/callback`;
  ...
};
```

### 7.2 구조 변경
아키텍처 변경 없음(동작 가드만 추가). 효과: **OAuth 비활성 환경에서 인증 코드 경로가 안전하게 no-op** — `useAuth`/`main`/`DashboardLayout`의 `getLoginUrl()` 호출이 더는 throw하지 않고 `"/"`를 반환.

## 8. 검증 (Verification, 2026-06-22)
| 케이스 | 결과 |
| --- | --- |
| Vite 서빙 `src/const.ts`에 가드 포함 | ✅ (`GUARD` — HMR 반영 확인) |
| `getLoginUrl` 호출처(useAuth/main/DashboardLayout) throw 전파 | ✅ 제거(미설정 시 `"/"`) |
| 유사 `new URL(env)` 크래시 잔존 | ✅ 없음(해당 1곳뿐) |
| 시뮬 `GET /` | 200 유지 |
| 백엔드 `/health` | `ok:true`(8090) |
| 브라우저 Home 렌더(새로고침) | **사용자 최종확인 권장**(throw 원인 제거 완료, 헤드리스 렌더 미실시) |

## 9. 재발 방지 (Prevention)
- **env 의존 URL 빌드는 미설정 가드 필수**(`new URL`에 환경변수를 직접 넣지 않기 / 빈 값 방어).
- `useAuth` 같은 **기본 인자에서의 부수효과 호출** 주의 — 미사용 기능이 마운트만으로 실행되지 않게.
- 프론트 점검 시 **HTTP 200(셸) ≠ 정상 렌더** — React 런타임/콘솔까지 확인(본 건이 1차 오진 사례).
- `.env.example`에 OAuth 변수는 **선택(미사용)** 임을 주석으로 명시 권장.

## 10. 잔여 (Remaining)
- Manus OAuth 잔존 코드(`useAuth`의 trpc `auth.me/logout`, `DashboardLayout` 리다이렉트)는 우리 시뮬에서 미사용 — 완전 제거(슬림화)는 별건.
- 서버 로그 `[OAuth] OAUTH_SERVER_URL is not configured` 경고는 **무해**(동작 무관), 정리는 선택.
- 대시보드 데이터 패널 와이어링(BUG-001 §10)과는 별개 항목.

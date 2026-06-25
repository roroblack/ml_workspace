# vercel_site (이커머스 시뮬레이션 사이트 + Thin API)

실배포용 자리. **샘플에서는 `src/simulate_events.py`(로컬)** 가 이 역할을 대신합니다.

## 역할 (05-5 확정안 B)
- 고객이 상품 조회/장바구니/구매를 **시뮬레이션** → 이벤트를 Neon에 적재
- **Thin API**: 전송·DB I/O만, ML 추론 없음(추론은 로컬 Streamlit torch)

## 권장 구성
- 프론트: Next.js(또는 정적) — Vercel 배포
- API 라우트: `/api/events`(POST), `/api/predictions/latest`(GET), `/api/dashboard/summary`(GET), `/api/auth/login-log`(POST)
- 환경변수: `DATABASE_URL`(Neon, 풀링·SSL), `API_SECRET`

## 예시 (events 적재 의사코드)
```js
// POST /api/events  → Neon.realtime_event_log INSERT
export default async function handler(req, res) {
  const { user_id, event_type, product_id, price, event_time } = req.body;
  await sql`INSERT INTO realtime_event_log(user_id,event_type,product_id,price,event_time,source)
            VALUES (${user_id},${event_type},${product_id},${price},${event_time},'vercel_sim')`;
  res.json({ ok: true });
}
```
> 추론은 여기서 하지 않는다(서버리스에 torch 번들 금지). 로컬 Streamlit predictor가 Neon을 폴링해 예측.

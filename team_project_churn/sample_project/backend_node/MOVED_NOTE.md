# Node.js 백엔드 (보관/테스트용)

이 폴더는 **이전 Node.js 운영 백엔드(19-1 설계)** 의 보관본이다.

- 2026-06-21 결정: 운영 백엔드를 **FastAPI/Python(19-2)** 로 전환.
- Node 서버는 **가지마(`SKN32-2nd_GAJIMA_Dev/`)에서 제거**하고, **sample_project(백업·테스트 측)에만 보존**한다.
- 운영 정본 백엔드 = `SKN32-2nd_GAJIMA_Dev/backend/app/` (FastAPI).

## 실행(참고)
```bash
npm install        # mysql2/dotenv/pg (optionalDependencies)
cp .env.example .env   # MYSQL_* 채우기
node scripts/migrate.js
node src/server.js     # http://localhost:8090
```

DB 스키마(`db/schema_mysql.sql`)와 I/O 계약(`CONTRACT.md`)은 FastAPI 백엔드와 **동일**하다(19-2 §7.4 계약 불변).

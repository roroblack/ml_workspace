# 지시 기록 (2026-06-21) — MySQL 전환 · 가지마 repo 커밋경로 · 네비 수정 · SB~S9 진행

## 사용자 지시 (이번 차)
1. **대시보드 네비 수정**: 아이디 입력칸 + 뒤로가기 기능 추가(누락).
2. **가지마 커밋 경로 질문** → 답: 가지마(`SKN32-2nd_GAJIMA_Dev`)는 **자체 git repo**(`feature/backend` 브랜치). ml_workspace에서 add하면 **gitlink만** 기록되어 가지마 코드가 실제 커밋 안 됨. → **가지마는 자기 repo 안에서 커밋**. team_project_churn(리포트 등)은 ml_workspace repo.
3. **SQLite 금지 → MySQL 서버 사용**(우리 운영 DB). **Neon은 별도 = 시뮬레이션 전용.** 
4. 남은 단계 SB→S6→S3→S1→S9 **쭉 진행**, 매 스텝마다 지시 기록 + 결과 리포트 + 체크포인트.

## 적용 방침
- `app/db.py` = **MySQL(mysql.connector)** 전용(운영: face_user·login_log·churn_prediction·recommendation). 연결 미설정 시 **경고**(SQLite 폴백 제거).
- **Neon = simulation_site 로그 전용**(sim_event_log). 운영 DB와 분리.
- 커밋: 가지마 변경 → 가지마 repo(feature/backend) / team_project_churn 변경 → ml_workspace repo. 단계별 양쪽 체크포인트.
- 데이터·모델·venv는 .gitignore(코드만 커밋).

## 단계 (체크포인트·리포트)
SB 바운스 세션 실시간(29) → S6 백엔드 배치(30) → S3 가지마 구조+데이터카드(31) → S1 sample 정렬(32) → S9 점검+최종(33).

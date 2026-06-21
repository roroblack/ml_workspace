# 32. MySQL(project2db) 연결 · .env 보안 · SHAP 결과

체크포인트: 가지마 `feature/backend` `e121c0f`(db.py). 기준: 사용자 제공 MySQL 자격증명.

---

## 1. MySQL 연결 (project2db) ✅
- 자격증명(사용자 제공)을 **`.env`** 에 설정: `MYSQL_HOST/PORT/USER(project2)/PASSWORD/DATABASE(project2db)`.
- `init_db` 보강: **기존 DB(project2db) 직접 연결**(제한권한 사용자 대응 — CREATE DATABASE는 없을 때만).
- **교육과제 5개 실DB 검증 통과**:
  - ①로그인 로그(N제한) ②중복ID(ValueError) ③관리자권한(seed admin) ④이탈예측 저장 ⑤추천 저장 — 전부 `project2db`에 기록 확인.
- → 대시보드의 DB 기능(로그·이력·예측·추천)이 **실제 MySQL로 영속**. 데모 완전 DB-백업.

## 2. 보안 (.env / 개인정보 git 제외) ✅
- **가지마 `.gitignore`에 `.env` 이미 존재** + 추적된 `.env` 0개 확인.
- **ml_workspace `.gitignore`에 `.env`/`**/.env`/`*.env`(+`!*.env.example`) 추가**.
- 생성한 `.env`(가지마 루트·backend) **둘 다 `check-ignore` 통과 = git 추적 안 됨** 재확인.
- 비밀번호 등 개인정보는 `.env`에만, **`.env.example`은 placeholder만**. → git/원격 공유 0.

## 3. SHAP(시각화 13번) — 보류(환경 비호환) ⚠
- `shap 0.52`는 **numpy≥2 요구**, 본 env(numpy 1.26.4: scipy/sklearn/pandas와 호환)와 **충돌**. 설치 시 numpy가 2.0으로 올라가 **전 패키지 import 깨짐**(즉시 numpy<2로 복구·shap 제거, env 정상화).
- → **viz 13(SHAP)은 graceful 안내**("install shap")로 둠. 필요 시 **별도 env(numpy2 전용)** 또는 **구버전 shap(numpy1 호환)** 에서 `pp_eval_package` 재실행 → `shap_summary.json` 생성·배치.
- 나머지 14개 시각화는 정상. 기존 eval 산출물 무사.

## 4. 현재 완성 상태
- 대시보드: 로그인(아이디/얼굴) → 개요(7모델·ROC/PR) · 고객조회+추천(MySQL 저장) · 실시간 바운스 · (관리자)15시각화·로그/이력(MySQL). 스크롤·폼 정상.
- 백엔드(Node, feature/backend): `.env`로 MySQL 연결 가능(단 `npm i mysql2 dotenv` 후). 현재 대시보드는 Python(mysql-connector)로 직접 연결.
- 보안: 비밀 git 제외 완료.

## 5. 잔여(저우선)
- SHAP: 별도 env 필요(위).
- S1 sample_project 재구조화: 운영=가지마라 보류.
- 백엔드 Node의 MySQL 실연동: `npm i` 후(대시보드는 이미 MySQL 동작).

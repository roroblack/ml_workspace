# -*- coding: utf-8 -*-
"""v3 회귀 스모크 — 이벤트 시퀀스 추가가 기존 '일별 배포본'을 깨지 않았는지 확인.
검사: ① 일별 산출물 파일 존재 ② DB daily 시퀀스 행 보존 + event 행 추가 분리
     ③ 배포본 predictor(LSTM 일별)로 실제 1명 예측이 정상 동작.
실행: python preprocessing_project/v3_event_additive/src/smoke_regression.py
"""
import os, sys, sqlite3
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SP = os.path.join(HERE, "sample_project")
DB_PATH = os.path.join(SP, "data", "churn.db")


def main():
    ok = True
    # ① 일별 산출물 파일
    daily_files = [os.path.join(SP, "data", "processed", "features.csv"),
                   os.path.join(SP, "data", "processed", "sequences.npz"),
                   os.path.join(SP, "models", "lstm.pth"),
                   os.path.join(SP, "models", "seq_scaler.npz")]
    for f in daily_files:
        exists = os.path.exists(f)
        ok &= exists
        print(f"  [파일] {'OK' if exists else 'MISSING'} {os.path.relpath(f, HERE)}")

    # ② DB seq_type 분리
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute("PRAGMA table_info(sequence_snapshot)")
    has = any(r[1] == "seq_type" for r in cur.fetchall())
    print(f"  [DB] seq_type 컬럼: {'OK' if has else 'MISSING'}")
    if has:
        cur.execute("SELECT seq_type, COUNT(*) FROM sequence_snapshot GROUP BY seq_type")
        dist = dict(cur.fetchall())
        print(f"  [DB] seq_type 분포: {dist}")
        # daily 행이 존재(보존)해야 회귀 통과
        ok &= dist.get("daily", 0) > 0
    conn.close()

    # ③ 배포본 predictor 실제 동작
    try:
        sys.path.insert(0, SP)
        from db import db_client as db
        from frontend_streamlit.services import predictor
        ids = db.all_user_ids()
        if ids:
            r = predictor.predict_user(ids[0], log=False)
            good = isinstance(r.get("churn_probability"), float) and 0.0 <= r["churn_probability"] <= 1.0
            ok &= good
            print(f"  [예측] OK user={ids[0]} prob={r['churn_probability']:.3f} risk={r['risk_level']}")
        else:
            print("  [예측] SKIP — realtime_event_log 비어있음(파이프라인 미실행)")
    except Exception as e:
        ok = False
        print(f"  [예측] FAIL — {type(e).__name__}: {e}")

    print("=" * 50)
    print(f"회귀 스모크 결과: {'PASS ✅ (일별 배포본 무영향)' if ok else 'FAIL ❌'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

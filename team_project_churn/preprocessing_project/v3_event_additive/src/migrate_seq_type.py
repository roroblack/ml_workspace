# -*- coding: utf-8 -*-
"""v3 추가형 마이그레이션: sequence_snapshot 에 seq_type 컬럼 추가(기본값 'daily').
- 멱등(이미 있으면 스킵). 기존 행은 ADD COLUMN DEFAULT 로 'daily' 채워짐.
- 기존 일별 배포본을 건드리지 않는다(컬럼 추가만). 신규 이벤트 시퀀스는 seq_type='event'.
실행: python preprocessing_project/v3_event_additive/src/migrate_seq_type.py
"""
import os, sys, sqlite3
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # team_project_churn
DB_PATH = os.path.join(HERE, "sample_project", "data", "churn.db")


def has_column(cur, table, col):
    cur.execute(f"PRAGMA table_info({table})")
    return any(r[1] == col for r in cur.fetchall())


def main():
    if not os.path.exists(DB_PATH):
        print(f"[migrate] DB 없음: {DB_PATH} — sample_project 파이프라인을 먼저 실행하세요."); return False
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    changed = False
    if not has_column(cur, "sequence_snapshot", "seq_type"):
        cur.execute("ALTER TABLE sequence_snapshot ADD COLUMN seq_type TEXT DEFAULT 'daily'")
        changed = True
        print("[migrate] sequence_snapshot.seq_type 추가(DEFAULT 'daily')")
    else:
        print("[migrate] seq_type 이미 존재 — 스킵")
    # 기존 NULL 방어(혹시 모를)
    cur.execute("UPDATE sequence_snapshot SET seq_type='daily' WHERE seq_type IS NULL")
    conn.commit()
    cur.execute("SELECT seq_type, COUNT(*) FROM sequence_snapshot GROUP BY seq_type")
    dist = cur.fetchall()
    conn.close()
    print(f"[migrate] seq_type 분포: {dict(dist) if dist else '(빈 테이블)'} | changed={changed}")
    return True


if __name__ == "__main__":
    main()

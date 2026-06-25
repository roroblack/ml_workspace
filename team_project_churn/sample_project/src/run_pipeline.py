# -*- coding: utf-8 -*-
"""전체 오케스트레이션: 생성 → 피처 → ML → DL → 배치 예측 → 요약."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src import generate_sample_data, ingest_rees46, build_features, train_ml, train_dl
from frontend_streamlit.services import predictor
from db import db_client as db


def main():
    print("=" * 60)
    # PROJECT_RULES: 최종 데이터는 REES46만. 기본값=REES46(인자 없어도 REES46).
    # 'synthetic'/'demo' 를 명시할 때만 합성 데모를 쓰며, 그 외엔 절대 합성으로 폴백하지 않는다.
    src_arg = (sys.argv[1] if len(sys.argv) > 1 else "real").lower()
    if src_arg in ("synthetic", "demo", "fake"):
        print("[pipeline] ⚠ 합성 데모 데이터 사용 — 최종 산출물 아님(REES46 아님).")
        generate_sample_data.main()
    else:
        path = src_arg if (src_arg not in ("real", "rees46") and os.path.exists(src_arg)) else None
        if not ingest_rees46.main(path):
            print("[pipeline] REES46 적재 실패 — 중단(합성 자동대체 안 함). src/2019-Nov.csv.zip 확인.")
            return
    build_features.main()
    train_ml.main()
    train_dl.main()
    print("[pipeline] 전체 사용자 배치 예측...")
    res = predictor.predict_all(log=True)
    n = len(res)
    high = sum(1 for r in res if r["risk_level"] == "high")
    med = sum(1 for r in res if r["risk_level"] == "medium")
    avg = sum(r["churn_probability"] for r in res) / max(n, 1)
    print("=" * 60)
    print(f"[완료] 예측 {n}명 | 고위험 {high} 중위험 {med} | 평균 이탈확률 {avg:.3f}")
    print(f"  DB prediction_log 행수: {db.query('SELECT COUNT(*) AS c FROM prediction_log')[0]['c']}")
    print(f"  활성 모델: tabular={db.active_model('tabular')['model_name'] if db.active_model('tabular') else '-'}, "
          f"sequence={db.active_model('sequence')['model_name'] if db.active_model('sequence') else '-'}")
    print("다음: streamlit run frontend_streamlit/app.py")


if __name__ == "__main__":
    main()

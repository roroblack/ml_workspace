# -*- coding: utf-8 -*-
"""학습된 정형 ML vs 시퀀스 DL 지표 비교 출력."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from db import db_client as db


def main():
    print("모델 비교 (model_registry)")
    print(f"{'model':16s} {'type':9s} {'AUC':>6s} {'F1':>6s} {'Recall':>7s} active")
    for r in db.query("SELECT * FROM model_registry ORDER BY model_type, model_id"):
        m = json.loads(r["metrics_json"]) if r["metrics_json"] else {}
        print(f"{r['model_name']:16s} {r['model_type']:9s} "
              f"{m.get('auc',0):6.3f} {m.get('f1',0):6.3f} {m.get('recall',0):7.3f} "
              f"{'★' if r['is_active'] else ''}")


if __name__ == "__main__":
    main()

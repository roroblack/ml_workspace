# -*- coding: utf-8 -*-
"""v4 CatBoost 설정(scaler/log/HP)을 그대로 rec 데이터에 적용 → CatBoost용 전처리 파일을 모델학습 없이 즉시 생성.
스케일러는 Y무관(X에만 fit)이라 churn→rec로 설정 재사용 가능. 산출은 _v4cfg 접미사(실탐색 결과와 충돌 방지).
실행: python make_catboost_prep_from_v4.py <cat|item|both>
"""
import os, sys, json
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PP = os.path.join(HERE, "preprocessing_project")
V4CB = os.path.join(PP, "v4_model_prep", "output", "CatBoost", "prep_CatBoost_v2.joblib")
FEAT = ["recency_days","tenure_days","ndays","n_events","n_view","n_cart","n_remove_from_cart","n_purchase","avg_price","purch_amt",
        "min_price","max_price","std_price","purchase_avg_price","remove_ratio","cart_purchase_ratio","n_categories","cat_entropy",
        "n_brands","brand_loyalty","n_sessions","events_per_session"]
COUNTS = ["ndays","n_events","n_view","n_cart","n_remove_from_cart","n_purchase","purch_amt","n_categories","n_brands","n_sessions"]
COUNT_IDX = [FEAT.index(c) for c in COUNTS]
DSETS = {"cat":  dict(dir="v4-1_rec_category", train="train_cat.parquet", test="test_cat.parquet", y="y_next_category"),
         "item": dict(dir="v4-2_rec_item",     train="train_item.parquet", test="test_item.parquet", y="y_next_item")}
MIN_COUNT = 10


def transform(Xfit, prep):
    a = Xfit.copy()
    if prep["log_counts"]:
        a[:, COUNT_IDX] = np.log1p(np.clip(a[:, COUNT_IDX], 0, None))
    sc = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler()}.get(prep["scaler"])
    if sc is not None:
        sc.fit(a); a = sc.transform(a)
    return a, sc


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    v4 = joblib.load(V4CB)
    prep = v4["prep"]; hp = v4["hp"]
    print(f"[v4 CatBoost 설정] prep={prep} | hp={hp}")
    keys = ["cat", "item"] if which == "both" else [which]
    for key in keys:
        ds = DSETS[key]; DIR = os.path.join(PP, ds["dir"], "output")
        tr = pd.read_parquet(os.path.join(DIR, ds["train"]))
        vc = tr[ds["y"]].value_counts(); keep = set(vc[vc >= MIN_COUNT].index)
        tr = tr[tr[ds["y"]].isin(keep)]
        le = LabelEncoder().fit(tr[ds["y"]].values); classes = le.classes_.tolist()
        ytr = le.transform(tr[ds["y"]].values)
        Xtr = np.nan_to_num(tr[FEAT].values.astype(float))
        Xa, scaler = transform(Xtr, prep)
        mo = os.path.join(DIR, "CatBoost"); os.makedirs(mo, exist_ok=True)
        # 전처리 번들(모델 없음) — v4 설정 재사용
        joblib.dump({"model_name": f"CatBoost_rec_{key}_v4cfg", "model_type": "tree",
                     "task": "multiclass_recommendation", "target": ds["y"], "feature_order": FEAT,
                     "prep": prep, "hp": hp, "scaler": scaler, "classifier": None, "classes": classes,
                     "source": "v4_CatBoost_config_no_model",
                     "note": "v4(churn) CatBoost의 전처리설정·HP 재사용. 모델 미포함 — 학습은 별도(CatBoost_rec_train_v4cfg.parquet 사용)."},
                    os.path.join(mo, "prep_CatBoost_rec_v4cfg.joblib"))
        pd.DataFrame(Xa, columns=FEAT).assign(**{ds["y"]: ytr, "user_id": tr["user_id"].values}).to_parquet(
            os.path.join(mo, "CatBoost_rec_train_v4cfg.parquet"), index=False)
        json.dump({"source": "v4 CatBoost config", "prep": prep, "hp": hp, "n_classes": len(classes),
                   "n_rows": int(len(tr)), "n_features": len(FEAT)},
                  open(os.path.join(mo, "CatBoost_v4cfg_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        with open(os.path.join(mo, "CatBoost_v4cfg_first30.txt"), "w", encoding="utf-8") as f:
            f.write(f"# CatBoost 추천({key}) — v4설정 전처리(모델없음) | {len(classes)}클래스 {len(FEAT)}피처\n")
            f.write(f"# 설정(v4 재사용): scaler={prep['scaler']} log={prep['log_counts']} | HP={hp}\n\n")
            f.write(tr[["user_id"] + FEAT + [ds["y"]]].head(30).to_string(index=False))
        print(f"  [{key}] {len(tr):,}행 · {len(classes)}클래스 → prep_CatBoost_rec_v4cfg.joblib + CatBoost_rec_train_v4cfg.parquet 저장")


if __name__ == "__main__":
    main()

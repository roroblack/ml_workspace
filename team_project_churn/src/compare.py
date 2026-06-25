# -*- coding: utf-8 -*-
"""
모델 비교 및 최적 모델 선정
============================
ML/DL 결과를 모아 비교표를 만들고, ROC-AUC 기준 최적 모델을 선정합니다.
산출물: outputs/comparison.json, outputs/comparison_table.md
"""
import os, json, sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "outputs")
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def main():
    rows = {}
    with open(os.path.join(OUT, "metrics_ml.json"), encoding="utf-8") as f:
        ml = json.load(f)
    for k, v in ml.items():
        if not k.startswith("_"):
            rows[k] = v
    with open(os.path.join(OUT, "metrics_dl.json"), encoding="utf-8") as f:
        rows.update(json.load(f))

    best = max(rows, key=lambda k: rows[k]["roc_auc"])
    out = {"models": rows, "best_model": best,
           "best_metric": "roc_auc", "best_value": rows[best]["roc_auc"]}
    with open(os.path.join(OUT, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 마크다운 비교표
    lines = ["| 모델 | Accuracy | Precision | Recall | F1 | ROC-AUC |",
             "| --- | --- | --- | --- | --- | --- |"]
    for k, v in sorted(rows.items(), key=lambda x: -x[1]["roc_auc"]):
        mark = " ⭐" if k == best else ""
        lines.append(f"| {k}{mark} | " + " | ".join(f"{v[m]:.3f}" for m in METRICS) + " |")
    table = "\n".join(lines)
    with open(os.path.join(OUT, "comparison_table.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(table)
    print(f"\n최적 모델: {best} (ROC-AUC {rows[best]['roc_auc']:.3f})")


if __name__ == "__main__":
    main()

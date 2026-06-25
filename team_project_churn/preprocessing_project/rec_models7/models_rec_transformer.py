# -*- coding: utf-8 -*-
"""v4-1/v4-2 추천 — Transformer(7번째 모델, 다중분류).

v4의 Transformer는 '주별 시퀀스' 입력이지만, v4-1/v4-2는 유저별 **집계 22피처(탭ular)** 이므로
동일 데이터에 맞춰 **Tabular Transformer**(각 피처를 토큰화 → self-attention → 다중분류)로 구성한다.
(FT-Transformer 계열의 numerical feature tokenizer + TransformerEncoder + mean-pool head)

속도최적화: StandardScaler·작은 모델(h64·2layer)·AdamW·**early stopping(val top5, patience5)**·CPU 미니배치.
산출(output/Transformer/): prep_Transformer_rec.joblib · Transformer_rec_bayes.json · Transformer_rec_train.parquet · Transformer_first30.txt
실행: python models_rec_transformer.py <cat|item> [epochs]
"""
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, joblib
import torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
DSETS = {
    "cat":  dict(dir="v4-1_rec_category", train="train_cat.parquet", test="test_cat.parquet", y="y_next_category"),
    "item": dict(dir="v4-2_rec_item",     train="train_item.parquet", test="test_item.parquet", y="y_next_item"),
}
FEAT_V2 = ["recency_days", "tenure_days", "ndays", "n_events", "n_view", "n_cart", "n_remove_from_cart",
           "n_purchase", "avg_price", "purch_amt", "min_price", "max_price", "std_price", "purchase_avg_price",
           "remove_ratio", "cart_purchase_ratio", "n_categories", "cat_entropy", "n_brands", "brand_loyalty",
           "n_sessions", "events_per_session"]


class TabTransformer(nn.Module):
    """numerical feature tokenizer + TransformerEncoder + mean-pool 다중분류 head."""
    def __init__(self, n_feat, n_cls, h=64, heads=4, layers=2, ff=128, drop=0.1):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_feat, h) * 0.02)   # 피처별 스칼라→h
        self.femb = nn.Parameter(torch.randn(n_feat, h) * 0.02)  # 피처 임베딩(위치)
        enc = nn.TransformerEncoderLayer(h, nhead=heads, dim_feedforward=ff, batch_first=True, dropout=drop)
        self.tr = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(h), nn.Dropout(drop), nn.Linear(h, n_cls))

    def forward(self, x):                       # x:[B,F]
        tok = x.unsqueeze(-1) * self.W + self.femb     # [B,F,h]
        z = self.tr(tok)                                # self-attention over feature tokens
        return self.head(z.mean(1))                     # [B,n_cls]


def topk_mrr(logits, y, k=5):
    order = np.argsort(-logits, axis=1)
    t1 = float(np.mean(order[:, 0] == y))
    tk = float(np.mean([y[i] in order[i, :k] for i in range(len(y))]))
    rr = 0.0
    for i, yt in enumerate(y):
        r = np.where(order[i] == yt)[0]
        if len(r): rr += 1.0 / (r[0] + 1)
    return t1, tk, float(rr / len(y))


def predict_logits(model, X, bs=2048):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(torch.from_numpy(X[i:i+bs]).float()).numpy())
    return np.concatenate(out)


def main():
    dskey = sys.argv[1] if len(sys.argv) > 1 else "cat"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    ds = DSETS[dskey]
    DIR = os.path.join(HERE, "preprocessing_project", ds["dir"], "output")
    tr = pd.read_parquet(os.path.join(DIR, ds["train"])); te = pd.read_parquet(os.path.join(DIR, ds["test"]))
    vc = tr[ds["y"]].value_counts(); keep = set(vc[vc >= 10].index)          # 희소클래스 제거(bayes판과 동일)
    tr = tr[tr[ds["y"]].isin(keep)]; te = te[te[ds["y"]].isin(keep)]
    le = LabelEncoder().fit(tr[ds["y"]].values); classes = le.classes_.tolist(); ncls = len(classes)
    sc = StandardScaler().fit(np.nan_to_num(tr[FEAT_V2].values.astype(float)))
    Xtr = sc.transform(np.nan_to_num(tr[FEAT_V2].values.astype(float))).astype(np.float32)
    Xte = sc.transform(np.nan_to_num(te[FEAT_V2].values.astype(float))).astype(np.float32)
    ytr = le.transform(tr[ds["y"]].values); yte = le.transform(te[ds["y"]].values)
    Xh, Xv, yh, yv = train_test_split(Xtr, ytr, test_size=0.2, random_state=SEED, stratify=ytr)
    mo = os.path.join(DIR, "Transformer"); os.makedirs(mo, exist_ok=True)
    print(f"[{dskey}/Transformer] train {len(Xtr):,}/test {len(Xte):,} | {ncls}클래스 | epochs≤{epochs}", flush=True)

    model = TabTransformer(len(FEAT_V2), ncls)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    Xh_t, yh_t = torch.from_numpy(Xh).float(), torch.from_numpy(yh).long()
    bs, best, best_state, patience, bad = 512, -1.0, None, 6, 0
    t0 = time.time()
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xh_t))
        for i in range(0, len(Xh_t), bs):
            idx = perm[i:i+bs]; opt.zero_grad()
            loss = lossf(model(Xh_t[idx]), yh_t[idx]); loss.backward(); opt.step()
        _, vk, _ = topk_mrr(predict_logits(model, Xv), yv, k=5)            # val top5
        if vk > best:
            best, best_state, bad = vk, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop @ep{ep} (val top5 {best:.4f})", flush=True); break
    if best_state: model.load_state_dict(best_state)
    t1, t5, mr = topk_mrr(predict_logits(model, Xte), yte, k=5)
    maj = float((yte == pd.Series(ytr).mode()[0]).mean())
    metrics = {"val_top5": round(best, 4), "oot_top1": round(t1, 4), "oot_top5": round(t5, 4),
               "oot_mrr": round(mr, 4), "n_classes": ncls, "n_features": len(FEAT_V2), "majority_top1": round(maj, 4)}
    joblib.dump({"model_name": f"Transformer_rec_{dskey}", "model_type": "transformer_tabular",
                 "task": "multiclass_recommendation", "target": ds["y"], "feature_order": FEAT_V2,
                 "scaler": sc, "state_dict": {k: v.numpy() for k, v in model.state_dict().items()},
                 "arch": {"h": 64, "heads": 4, "layers": 2, "ff": 128}, "classes": classes,
                 "metrics": metrics}, os.path.join(mo, "prep_Transformer_rec.joblib"))
    json.dump({"metrics": metrics, "epochs_max": epochs, "arch": {"h": 64, "heads": 4, "layers": 2}},
              open(os.path.join(mo, "Transformer_rec_bayes.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    pd.DataFrame(Xtr, columns=FEAT_V2).assign(**{ds["y"]: ytr, "user_id": tr["user_id"].values}).to_parquet(
        os.path.join(mo, "Transformer_rec_train.parquet"), index=False)
    with open(os.path.join(mo, "Transformer_first30.txt"), "w", encoding="utf-8") as f:
        f.write(f"# Transformer(tabular) 추천({dskey}) — 앞 30행 | {ncls}클래스 {len(FEAT_V2)}피처\n")
        f.write(f"# 성능: top1 {t1:.4f} | top5 {t5:.4f} | MRR {mr:.4f} (majority {maj:.4f})\n\n")
        f.write(tr[["user_id"] + FEAT_V2 + [ds["y"]]].head(30).to_string(index=False))
    print(f"RESULTREC\t{dskey}\tTransformer\ttop1={t1:.4f}\ttop5={t5:.4f}\tMRR={mr:.4f}\t({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

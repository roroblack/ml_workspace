# -*- coding: utf-8 -*-
"""v4 · Transformer(7번째 모델) — 시퀀스 전처리 탐색 + 내부 검증.
정형 6모델과 달리 입력이 '주별 시퀀스'([N,17,3] train / [N,3,3] test). 전처리=정규화(norm) 탐색.
- norm ∈ {none, log, zscore(per-feature)} 를 비교 → 내부 80/20 val AUC로 best 선택.
- train 17주 / test 3주 길이 상이 → Feb 직접평가는 길이정렬 후(문서화). 본 단계는 val 기준.
- 산출(output/Transformer/): Transformer_train_seq.npz(정규화본), transformer_meta.json, Transformer_전처리리포트.md
실행: python .../transformer_seq_prep.py [epochs]
"""
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
D = os.path.join(HERE, "sample_project", "data", "processed_5m")
MO = os.path.join(HERE, "preprocessing_project", "v4_model_prep", "output", "Transformer"); os.makedirs(MO, exist_ok=True)
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)


class TFEnc(nn.Module):
    def __init__(self, f=3, h=48, L=17):
        super().__init__(); self.proj = nn.Linear(f, h); self.pos = nn.Parameter(torch.zeros(1, L, h))
        enc = nn.TransformerEncoderLayer(h, nhead=4, dim_feedforward=96, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(enc, num_layers=2); self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(h, 1))
    def forward(self, x): z = self.tr(self.proj(x) + self.pos); return self.head(z.mean(1)).squeeze(1)


def normalize(X, norm, stats=None):
    a = X.astype(np.float32).copy()
    if norm == "log":
        a = np.log1p(a)
    if norm in ("zscore", "log"):
        if stats is None:
            mu = a.reshape(-1, a.shape[-1]).mean(0); sd = a.reshape(-1, a.shape[-1]).std(0) + 1e-6
        else:
            mu, sd = stats
        a = (a - mu) / sd; return a, (mu, sd)
    return a, None


def train_eval(Xtr, ytr, Xva, yva, epochs):
    torch.manual_seed(SEED)
    xt = torch.tensor(Xtr); xv = torch.tensor(Xva); yt = torch.tensor(ytr, dtype=torch.float32)
    net = TFEnc(L=Xtr.shape[1]); opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    pw = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw); n = len(xt); bs = 512
    best = 0
    for _ in range(epochs):
        net.train(); idx = torch.randperm(n)
        for i in range(0, n, bs):
            b = idx[i:i + bs]; opt.zero_grad(); lossf(net(xt[b]), yt[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad(): best = max(best, roc_auc_score(yva, torch.sigmoid(net(xv)).numpy()))
    return best


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    tr = pd.read_parquet(os.path.join(D, "train_cohort_tabular.parquet"))
    z = np.load(os.path.join(D, "train_seq.npz"))
    sel = np.isin(z["user_id"], tr["user_id"].to_numpy())
    X = z["X"][sel]; y = z["churn"][sel]                  # [Ncohort, 17, 3]
    rng = np.random.default_rng(SEED); idx = rng.permutation(len(X)); cut = int(len(idx) * 0.8)
    tri, vai = idx[:cut], idx[cut:]
    print(f"[Transformer] 코호트 시퀀스 {X.shape} | 이탈률 {y.mean()*100:.1f}% | 내부 val {len(vai)}", flush=True)

    res = {}
    t0 = time.time()
    for norm in ["none", "log", "zscore"]:
        Xa, stats = normalize(X[tri], norm)
        Xv, _ = normalize(X[vai], norm, stats)           # none이면 stats 무시
        auc = train_eval(Xa, y[tri], Xv, y[vai], epochs)
        res[norm] = round(float(auc), 4)
        print(f"  norm={norm:6s} val AUC {auc:.4f}", flush=True)
    best_norm = max(res, key=res.get)

    # best 정규화본 저장(시퀀스 전처리 결과 백업)
    Xn, stats = normalize(X, best_norm)
    np.savez_compressed(os.path.join(MO, "Transformer_train_seq.npz"), X=Xn, churn=y, user_id=z["user_id"][sel])
    meta = {"model": "Transformer", "input": "주별 시퀀스 [N,17,3] step=[view,cart,purchase]",
            "norm_search": res, "best_norm": best_norm, "val_auc": res[best_norm],
            "Y": "churn(7일 무활동)", "seq_len_train": 17, "seq_len_test": 3,
            "note": "train17주/test3주 길이 상이 → Feb 직접평가는 길이정렬(주별 고정+패딩) 후. 본 단계는 내부 val 기준.",
            "elapsed_sec": round(time.time() - t0, 1)}
    json.dump(meta, open(os.path.join(MO, "transformer_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    rep = [f"# v4 · Transformer — 시퀀스 전처리 결과 (인수인계용)\n",
           "## 1. 입력 X/Y",
           "- X = 주별 시퀀스 `[N,17,3]`(train)/`[N,3,3]`(test), 채널=[view,cart,purchase] 카운트.",
           f"- Y = churn(7일 무활동). 코호트 이탈률 {y.mean()*100:.1f}%.",
           "\n## 2. 전처리(정규화) 탐색 — 내부 80/20 val AUC",
           "| norm | val AUC |", "| --- | --- |"]
    for k, v in sorted(res.items(), key=lambda x: -x[1]): rep.append(f"| {k} | {v} |")
    rep += [f"\n- **선택 = {best_norm}** (val AUC {res[best_norm]}). per-feature 정규화로 채널 스케일 차이 보정.",
            "\n## 3. 산출물",
            "- `Transformer_train_seq.npz`(정규화 적용본), `transformer_meta.json`.",
            "- **주의**: train 17주 / test 3주로 길이가 달라 Feb 직접평가는 **주별 고정길이+패딩 마스크로 정렬 후** 수행(미정렬 시 val 기준). [17-6-5] S4 시퀀스 통일과 연결.",
            "\n## 4. 학습 담당 인수",
            "- 본 정규화본으로 TFEnc(2-layer, h=48) 학습. 3~5시드 평균 권장. 정형 부스팅(AUC~0.79)과 비교는 길이정렬 후 Feb로."]
    open(os.path.join(MO, "Transformer_전처리리포트.md"), "w", encoding="utf-8").write("\n".join(rep))
    print(f"[Transformer] best norm={best_norm} val AUC {res[best_norm]} | 산출 → output/Transformer/", flush=True)
    print(f"RESULT\tTransformer\t-\t{res[best_norm]}(val)\t-\t-\tnorm={best_norm}", flush=True)


if __name__ == "__main__":
    main()

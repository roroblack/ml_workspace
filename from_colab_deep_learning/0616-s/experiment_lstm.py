# -*- coding: utf-8 -*-
# IMDB LSTM 성능 개선 실험 스크립트
# 섹션 13의 개선안(MAX_LEN/EMBED/HIDDEN/NUM_LAYERS/양방향/학습률)을 통제 실험으로 비교합니다.
# tensorflow 없이 keras imdb.load_data / pad_sequences 동작을 직접 재현합니다.

import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix

NUM_WORDS = 10000
SEED = 111
DATA_DIR = os.path.expanduser("~/.keras/datasets")
RESULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_results.json")
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_log.txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(os.cpu_count() or 4)

logf = open(LOG_PATH, "w", encoding="utf-8")
def log(msg):
    print(msg, flush=True)
    logf.write(str(msg) + "\n"); logf.flush()


def load_imdb(num_words, start_char=1, oov_char=2, index_from=3, seed=113):
    # keras.datasets.imdb.load_data 동작 재현
    with np.load(os.path.join(DATA_DIR, "imdb.npz"), allow_pickle=True) as f:
        x_train, labels_train = f["x_train"], f["y_train"]
        x_test, labels_test = f["x_test"], f["y_test"]
    rng = np.random.RandomState(seed)
    idx = np.arange(len(x_train)); rng.shuffle(idx)
    x_train = x_train[idx]; labels_train = labels_train[idx]
    idx = np.arange(len(x_test)); rng.shuffle(idx)
    x_test = x_test[idx]; labels_test = labels_test[idx]
    xs = np.concatenate([x_train, x_test])
    labels = np.concatenate([labels_train, labels_test])
    xs = [[start_char] + [w + index_from for w in x] for x in xs]
    xs = [[w if w < num_words else oov_char for w in x] for x in xs]
    split = len(x_train)
    return (xs[:split], labels[:split]), (xs[split:], labels[split:])


def pad_sequences(seqs, maxlen, value=0):
    # keras pad_sequences(padding='post', truncating='post') 재현
    out = np.full((len(seqs), maxlen), value, dtype=np.int64)
    for i, s in enumerate(seqs):
        if len(s) == 0:
            continue
        t = s[:maxlen]
        out[i, :len(t)] = t
    return out


class IMDBDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts, dtype=torch.long)
        self.labels = torch.tensor(np.asarray(labels), dtype=torch.float32)
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


class SentimentLSTM(nn.Module):
    def __init__(self, num_words, embed_dim, hidden_dim, num_layers, dropout, bidirectional):
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(num_words, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)
    def forward(self, x):
        emb = self.embedding(x)
        out, (h, c) = self.lstm(emb)
        last = out[:, -1, :]   # 마지막 시점 출력 (단방향과 동일 규칙 유지)
        return self.fc(self.dropout(last)).squeeze(1)


def run_one(cfg, raw):
    (xtr_raw, ytr), (xte_raw, yte) = raw
    torch.manual_seed(SEED); np.random.seed(SEED)
    Xtr = pad_sequences(xtr_raw, cfg["MAX_LEN"])
    Xte = pad_sequences(xte_raw, cfg["MAX_LEN"])
    train_loader = DataLoader(IMDBDataset(Xtr, ytr), batch_size=128, shuffle=True)
    test_loader = DataLoader(IMDBDataset(Xte, yte), batch_size=256, shuffle=False)

    model = SentimentLSTM(NUM_WORDS, cfg["EMBED_DIM"], cfg["HIDDEN_DIM"],
                          cfg["NUM_LAYERS"], 0.3, cfg["BIDIRECTIONAL"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["LR"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    hist = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], "f1": []}
    best = {"test_acc": 0.0, "f1": 0.0, "epoch": 0}
    t0 = time.time()
    for ep in range(cfg["EPOCHS"]):
        model.train()
        rl = c = n = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(texts)
            loss = criterion(logits, labels)
            loss.backward(); optimizer.step()
            rl += loss.item() * texts.size(0)
            c += ((torch.sigmoid(logits) >= 0.5).float() == labels).sum().item()
            n += labels.size(0)
        tr_loss, tr_acc = rl / n, c / n

        model.eval(); rl = c = n = 0; preds = []; labs = []
        with torch.no_grad():
            for texts, labels in test_loader:
                texts, labels = texts.to(device), labels.to(device)
                logits = model(texts)
                rl += criterion(logits, labels).item() * texts.size(0)
                p = (torch.sigmoid(logits) >= 0.5).float()
                c += (p == labels).sum().item(); n += labels.size(0)
                preds.append(p.cpu()); labs.append(labels.cpu())
        te_loss, te_acc = rl / n, c / n
        preds = torch.cat(preds).numpy().astype(int)
        labs = torch.cat(labs).numpy().astype(int)
        f1 = f1_score(labs, preds)
        scheduler.step()

        hist["train_loss"].append(tr_loss); hist["train_acc"].append(tr_acc)
        hist["test_loss"].append(te_loss); hist["test_acc"].append(te_acc); hist["f1"].append(f1)
        if te_acc > best["test_acc"]:
            best = {"test_acc": te_acc, "f1": f1, "epoch": ep + 1,
                    "cm": confusion_matrix(labs, preds).tolist()}
        log(f"  [{cfg['name']}] ep{ep+1}/{cfg['EPOCHS']} "
            f"tr_loss {tr_loss:.4f} tr_acc {tr_acc*100:.2f}% | "
            f"te_loss {te_loss:.4f} te_acc {te_acc*100:.2f}% f1 {f1*100:.2f}%")
    elapsed = time.time() - t0
    return {"name": cfg["name"], "desc": cfg["desc"], "cfg": cfg,
            "n_params": n_params, "elapsed_sec": round(elapsed, 1),
            "history": hist, "best": best,
            "final": {"test_acc": hist["test_acc"][-1], "f1": hist["f1"][-1],
                      "test_loss": hist["test_loss"][-1]}}


def main():
    log(f"device={device}, threads={torch.get_num_threads()}, cpu_count={os.cpu_count()}")
    log("IMDB 데이터 로딩...")
    raw = load_imdb(NUM_WORDS)
    log(f"train={len(raw[0][0])}, test={len(raw[1][0])}")

    base = dict(EMBED_DIM=30, HIDDEN_DIM=100, NUM_LAYERS=1, BIDIRECTIONAL=False,
                MAX_LEN=80, LR=0.001, EPOCHS=8)
    configs = [
        {**base, "name": "B0_baseline", "desc": "노트북 기본값 재현"},
        {**base, "name": "E1_maxlen200", "desc": "MAX_LEN 80->200", "MAX_LEN": 200},
        {**base, "name": "E2_bigger_dims", "desc": "EMBED 30->128, HIDDEN 100->128",
         "EMBED_DIM": 128, "HIDDEN_DIM": 128},
        {**base, "name": "E3_bilstm_2layer", "desc": "양방향 + 2층",
         "NUM_LAYERS": 2, "BIDIRECTIONAL": True},
        {**base, "name": "E4_lower_lr", "desc": "학습률 0.001->0.0005", "LR": 0.0005},
        {**base, "name": "FULL_combined", "desc": "모든 개선안 결합",
         "MAX_LEN": 200, "EMBED_DIM": 128, "HIDDEN_DIM": 128,
         "NUM_LAYERS": 2, "BIDIRECTIONAL": True, "LR": 0.0005, "EPOCHS": 10},
    ]

    results = []
    for cfg in configs:
        log(f"\n===== {cfg['name']}: {cfg['desc']} =====")
        try:
            r = run_one(cfg, raw)
            results.append(r)
            log(f"  -> best te_acc {r['best']['test_acc']*100:.2f}% "
                f"(f1 {r['best']['f1']*100:.2f}%, ep{r['best']['epoch']}) "
                f"| params {r['n_params']:,} | {r['elapsed_sec']}s")
            with open(RESULT_PATH, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log(f"  !! ERROR in {cfg['name']}: {e}")
    log("\n완료. 결과 저장: " + RESULT_PATH)


if __name__ == "__main__":
    main()
    logf.close()

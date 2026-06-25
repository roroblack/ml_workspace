# -*- coding: utf-8 -*-
"""
IMDB LSTM 하이퍼파라미터 베이지안 탐색 스크립트
==================================================
movie_imdb_lstm_pytorch.ipynb 섹션 13의 개선안을 자동으로 탐색합니다.

탐색 대상 하이퍼파라미터
  - MAX_LEN       : 리뷰 최대 길이        [80, 120, 200]
  - EMBED_DIM     : 임베딩 차원           [30, 64, 128]
  - HIDDEN_DIM    : LSTM 은닉 차원        [64, 128, 256]
  - NUM_LAYERS    : LSTM 층 수            [1, 2]
  - BIDIRECTIONAL : 양방향 여부           [False, True]
  - LEARNING_RATE : 학습률 (로그 스케일)  [3e-4, 2e-3]
  - DROPOUT       : 드롭아웃 비율         [0.2, 0.5]

방법
  - optuna 의 TPE(Tree-structured Parzen Estimator) 샘플러를 사용합니다.
    TPE 는 이전 시도 결과로 사후분포를 갱신해 다음 후보를 고르는 베이지안 최적화 기법입니다.
  - MedianPruner 로 성능이 낮은 시도를 에포크 도중에 조기 종료해 시간을 절약합니다.
  - 테스트셋에 과적합하지 않도록 학습 데이터에서 검증셋(20%)을 분리해
    "검증 정확도"를 최적화 목표로 삼고, 최종 best 구성만 테스트셋으로 평가합니다.

실행 예시
  python bayesian_search.py --trials 30 --epochs 4
  python bayesian_search.py --trials 50 --epochs 5 --timeout 7200
스터디는 sqlite 에 저장되므로 중단 후 같은 명령으로 재실행하면 이어서 탐색합니다.
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import f1_score
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# ----------------------------------------------------------------------
# 기본 설정
# ----------------------------------------------------------------------
NUM_WORDS = 10000
SEED = 111
VAL_RATIO = 0.2  # 학습 데이터 중 검증셋 비율

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.expanduser("~/.keras/datasets")
RESULT_PATH = os.path.join(HERE, "bayesian_results.json")
LOG_PATH = os.path.join(HERE, "bayesian_log.txt")
STUDY_DB = "sqlite:///" + os.path.join(HERE, "bayesian_study.db").replace("\\", "/")
STUDY_NAME = "imdb_lstm_tpe"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(os.cpu_count() or 4)

_logf = open(LOG_PATH, "a", encoding="utf-8")
def log(msg):
    print(msg, flush=True)
    _logf.write(str(msg) + "\n")
    _logf.flush()


# ----------------------------------------------------------------------
# 데이터 준비 (tensorflow 없이 keras imdb.load_data / pad_sequences 재현)
# ----------------------------------------------------------------------
def load_imdb(num_words, start_char=1, oov_char=2, index_from=3, seed=113):
    with np.load(os.path.join(DATA_DIR, "imdb.npz"), allow_pickle=True) as f:
        x_train, labels_train = f["x_train"], f["y_train"]
        x_test, labels_test = f["x_test"], f["y_test"]
    rng = np.random.RandomState(seed)
    idx = np.arange(len(x_train)); rng.shuffle(idx)
    x_train, labels_train = x_train[idx], labels_train[idx]
    idx = np.arange(len(x_test)); rng.shuffle(idx)
    x_test, labels_test = x_test[idx], labels_test[idx]
    xs = np.concatenate([x_train, x_test])
    labels = np.concatenate([labels_train, labels_test])
    xs = [[start_char] + [w + index_from for w in x] for x in xs]
    xs = [[w if w < num_words else oov_char for w in x] for x in xs]
    split = len(x_train)
    return (xs[:split], labels[:split]), (xs[split:], labels[split:])


def pad_sequences(seqs, maxlen, value=0):
    out = np.full((len(seqs), maxlen), value, dtype=np.int64)
    for i, s in enumerate(seqs):
        if s:
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


# 전역 데이터 (한 번만 로딩)
RAW = None
PAD_CACHE = {}        # MAX_LEN -> (X_train_full padded)
TEST_CACHE = {}       # MAX_LEN -> X_test padded
TRAIN_IDX = None      # 학습용 인덱스
VAL_IDX = None        # 검증용 인덱스


def prepare_data():
    global RAW, TRAIN_IDX, VAL_IDX
    log("IMDB 데이터 로딩...")
    RAW = load_imdb(NUM_WORDS)
    n_train = len(RAW[0][0])
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n_train)
    n_val = int(n_train * VAL_RATIO)
    VAL_IDX = perm[:n_val]
    TRAIN_IDX = perm[n_val:]
    log(f"학습 {len(TRAIN_IDX)} / 검증 {len(VAL_IDX)} / 테스트 {len(RAW[1][0])}")


def get_padded(maxlen):
    if maxlen not in PAD_CACHE:
        xs_tr, _ = RAW[0]
        xs_te, _ = RAW[1]
        PAD_CACHE[maxlen] = pad_sequences(xs_tr, maxlen)
        TEST_CACHE[maxlen] = pad_sequences(xs_te, maxlen)
    return PAD_CACHE[maxlen], TEST_CACHE[maxlen]


# ----------------------------------------------------------------------
# 모델
# ----------------------------------------------------------------------
class SentimentLSTM(nn.Module):
    def __init__(self, num_words, embed_dim, hidden_dim, num_layers, dropout, bidirectional):
        super().__init__()
        self.embedding = nn.Embedding(num_words, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)
    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        last = out[:, -1, :]
        return self.fc(self.dropout(last)).squeeze(1)


# ----------------------------------------------------------------------
# 학습/평가 한 사이클
# ----------------------------------------------------------------------
def build_loaders(maxlen, batch_size):
    X_train_full, X_test = get_padded(maxlen)
    y_train_full = np.asarray(RAW[0][1])
    y_test = np.asarray(RAW[1][1])

    full_train_ds = IMDBDataset(X_train_full, y_train_full)
    test_ds = IMDBDataset(X_test, y_test)

    train_ds = Subset(full_train_ds, TRAIN_IDX.tolist())
    val_ds = Subset(full_train_ds, VAL_IDX.tolist())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    rl = c = n = 0
    preds, labs = [], []
    for texts, labels in loader:
        texts, labels = texts.to(device), labels.to(device)
        logits = model(texts)
        rl += criterion(logits, labels).item() * texts.size(0)
        p = (torch.sigmoid(logits) >= 0.5).float()
        c += (p == labels).sum().item(); n += labels.size(0)
        preds.append(p.cpu()); labs.append(labels.cpu())
    preds = torch.cat(preds).numpy().astype(int)
    labs = torch.cat(labs).numpy().astype(int)
    return rl / n, c / n, f1_score(labs, preds)


def train_eval(params, epochs, train_loader, val_loader, trial=None):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = SentimentLSTM(NUM_WORDS, params["embed_dim"], params["hidden_dim"],
                          params["num_layers"], params["dropout"],
                          params["bidirectional"]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_val_acc = 0.0
    for ep in range(epochs):
        model.train()
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(texts), labels)
            loss.backward(); optimizer.step()
        scheduler.step()
        _, val_acc, _ = evaluate(model, val_loader, criterion)
        best_val_acc = max(best_val_acc, val_acc)
        if trial is not None:
            trial.report(val_acc, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return best_val_acc, model


# ----------------------------------------------------------------------
# optuna objective
# ----------------------------------------------------------------------
def make_objective(epochs):
    def objective(trial):
        params = {
            "max_len": trial.suggest_categorical("max_len", [80, 120, 200]),
            "embed_dim": trial.suggest_categorical("embed_dim", [30, 64, 128]),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            "num_layers": trial.suggest_int("num_layers", 1, 2),
            "bidirectional": trial.suggest_categorical("bidirectional", [False, True]),
            "lr": trial.suggest_float("lr", 3e-4, 2e-3, log=True),
            "dropout": trial.suggest_float("dropout", 0.2, 0.5),
            "batch_size": 128,
        }
        train_loader, val_loader, _ = build_loaders(params["max_len"], params["batch_size"])
        t0 = time.time()
        best_val_acc, _ = train_eval(params, epochs, train_loader, val_loader, trial)
        dt = time.time() - t0
        log(f"[trial {trial.number}] val_acc {best_val_acc*100:.2f}% | {dt:.0f}s | {params}")
        return best_val_acc
    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30, help="탐색 시도 횟수")
    ap.add_argument("--epochs", type=int, default=4, help="시도당 학습 에포크")
    ap.add_argument("--timeout", type=int, default=None, help="전체 제한 시간(초)")
    ap.add_argument("--final-epochs", type=int, default=10, help="best 구성 재학습 에포크")
    args = ap.parse_args()

    log("=" * 70)
    log(f"device={device}, threads={torch.get_num_threads()}, "
        f"trials={args.trials}, epochs/trial={args.epochs}")
    prepare_data()

    sampler = TPESampler(seed=SEED)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(
        direction="maximize", study_name=STUDY_NAME, storage=STUDY_DB,
        load_if_exists=True, sampler=sampler, pruner=pruner)

    study.optimize(make_objective(args.epochs), n_trials=args.trials,
                   timeout=args.timeout, show_progress_bar=False)

    log("\n===== 탐색 종료 =====")
    log(f"완료 시도: {len(study.trials)}")
    best = study.best_trial
    log(f"best 검증 정확도: {best.value*100:.2f}%")
    log(f"best 하이퍼파라미터: {best.params}")

    # best 구성으로 전체 학습 데이터(학습+검증) 사용해 재학습 후 테스트셋 평가
    log("\nbest 구성으로 재학습 + 테스트셋 평가...")
    bp = dict(best.params); bp["batch_size"] = 128
    X_train_full, X_test = get_padded(bp["max_len"])
    full_train_ds = IMDBDataset(X_train_full, np.asarray(RAW[0][1]))
    test_ds = IMDBDataset(X_test, np.asarray(RAW[1][1]))
    full_loader = DataLoader(full_train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    _, model = train_eval(bp, args.final_epochs, full_loader, test_loader)
    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion)
    log(f"[FINAL] 테스트 정확도 {test_acc*100:.2f}% | F1 {test_f1*100:.2f}% | loss {test_loss:.4f}")

    # 결과 저장
    trials_summary = [
        {"number": t.number, "value": t.value, "state": str(t.state), "params": t.params}
        for t in study.trials
    ]
    out = {
        "best_val_acc": best.value,
        "best_params": best.params,
        "final_test_acc": test_acc,
        "final_test_f1": test_f1,
        "n_trials": len(study.trials),
        "trials": trials_summary,
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log("결과 저장: " + RESULT_PATH)

    # best 모델 가중치 저장
    torch.save(model.state_dict(), os.path.join(HERE, "imdb_lstm_best.pth"))
    log("best 모델 저장: imdb_lstm_best.pth")


if __name__ == "__main__":
    main()
    _logf.close()

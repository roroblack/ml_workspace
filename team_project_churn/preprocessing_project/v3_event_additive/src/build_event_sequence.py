# -*- coding: utf-8 -*-
"""v3 추가형 — 이벤트-단위 시퀀스 전처리(전처리만, 학습 X).
TiSASRec/Wide&Deep 계열을 위한 입력: 유저별 이벤트 시퀀스에 '행동 간 시간간격(Δt)'·
가격·시간대·event_type·category 임베딩 인덱스를 담는다. 기존 일별 배포본/5m 산출물은
건드리지 않고 신규 디렉토리·파일로만 적재(추가형).

산출(processed_eventseq/):
  event_sequences.npz : X_cont[N,L,Fc], evt_idx[N,L], cat_idx[N,L], mask[N,L], user_id[N], churn[N]
  event_cat_vocab.json : {category_id: idx} (0=PAD, 1=UNK)
  event_scaler.joblib  : 연속피처 StandardScaler(관찰구간 fit, 누수차단)
  event_schema.json    : 컬럼·X/Y 명세
DB: sequence_snapshot 에 seq_type='event' 행 적재(기존 'daily' 행 불변).

실행: python preprocessing_project/v3_event_additive/src/build_event_sequence.py [zip경로]
기본 zip: src/2019-Nov.csv.zip , USER_CAP 으로 CPU 보호.
"""
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, sqlite3, joblib

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # team_project_churn
SRC = os.path.join(HERE, "src")
OUT = os.path.join(HERE, "preprocessing_project", "v3_event_additive", "output", "processed_eventseq")
DB_PATH = os.path.join(HERE, "sample_project", "data", "churn.db")
os.makedirs(OUT, exist_ok=True)

EVENT_TYPES = ["view", "cart", "remove_from_cart", "purchase"]   # idx 0..3
DEFAULT_ZIP = os.path.join(SRC, "2019-Nov.csv.zip")
USER_CAP = 4000          # CPU 보호용 유저 표본
SEQ_LEN = 50             # 유저당 최근 이벤트 수(패딩/마스크)
TOPK_CAT = 200           # category vocab 상위 K (+PAD+UNK)
OBS_DAYS = 14            # 관찰(피처)
OUTCOME_DAYS = 7         # 결과(라벨): 무활동=이탈 (배포본 라벨과 동일 의미)
CONT = ["price_log", "dt_next_log", "dt_prev_log", "hour_norm"]  # 연속피처(Fc=4)


def load_events(zip_path):
    dt = {"event_type": "category", "price": "float32", "user_id": "int64",
          "product_id": "int64", "category_id": "int64"}
    df = pd.read_csv(zip_path, usecols=list(dt) + ["event_time"], dtype=dt)
    df["event_time"] = pd.to_datetime(df["event_time"], format="%Y-%m-%d %H:%M:%S UTC", errors="coerce")
    df = df.dropna(subset=["event_time", "user_id"])
    return df


def main(zip_path=None):
    zip_path = zip_path or DEFAULT_ZIP
    if not os.path.exists(zip_path):
        print(f"[event] zip 없음: {zip_path}"); return False
    t0 = time.time()
    df = load_events(zip_path)
    BASE = df["event_time"].min().normalize()
    CUT = BASE + pd.Timedelta(days=OBS_DAYS)
    OUT_END = CUT + pd.Timedelta(days=OUTCOME_DAYS)
    obs = df[(df.event_time >= BASE) & (df.event_time < CUT)].copy()
    out = df[(df.event_time >= CUT) & (df.event_time < OUT_END)]

    # 유저 표본(관찰구간 활동 유저)
    users_all = obs["user_id"].drop_duplicates()
    rng = np.random.default_rng(42)
    users = np.sort(users_all.sample(min(USER_CAP, len(users_all)), random_state=42).to_numpy()) \
        if len(users_all) > USER_CAP else np.sort(users_all.to_numpy())
    obs = obs[obs.user_id.isin(users)].sort_values(["user_id", "event_time"])

    # category vocab (상위 K) : 0=PAD, 1=UNK
    top = obs["category_id"].value_counts().head(TOPK_CAT).index.tolist()
    vocab = {int(c): i + 2 for i, c in enumerate(top)}
    json.dump({"PAD": 0, "UNK": 1, **{str(k): v for k, v in vocab.items()}},
              open(os.path.join(OUT, "event_cat_vocab.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 이벤트별 파생: Δt(이전/다음), price_log, hour
    g = obs.groupby("user_id", sort=False)
    obs["t"] = obs["event_time"].astype("int64") // 10**9                 # 초
    obs["dt_prev"] = obs["t"] - g["t"].shift(1)
    obs["dt_next"] = g["t"].shift(-1) - obs["t"]
    obs["dt_prev"] = obs["dt_prev"].fillna(0).clip(0, 3600 * 24)
    obs["dt_next"] = obs["dt_next"].fillna(3600 * 24).clip(0, 3600 * 24)  # 세션종료=상한
    obs["price_log"] = np.log1p(obs["price"].clip(lower=0).fillna(0))
    obs["dt_next_log"] = np.log1p(obs["dt_next"])
    obs["dt_prev_log"] = np.log1p(obs["dt_prev"])
    obs["hour_norm"] = obs["event_time"].dt.hour / 23.0
    obs["evt_idx"] = obs["event_type"].astype(str).map({e: i for i, e in enumerate(EVENT_TYPES)}).fillna(0).astype(int)
    obs["cat_idx"] = obs["category_id"].map(vocab).fillna(1).astype(int)  # UNK=1

    # 연속피처 스케일러(관찰구간 fit, 누수차단)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(obs[CONT].values.astype("float32"))
    joblib.dump({"scaler": scaler, "cont_cols": CONT}, os.path.join(OUT, "event_scaler.joblib"))
    obs[CONT] = scaler.transform(obs[CONT].values.astype("float32"))

    # 라벨: 결과구간 무활동=이탈
    out_users = set(out["user_id"].unique())
    uidx = {int(u): i for i, u in enumerate(users)}
    N, L, Fc = len(users), SEQ_LEN, len(CONT)
    Xc = np.zeros((N, L, Fc), np.float32)
    evt = np.zeros((N, L), np.int64); cat = np.zeros((N, L), np.int64)
    mask = np.zeros((N, L), np.float32)
    churn = np.array([0 if int(u) in out_users else 1 for u in users], np.int64)

    for u, grp in obs.groupby("user_id", sort=False):
        i = uidx[int(u)]
        gg = grp.tail(L)
        n = len(gg)
        sl = slice(L - n, L)                      # 뒤쪽 정렬(최근이 끝), 앞은 패딩
        Xc[i, sl, :] = gg[CONT].values.astype(np.float32)
        evt[i, sl] = gg["evt_idx"].values
        cat[i, sl] = gg["cat_idx"].values
        mask[i, sl] = 1.0

    seq_path = os.path.join(OUT, "event_sequences.npz")
    np.savez_compressed(seq_path, X_cont=Xc, evt_idx=evt, cat_idx=cat, mask=mask,
                        user_id=users.astype("int64"), churn=churn)

    schema = {
        "version": 2, "kind": "event_sequence",
        "shape": {"X_cont": [N, L, Fc], "evt_idx": [N, L], "cat_idx": [N, L], "mask": [N, L]},
        "seq_len": L, "cont_features": CONT, "event_types": EVENT_TYPES,
        "cat_vocab_size": len(vocab) + 2, "cat_vocab_file": "event_cat_vocab.json",
        "scaler_file": "event_scaler.joblib",
        "label": {"churn": f"결과구간 {OUTCOME_DAYS}일 무활동=1"},
        "windows": {"BASE": str(BASE.date()), "obs_days": OBS_DAYS, "outcome_days": OUTCOME_DAYS},
        "note": "추가형(v3). 기존 daily/5m 산출물과 독립. TiSASRec/Wide&Deep용 Δt·category 임베딩 인덱스 포함.",
    }
    json.dump(schema, open(os.path.join(OUT, "event_schema.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # ---- DB 적재: seq_type='event' (기존 daily 행 불변) ----
    db_ok = False
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
        cur.execute("PRAGMA table_info(sequence_snapshot)")
        has_seqtype = any(r[1] == "seq_type" for r in cur.fetchall())
        if not has_seqtype:
            print("[event] ⚠ seq_type 미존재 — migrate_seq_type.py 를 먼저 실행하세요. DB 적재 스킵.")
        else:
            cur.execute("DELETE FROM sequence_snapshot WHERE seq_type='event'")  # 재실행 시 event만 갱신
            cur.executemany(
                "INSERT INTO sequence_snapshot(user_id,seq_len,n_features,storage_format,artifact_path,row_index,label,seq_type) "
                "VALUES(?,?,?,?,?,?,?, 'event')",
                [(str(int(u)), L, Fc, "npz", seq_path, uidx[int(u)], int(churn[uidx[int(u)]])) for u in users])
            conn.commit(); db_ok = True
            cur.execute("SELECT seq_type, COUNT(*) FROM sequence_snapshot GROUP BY seq_type")
            print(f"[event] DB seq_type 분포: {dict(cur.fetchall())}")
        conn.close()

    print(f"[event] 유저 {N} | 이탈률 {churn.mean()*100:.1f}% | X_cont{Xc.shape} evt/cat[{N},{L}] | "
          f"vocab {len(vocab)} | DB적재={db_ok} | {time.time()-t0:.0f}s")
    print(f"[event] 저장 → {OUT}")
    return True


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

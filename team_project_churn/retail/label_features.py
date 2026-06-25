# -*- coding: utf-8 -*-
"""
Online Retail — 시계열 라벨링 + 피처 생성
==========================================
churn 정의: 비계약형 이커머스 표준 = "구매 비활동(recency)".
  관찰기간(2010-12-01~2011-08-31, 9개월)으로 피처 생성,
  결과기간(2011-09-01~2011-12-09)에 구매 없으면 이탈(1).  (누수 방지: 피처는 관찰기간만)
산출물:
  retail/data/tabular.csv  (RFM+행동 피처 + churn, 고객순 정렬)
  retail/data/seq.npz       (월별 9스텝 시퀀스 X[N,9,4], y, customer_id — tabular와 동일 순서)
  retail/outputs/label_summary.json
"""
import os, io, json, zipfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP = os.path.join(HERE, "..", "..", "from_colab", "0605-s", "online+retail.zip")
DATA = os.path.join(HERE, "data"); OUT = os.path.join(HERE, "outputs")
os.makedirs(DATA, exist_ok=True); os.makedirs(OUT, exist_ok=True)

T_CUT = pd.Timestamp("2011-09-01")           # 관찰/결과 경계
OBS_START = pd.Timestamp("2010-12-01")
MONTHS = pd.period_range("2010-12", "2011-08", freq="M")   # 관찰 9개월
TOPK_COUNTRY = 6


def load_clean():
    z = zipfile.ZipFile(ZIP)
    name = [n for n in z.namelist() if n.endswith(".xlsx")][0]
    df = pd.read_excel(io.BytesIO(z.read(name)))
    df = df.dropna(subset=["CustomerID"]).copy()
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["is_cancel"] = df["InvoiceNo"].str.startswith("C")
    # 구매(유효) 거래
    buys = df[(~df["is_cancel"]) & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    buys["Amount"] = buys["Quantity"] * buys["UnitPrice"]
    buys["ym"] = buys["InvoiceDate"].dt.to_period("M")
    return df, buys


def main():
    raw, buys = load_clean()
    obs = buys[buys["InvoiceDate"] < T_CUT].copy()
    out = buys[buys["InvoiceDate"] >= T_CUT].copy()

    pop = sorted(obs["CustomerID"].unique())           # 관찰기간 활동 고객 = 모집단
    out_customers = set(out["CustomerID"].unique())
    churn = {c: (0 if c in out_customers else 1) for c in pop}

    # ---- 정형 피처 (관찰기간) ----
    g = obs.groupby("CustomerID")
    feat = pd.DataFrame(index=pop)
    feat.index.name = "customer_id"
    last = g["InvoiceDate"].max()
    first = g["InvoiceDate"].min()
    feat["recency"] = (T_CUT - last).dt.days
    feat["frequency"] = g["InvoiceNo"].nunique()
    feat["monetary"] = g["Amount"].sum()
    feat["tenure"] = (last - first).dt.days
    feat["avg_basket"] = feat["monetary"] / feat["frequency"]
    feat["n_unique_products"] = g["StockCode"].nunique()
    feat["total_quantity"] = g["Quantity"].sum()
    feat["active_months"] = g["ym"].nunique()
    feat["avg_interpurchase"] = feat["tenure"] / (feat["frequency"] - 1).clip(lower=1)
    # 취소 건수 (관찰기간)
    cobs = raw[(raw["is_cancel"]) & (raw["InvoiceDate"] < T_CUT)]
    cancel_cnt = cobs.groupby("CustomerID")["InvoiceNo"].nunique()
    feat["cancel_count"] = cancel_cnt.reindex(pop).fillna(0).astype(int)
    # 국가 (top-k + Other) one-hot
    country = g["Country"].agg(lambda s: s.mode().iloc[0])
    top = country.value_counts().head(TOPK_COUNTRY).index
    country = country.where(country.isin(top), "Other")
    feat = feat.join(pd.get_dummies(country.rename("country"), prefix="country"))

    feat["churn"] = [churn[c] for c in pop]
    feat = feat.reset_index()
    feat.to_csv(os.path.join(DATA, "tabular.csv"), index=False)

    # ---- 월별 시퀀스 [N,9,4] : 주문수, 수량, 매출, 고유상품수 ----
    obs["ym"] = obs["InvoiceDate"].dt.to_period("M")
    monthly = obs.groupby(["CustomerID", "ym"]).agg(
        n_invoices=("InvoiceNo", "nunique"),
        qty=("Quantity", "sum"),
        amount=("Amount", "sum"),
        n_items=("StockCode", "nunique"),
    )
    F = 4
    X = np.zeros((len(pop), len(MONTHS), F), dtype=np.float32)
    cust_idx = {c: i for i, c in enumerate(pop)}
    mon_idx = {m: j for j, m in enumerate(MONTHS)}
    for (c, m), row in monthly.iterrows():
        if m in mon_idx:
            X[cust_idx[c], mon_idx[m]] = [row.n_invoices, row.qty, row.amount, row.n_items]
    y = feat["churn"].to_numpy()
    np.savez(os.path.join(DATA, "seq.npz"), X=X, y=y, customer_id=np.array(pop))

    summary = {
        "rows_raw": int(len(raw)), "rows_buys": int(len(buys)),
        "t_cut": str(T_CUT.date()),
        "obs_period": [str(OBS_START.date()), "2011-08-31"],
        "outcome_period": ["2011-09-01", "2011-12-09"],
        "population": len(pop),
        "churn_count": int(sum(churn.values())),
        "churn_rate": round(float(np.mean(y)), 4),
        "tabular_features": [c for c in feat.columns if c not in ("customer_id", "churn")],
        "seq_shape": list(X.shape),
        "seq_feature_order": ["n_invoices", "quantity", "amount", "n_unique_items"],
        "months": [str(m) for m in MONTHS],
    }
    with open(os.path.join(OUT, "label_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("라벨링 완료")
    print(f"  모집단 {len(pop)}명 | 이탈 {summary['churn_count']}명 (이탈률 {summary['churn_rate']*100:.2f}%)")
    print(f"  정형 피처 {len(summary['tabular_features'])}개 | 시퀀스 {X.shape}")


if __name__ == "__main__":
    main()

"""트랜잭션(장바구니) 데이터 로더 — 연관규칙 분석용.

각 줄이 한 거래(장바구니)인 CSV를 list[list[str]] 로 읽는다.
(0605 연관규칙 실습의 shop_groceries.csv 형식)
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..config import PROJECT_ROOT, resolve_path
from .types import DatasetBundle

GROCERIES_CSV = "archive/raw_workspace/from_colab/0605-s/shop_groceries.csv"
ONLINE_RETAIL_ZIP = "archive/raw_workspace/from_colab/0605-s/online+retail.zip"
ONLINE_RETAIL_CACHE = PROJECT_ROOT / "runs" / "_data_cache" / "online_retail_baskets.csv"


def load_groceries(**_kwargs) -> DatasetBundle:
    return _load_basket_csv(GROCERIES_CSV, name="groceries")


def load_online_retail(**_kwargs) -> DatasetBundle:
    """Online Retail .xlsx(zip) -> InvoiceNo별 장바구니. 최초 1회 파싱 후 CSV로 캐시."""
    if not ONLINE_RETAIL_CACHE.exists():
        _build_online_retail_cache()
    return _load_basket_csv(ONLINE_RETAIL_CACHE, name="online_retail")


def _build_online_retail_cache():
    import zipfile

    import pandas as pd

    print("[data] Online Retail .xlsx 최초 파싱 중... (수십 초 소요, 이후 캐시 사용)")
    zip_path = resolve_path(ONLINE_RETAIL_ZIP)
    with zipfile.ZipFile(zip_path) as z:
        inner = z.namelist()[0]
        with z.open(inner) as f:
            df = pd.read_excel(f, engine="openpyxl",
                               usecols=["InvoiceNo", "Description"], dtype=str)

    df = df.dropna(subset=["InvoiceNo", "Description"])
    df = df[~df["InvoiceNo"].str.startswith("C")]  # 취소 거래(C...) 제외
    df["Description"] = df["Description"].str.strip()

    ONLINE_RETAIL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(ONLINE_RETAIL_CACHE, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        for _, items in df.groupby("InvoiceNo")["Description"]:
            basket = sorted(set(items))
            if len(basket) >= 2:
                writer.writerow(basket)
    print(f"[data] 캐시 생성 완료: {ONLINE_RETAIL_CACHE}")


def _load_basket_csv(path: str | Path, *, name: str) -> DatasetBundle:
    transactions: list[list[str]] = []
    with open(resolve_path(path), "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            items = [c.strip() for c in row if c and c.strip()]
            if items:
                transactions.append(items)
    return DatasetBundle(
        name=name,
        task="association_rules",
        kind="transactions",
        transactions=transactions,
    )

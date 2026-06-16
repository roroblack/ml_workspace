"""Apriori 알고리즘 — 빈발 항목집합과 연관규칙(support/confidence/lift).

mlxtend 등 외부 라이브러리 없이 순수 파이썬으로 구현한다.
0605 연관규칙 실습(min_support 0.005~0.05 비교)을 그대로 재현할 수 있다.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations


def apriori_frequent_itemsets(
    transactions: list[list[str]],
    *,
    min_support: float = 0.01,
    max_len: int = 2,
) -> dict[frozenset, float]:
    """min_support 이상인 빈발 항목집합 -> support(비율) 매핑을 반환."""
    n = len(transactions)
    if n == 0:
        return {}
    tx_sets = [set(t) for t in transactions]

    # 1-항목집합
    counts: dict[frozenset, int] = defaultdict(int)
    for t in tx_sets:
        for item in t:
            counts[frozenset([item])] += 1
    freq: dict[frozenset, float] = {
        iset: c / n for iset, c in counts.items() if c / n >= min_support
    }

    all_freq = dict(freq)
    current = list(freq.keys())
    k = 2
    while current and k <= max_len:
        # 빈발 (k-1)-항목집합에 등장한 항목만으로 후보 생성 (Apriori 가지치기)
        items = sorted({i for iset in current for i in iset})
        candidates = [frozenset(c) for c in combinations(items, k)]
        if not candidates:
            break
        c_counts: dict[frozenset, int] = defaultdict(int)
        for t in tx_sets:
            for c in candidates:
                if c <= t:
                    c_counts[c] += 1
        nxt = {c: cnt / n for c, cnt in c_counts.items() if cnt / n >= min_support}
        all_freq.update(nxt)
        current = list(nxt.keys())
        k += 1

    return all_freq


def association_rules(
    transactions: list[list[str]],
    *,
    min_support: float = 0.01,
    min_confidence: float = 0.3,
    max_len: int = 2,
) -> list[dict]:
    """연관규칙 목록을 lift 내림차순으로 반환.

    각 규칙: {antecedents, consequents, support, confidence, lift}
    """
    freq = apriori_frequent_itemsets(
        transactions, min_support=min_support, max_len=max_len
    )
    rules: list[dict] = []
    for iset, support in freq.items():
        if len(iset) < 2:
            continue
        for r in range(1, len(iset)):
            for ante in combinations(sorted(iset), r):
                ante = frozenset(ante)
                cons = iset - ante
                if ante not in freq or cons not in freq:
                    continue
                confidence = support / freq[ante]
                if confidence < min_confidence:
                    continue
                lift = confidence / freq[cons]
                rules.append({
                    "antecedents": ", ".join(sorted(ante)),
                    "consequents": ", ".join(sorted(cons)),
                    "support": round(support, 4),
                    "confidence": round(confidence, 4),
                    "lift": round(lift, 4),
                })
    rules.sort(key=lambda x: x["lift"], reverse=True)
    return rules

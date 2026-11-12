"""Step 3: 표본 쌍 만들기 (6장).
사용: python -m src.preprocess.pairs
"""
import numpy as np
import pandas as pd

from src.config import load_config
from src.preprocess.universe import load_daily


def adjustment_days(cfg):
    """종목별 기준가 조정일(분할·병합·무상증자 등) 목록 (16장).
    원래 종가 변화와 KRX 등락률(ret)이 adj_tol보다 크게 어긋난 날이다."""
    tol = cfg["pairs"]["adj_tol"]
    d = (load_daily(["date", "item_code", "close", "ret"])
         .sort_values(["item_code", "date"], ignore_index=True))
    prev = d.groupby("item_code")["close"].shift()           # 같은 종목의 바로 전 거래일 종가
    raw = d["close"] / prev.where(prev > 0) - 1               # 원래 종가 변화
    hit = d[(raw - d["ret"]).abs() > tol]
    return {code: g["date"].to_numpy() for code, g in hit.groupby("item_code")}


def adjusted_between(adj, item_code, start, end):
    """start ≤ 조정일 ≤ end 인 날이 하나라도 있으면 True (양 끝 포함, 16장)."""
    days = adj.get(item_code)
    if days is None:
        return False
    i = np.searchsorted(days, np.datetime64(start), side="left")
    return bool(i < len(days) and days[i] <= np.datetime64(end))
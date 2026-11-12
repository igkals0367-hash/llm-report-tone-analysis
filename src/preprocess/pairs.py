"""Step 3: 표본 쌍 만들기 (6장).
사용: python -m src.preprocess.pairs
"""
import pathlib

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

SECTORS = pathlib.Path("configs/sectors.csv")
OVERRIDES = pathlib.Path("configs/sector_overrides.csv")
KRX_SECTORS = pathlib.Path("data/raw/krx/sectors.parquet")


def load_sectors():
    """종목코드 → 업종 큰 분류 (4장). 업종 표에 없는 종목은 sector_overrides.csv에서 가져온다."""
    krx = pd.read_parquet(KRX_SECTORS)
    groups = pd.read_csv(SECTORS, encoding="utf-8")
    unknown = set(krx["krx_sector"]) - set(groups["krx_sector"])
    if unknown:
        raise ValueError(f"configs/sectors.csv에 없는 KRX 업종: {unknown}")
    s = pd.Series(krx["krx_sector"].map(dict(zip(groups["krx_sector"], groups["sector"]))).to_numpy(),
                  index=krx["item_code"])
    over = pd.read_csv(OVERRIDES, encoding="utf-8", dtype={"item_code": str})
    both = set(over["item_code"]) & set(s.index)
    if both:
        raise ValueError(f"업종 표에 이미 있는데 오버라이드에도 있는 종목: {both}")
    return pd.concat([s, over.set_index("item_code")["sector"]])
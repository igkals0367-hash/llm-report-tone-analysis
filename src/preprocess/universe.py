"""Step 3: 날짜별 시가총액 상위 N개 보통주 명단(유니버스)을 만든다 (4장).
사용: python -m src.preprocess.universe
"""
import pathlib

import pandas as pd

from src.config import load_config

DAILY = pathlib.Path("data/raw/krx/daily")
OUT = pathlib.Path("data/processed/universe.parquet")


def load_daily(columns=None):
    """일별 파일 941개를 하나의 표로 읽는다."""
    return pd.concat((pd.read_parquet(f, columns=columns) for f in sorted(DAILY.glob("*.parquet"))),
                     ignore_index=True)


def build_universe(cfg):
    top_n = cfg["universe"]["top_n"]
    d = load_daily(["date", "item_code", "mcap"])
    d = d[d["item_code"].str.endswith("0")].copy()      # 보통주: 끝자리 0 (17장)
    d["mcap_rank"] = (d.groupby("date")["mcap"]
                       .rank(ascending=False, method="first").astype(int))
    u = (d[d["mcap_rank"] <= top_n][["date", "item_code", "mcap_rank"]]
         .sort_values(["date", "mcap_rank"], ignore_index=True))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    u.to_parquet(OUT, index=False)
    return u


if __name__ == "__main__":
    u = build_universe(load_config())
    print(f"{len(u):,}행, {u['date'].nunique()}거래일, 한 번이라도 든 종목 {u['item_code'].nunique()}개")
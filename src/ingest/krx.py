"""Step 1: KRX 일별 시세(종가·등락률·시가총액)와 업종 분류를 받아 data/raw/krx/에 저장한다.
사용: python -m src.ingest.krx 20261111   (끝 날짜. 장이 끝난 거래일로 준다)
"""
import pathlib, sys, time

from dotenv import load_dotenv
load_dotenv()                      # 17장: 반드시 pykrx import보다 먼저

import pandas as pd
from pykrx import stock
from tqdm import tqdm

from src.config import load_config

CFG = load_config()
SLEEP = CFG["ingest"]["krx_sleep_sec"]                    # 호출 사이 쉬는 시간(초)
START = CFG["ingest"]["start_date"].strftime("%Y%m%d")    # 2023-01-01 → "20230101"
OUT = pathlib.Path("data/raw/krx")
MARKETS = ["KOSPI", "KOSDAQ"]


def trading_days(start, end):
    """삼성전자가 거래된 날 = 거래일 (17장 표). 휴일로 부르면 다른 날 데이터가 오므로 이 목록만 쓴다."""
    df = stock.get_market_ohlcv_by_date(start, end, "005930")
    return [d.strftime("%Y%m%d") for d in df.index]


def fetch_day(date, out=OUT / "daily"):
    """하루치 KOSPI·KOSDAQ 시세를 18장의 열 이름으로 YYYYMMDD.parquet에 저장하고 경로를 돌려준다."""
    parts = []
    for market in MARKETS:
        df = stock.get_market_ohlcv_by_ticker(date, market=market)
        if df.empty:
            raise RuntimeError(f"{date} {market}: 빈 응답 (로그인 만료나 접속 문제일 수 있음)")
        parts.append(pd.DataFrame({
            "date": pd.to_datetime(date, format="%Y%m%d"),
            "item_code": df.index.to_numpy(),
            "market": market,
            "close": df["종가"].to_numpy(),
            "ret": df["등락률"].astype("float64").round(2).to_numpy() / 100,  # 4.74(%) → 0.0474
            "mcap": df["시가총액"].to_numpy(),
        }))
        time.sleep(SLEEP)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{date}.parquet"
    pd.concat(parts, ignore_index=True).to_parquet(path, index=False)
    return path


def fetch_sectors(date, out=OUT):
    """수집일 하루치 업종 분류를 sectors.parquet에 저장하고 경로를 돌려준다 (17장 표: 하루치만)."""
    parts = []
    for market in MARKETS:
        df = stock.get_market_sector_classifications(date, market)
        parts.append(pd.DataFrame({
            "item_code": df.index.to_numpy(),
            "krx_sector": df["업종명"].to_numpy(),
        }))
        time.sleep(SLEEP)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sectors.parquet"
    pd.concat(parts, ignore_index=True).to_parquet(path, index=False)
    return path


if __name__ == "__main__":
    end = sys.argv[1]
    days = trading_days(START, end)
    todo = [d for d in days if not (OUT / "daily" / f"{d}.parquet").exists()]
    print(f"거래일 {len(days)}일 중 {len(days) - len(todo)}일은 이미 있음, {len(todo)}일 받기")
    for d in tqdm(todo):
        fetch_day(d)
    print("업종:", fetch_sectors(days[-1]))
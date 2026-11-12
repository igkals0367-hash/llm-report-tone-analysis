"""Step 3: 표본 쌍 만들기 (6장).
사용: python -m src.preprocess.pairs
"""
import pathlib
from functools import lru_cache

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

REPORTS = pathlib.Path("data/processed/reports.parquet")
UNIVERSE = pathlib.Path("data/processed/universe.parquet")
OUT = pathlib.Path("data/processed/pairs.parquet")
FUNNEL = pathlib.Path("data/processed/pairs_funnel.csv")


@lru_cache(maxsize=None)
def _adj(tol):
    """기준가 조정일을 한 번만 계산해 두고 다시 쓴다."""
    return adjustment_days({"pairs": {"adj_tol": tol}})


def is_maintained(later, earlier, cfg):
    """6장 '유지' 조건 다섯 가지를 모두 만족하면 True. later·earlier는 reports.parquet의 한 줄."""
    p = cfg["pairs"]
    gap_ok = later["write_date"] <= earlier["write_date"] + pd.DateOffset(months=p["max_gap_months"])
    range_ok = earlier["write_date"] >= pd.Timestamp(p["search_from"])
    opinion_ok = (later["opinion_group"] == "buy" and earlier["opinion_group"] == "buy"
                  and later["opinion_rank"] == earlier["opinion_rank"])
    goal_ok = (later["has_goal_price"] and earlier["has_goal_price"]
               and later["goal_price"] == earlier["goal_price"])
    adj_ok = not adjusted_between(_adj(p["adj_tol"]), later["item_code"],
                                  earlier["write_date"], later["write_date"])
    return bool(gap_ok and range_ok and opinion_ok and goal_ok and adj_ok)


def validation_end(trading_days, cfg):
    """검증 끝 날짜 (3장). (1) D + H1 기간 ≤ 마지막 거래일, (2) 진입일 + horizon 거래일의 종가가 있음."""
    by_h1 = trading_days[-1] - pd.DateOffset(months=cfg["h1"]["window_months"])
    by_h2 = trading_days[-1 - cfg["h2"]["horizon_days"]] - pd.Timedelta(days=1)
    return min(by_h1, by_h2)


STAGES = ["전체", "유니버스", "매수·목표가 있음", "6개월 안에 직전 리포트", "R–P 유지", "P–PP 유지", "요약 있음"]


def build_pairs(reports, universe, cfg):
    """→ pairs.parquet (6장). 표본 쌍, 기간별 단계 건수, 검증 끝 날짜를 돌려준다."""
    T = pd.DatetimeIndex(np.sort(universe["date"].unique()))          # 거래일
    v_end = validation_end(T, cfg)
    per = cfg["periods"]
    b0, b1, v0 = (pd.Timestamp(per[k]) for k in ("baseline_start", "baseline_end", "validation_start"))

    # 유니버스: 게시일의 명단. 휴장일이면 직전 거래일 명단 (16장)
    r = pd.merge_asof(reports, pd.DataFrame({"list_date": T}),
                      left_on="write_date", right_on="list_date", direction="backward")
    members = universe[["date", "item_code"]].rename(columns={"date": "list_date"}).assign(in_universe=True)
    r = r.merge(members, on=["list_date", "item_code"], how="left").reset_index(drop=True)
    r["in_universe"] = r["in_universe"].eq(True)

    # 연결: 같은 증권사·종목 줄에서 바로 앞(P)과 그 앞(PP)의 위치
    pos = pd.Series(r.index)
    g = pos.groupby([r["broker_code"], r["item_code"]])
    r["p_pos"], r["pp_pos"] = g.shift(1), g.shift(2)

    sectors = load_sectors()
    gap = pd.DateOffset(months=cfg["pairs"]["max_gap_months"])
    rec = r.to_dict("records")
    counts = {k: dict.fromkeys(STAGES, 0) for k in ("baseline", "validation")}
    rows = []
    for x in rec:
        d = x["write_date"]
        period = "baseline" if b0 <= d <= b1 else "validation" if v0 <= d <= v_end else None
        if period is None:
            continue
        c = counts[period]
        c["전체"] += 1
        if not x["in_universe"]:
            continue
        c["유니버스"] += 1
        if not (x["opinion_group"] == "buy" and x["has_goal_price"]):
            continue
        c["매수·목표가 있음"] += 1
        P = rec[int(x["p_pos"])] if pd.notna(x["p_pos"]) else None
        if P is None or d > P["write_date"] + gap:
            continue
        c["6개월 안에 직전 리포트"] += 1
        if not is_maintained(x, P, cfg):
            continue
        c["R–P 유지"] += 1
        PP = rec[int(x["pp_pos"])] if pd.notna(x["pp_pos"]) else None
        if PP is None or not is_maintained(P, PP, cfg):
            continue
        c["P–PP 유지"] += 1
        if not (x["has_summary"] and P["has_summary"]):
            continue
        c["요약 있음"] += 1
        rows.append({
            "r_nid": x["nid"], "p_nid": P["nid"], "pp_nid": PP["nid"],
            "write_date": d, "broker_code": x["broker_code"], "item_code": x["item_code"],
            "sector": sectors.get(x["item_code"]), "period": period,
            "gap_rp_days": (d - P["write_date"]).days,
            "gap_ppp_days": (P["write_date"] - PP["write_date"]).days,
        })

    pairs = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(OUT, index=False)
    pd.DataFrame(counts).rename_axis("stage").to_csv(FUNNEL, encoding="utf-8")   # 거르는 단계마다 남은 수 (대시보드용)
    return pairs, counts, v_end


if __name__ == "__main__":
    cfg = load_config()
    pairs, counts, v_end = build_pairs(pd.read_parquet(REPORTS), pd.read_parquet(UNIVERSE), cfg)
    print("검증 끝 날짜:", v_end.date())
    print(pd.DataFrame(counts).to_string())
    for k, c in counts.items():
        print(f"{k}: 유지 리포트 중 직전도 유지 {c['P–PP 유지'] / c['R–P 유지']:.1%}")
    print("기간별 표본:", pairs["period"].value_counts().to_dict(), "| 업종 빈칸:", int(pairs["sector"].isna().sum()))
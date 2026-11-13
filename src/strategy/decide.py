"""decide: 유니버스 → 신호 → 목표 비중 → 위험 점검 (11장). 백테스트와 실전이 이 함수를 같이 쓴다."""
import pandas as pd

from src.risk.check import risk_check

NEXT_REPORT = "next_report"


def build_signal(reports, days, as_of, hold):
    """종목별로 가장 최근 A·C 리포트의 등급을 따른다. B는 포지션을 만들지도 바꾸지도 않는다 (10장, 16장).
    hold가 숫자면 진입(게시일 다음 거래일) 뒤 그만큼의 거래일 동안, NEXT_REPORT면 같은 증권사가
    그 종목에 다음 리포트를 낼 때까지 살아 있다. 유니버스는 게시일에 이미 확인한 표본 리포트만 등급이 있다."""
    s = reports[reports["grade"].isin(["A", "C"])]
    if hold == NEXT_REPORT:
        alive = ~reports.duplicated(["broker_code", "item_code"], keep="last")      # 그 증권사·종목의 마지막 리포트인가
        alive = alive[s.index].to_numpy()
    else:
        entry = days.searchsorted(s["write_date"].to_numpy(), side="right")         # 게시일 다음 거래일의 위치
        alive = days.searchsorted(pd.Timestamp(as_of)) - entry < hold
    s = s.assign(alive=alive).drop_duplicates("item_code", keep="last")             # 종목별 가장 최근 리포트
    s = s[s["alive"]]
    return dict(zip(s["item_code"], s["grade"]))


def to_weights(signal, cfg):
    """롱(A)·숏(C)에 각각 계좌의 50%, 쪽마다 같은 비중. 종목이 없는 쪽의 몫은 현금이다 (10장, 16장)."""
    b = cfg["backtest"]
    longs = [c for c, g in signal.items() if g == "A"]
    shorts = [c for c, g in signal.items() if g == "C"]
    w = {c: b["long_weight"] / len(longs) for c in longs}
    w.update({c: -b["short_weight"] / len(shorts) for c in shorts})
    return w


def decide(as_of, store, cfg, hold=None, enforce_ban=True, detail=False):
    """as_of 날짜라고 치고 종목별 목표 비중을 정한다. as_of 전날까지 게시된 리포트만 본다.
    detail=True면 위험 점검 전 비중도 함께 돌려준다 (Long side·Short side 계산용)."""
    reports = store.reports(before=as_of)
    signal = build_signal(reports, store.days, as_of, hold or cfg["backtest"]["holding_days"])
    raw = to_weights(signal, cfg)
    weights = risk_check(raw, as_of, cfg, enforce_ban)
    return (weights, raw) if detail else weights

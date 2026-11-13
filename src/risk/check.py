"""주문 전 점검 (11장). 공매도 금지 기간에는 숏을 끈다. 꺼진 숏의 몫은 현금으로 남는다 (10장)."""
import pandas as pd


def risk_check(weights, as_of, cfg, enforce_ban=True):
    """enforce_ban=False는 '금지 무시 (가정)' 버전이다 (16장)."""
    ban = cfg["short_ban"]
    if enforce_ban and pd.Timestamp(ban["start"]) <= pd.Timestamp(as_of) <= pd.Timestamp(ban["end"]):
        return {c: w for c, w in weights.items() if w > 0}
    return dict(weights)

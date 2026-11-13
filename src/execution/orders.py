"""주문표 (11장). 주문을 증권사로 보내는 단계는 없다."""
import pandas as pd


def make_orders(weights, holdings, prices, equity):
    """목표 비중과 지금 가진 금액의 차이만큼만 사고판다. holdings: 종목 → 보유 금액(숏은 음수), prices: 종목 → 종가."""
    rows = []
    for c in sorted(set(weights) | set(holdings)):
        have, want = holdings.get(c, 0.0), weights.get(c, 0.0) * equity
        price = prices.get(c)
        rows.append({"item_code": c, "holding": have, "target_weight": weights.get(c, 0.0), "order": want - have,
                     "order_shares": round((want - have) / price) if price else None})
    return pd.DataFrame(rows, columns=["item_code", "holding", "target_weight", "order", "order_shares"])

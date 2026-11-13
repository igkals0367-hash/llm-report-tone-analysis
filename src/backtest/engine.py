"""Step 11 백테스트 엔진 (10장). 거래일마다 decide(그날)을 부르고 그날 종가에 체결됐다고 가정한다.
등급은 인자로 받는다. 실제 등급으로 돌리는 것은 src.backtest.report이고, 그쪽이 require_lock을 거친다 (18장)."""
import numpy as np
import pandas as pd

from src.strategy.decide import decide


def run(start, end, cfg, grades, store, hold=None, enforce_ban=True, cap=None):
    """start~end 거래일마다 decide → 종가 체결 → 다음 날 손익. → 일별 표
    ret_gross: 비용 전, ret_net: 거래비용(한쪽 = 왕복의 절반)과 공매도 이자를 뺀 뒤.
    long_side = A 평균 − 유니버스 평균, short_side = 유니버스 평균 − C 평균 (위험 점검 전 신호, 비용 전).
    market: 그날 유니버스 평균 수익률 (시장 비교용 참고, 16장).
    cap: 한 종목 비중 상한 (추가 분석, 16장). None이면 주 결과와 같은 규칙이다."""
    b = cfg["backtest"]
    cost, borrow = b["cost_roundtrip"] / 2, b["borrow_rate_annual"] / b["trading_days_per_year"]
    st = store.with_grades(grades)
    days = st.days[(st.days >= pd.Timestamp(start)) & (st.days <= pd.Timestamp(end))]
    w, raw, rows = {}, {}, []
    for day in days:
        r = st.prices(until=day).iloc[-1]                        # 그날 등락률
        held = pd.Series(w, dtype=float)
        rets = r.reindex(held.index).fillna(0.0)
        gross = float((held * rets).sum())
        short_cost = float(-held[held < 0].sum()) * borrow
        prev = st.days[st.days.searchsorted(day) - 1] if st.days[0] < day else None
        members = st.universe(on=prev) if prev is not None else []
        market = float(r.reindex(members).fillna(0.0).mean()) if len(members) else 0.0   # 전 거래일 명단 평균 (4장)
        side = lambda sign: [c for c, x in raw.items() if x * sign > 0]
        la, sc = side(1), side(-1)
        long_side = float(r.reindex(la).fillna(0.0).mean()) - market if la else np.nan
        short_side = market - float(r.reindex(sc).fillna(0.0).mean()) if sc else np.nan
        drift = held * (1 + rets) / (1 + gross)                  # 하루 지난 뒤의 실제 비중
        w, raw = decide(day, st, cfg, hold, enforce_ban, detail=True, cap=cap)
        target = pd.Series(w, dtype=float)
        turnover = float(target.sub(drift, fill_value=0.0).abs().sum())   # 차이만큼만 사고판다
        opened = sum(1 for c, x in w.items() if x * held.get(c, 0.0) <= 0)   # 새로 열었거나 방향을 바꾼 포지션
        rows.append({"date": day, "market": market, "ret_gross": gross, "ret_net": (1 + gross - short_cost) * (1 - cost * turnover) - 1,
                     "long_side": long_side, "short_side": short_side, "turnover": turnover,
                     "n_long": int((target > 0).sum()), "n_short": int((target < 0).sum()), "n_opened": opened})
    out = pd.DataFrame(rows)
    out["equity_gross"], out["equity_net"] = (1 + out["ret_gross"]).cumprod(), (1 + out["ret_net"]).cumprod()
    return out


def summarize(res, cfg):
    """누적 수익, 샤프(일별 평균 ÷ 표준편차 × √거래일, 무위험 0), 최대낙폭(계좌 금액 기준) (10장, 16장)."""
    n = cfg["backtest"]["trading_days_per_year"]

    def stats(ret, eq):
        sd = ret.std()
        return {"total": float(eq.iloc[-1] - 1), "sharpe": float(ret.mean() / sd * np.sqrt(n)) if sd > 0 else float("nan"),
                "mdd": float((eq / eq.cummax() - 1).min())}

    return {"gross": stats(res["ret_gross"], res["equity_gross"]), "net": stats(res["ret_net"], res["equity_net"]),
            "long_side": float(res["long_side"].sum()), "short_side": float(res["short_side"].sum()),
            "days": int(len(res)), "avg_long": float(res["n_long"].mean()), "avg_short": float(res["n_short"].mean()),
            "turnover_per_day": float(res["turnover"].mean()),
            "avg_hold_days": float((res["n_long"] + res["n_short"]).sum() / max(res["n_opened"].sum(), 1))}
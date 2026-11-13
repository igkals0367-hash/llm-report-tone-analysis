"""Step 10 H2 (9장): 초과수익 계산, 진입일, 중복 제거, 이상치 누르기, 고정 전에는 멈춤."""
import numpy as np
import pandas as pd
import pytest

from src.analysis.h2 import PRE_DAYS, abnormal_cumsum, cumulative_abnormal_returns, dedup, event_cars, winsorize

DAYS = pd.bdate_range("2024-01-01", periods=30)


def _market():
    # X는 매일 +2%, Y는 매일 0%. 명단은 늘 X·Y → 유니버스 평균 +1%, X의 일별 초과수익 +1%
    daily = pd.DataFrame([(d, c, r) for d in DAYS for c, r in [("X", 0.02), ("Y", 0.0)]],
                         columns=["date", "item_code", "ret"])
    universe = pd.DataFrame([(d, c) for d in DAYS for c in ["X", "Y"]], columns=["date", "item_code"])
    return daily, universe


def test_car_starts_the_day_after_entry():
    T, col, cum = abnormal_cumsum(*_market())
    pub = DAYS[PRE_DAYS + 1]
    e, car, pre = event_cars(T, col, cum, [pub], ["X"], 5)
    assert T[e[0]] == DAYS[PRE_DAYS + 2]                         # 진입 = 게시일 다음 거래일
    assert np.allclose(car[0], [0.01, 0.02, 0.03, 0.04, 0.05])   # 진입 다음 날부터 하루 +1%씩
    assert np.isclose(pre[0], 0.01 * PRE_DAYS)                   # 게시일 전날까지 20거래일, 하루 +1%씩


def test_missing_day_counts_as_zero_return():                    # 16장: 데이터가 없는 날은 수익률 0
    daily, universe = _market()
    daily = daily[~((daily["item_code"] == "X") & (daily["date"] == DAYS[25]))]
    T, col, cum = abnormal_cumsum(daily, universe)
    day = cum[25, col["X"]] - cum[24, col["X"]]
    assert np.isclose(day, 0.0 - 0.0)                            # X 0%, 평균도 (0 + 0) / 2 = 0%


def test_dedup_compares_with_last_kept():                        # 1·30·70일 → 1일과 70일이 남는다
    keep = dedup(np.array([1, 30, 70, 5]), ["X|C", "X|C", "X|C", "X|A"], [11, 12, 13, 14], 60)
    assert keep.tolist() == [True, False, True, True]
    assert dedup(np.array([1, 61]), ["k", "k"], [1, 2], 60).tolist() == [True, True]   # 60거래일 차이면 안 겹친다


def test_winsorize_clips_not_drops():
    x = np.arange(101, dtype=float)
    w = winsorize(x, 1)
    assert len(w) == 101 and w.min() == 1 and w.max() == 99


def test_stops_before_lock(tmp_path, monkeypatch):               # 18장
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError):
        cumulative_abnormal_returns(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"h2": {"horizon_days": 60}})
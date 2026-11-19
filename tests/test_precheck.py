"""Step 7 고정 전 점검 (8장): 다음 리포트 찾기, 3개월 창, 쏠림 계산."""
import pandas as pd

from src.analysis.precheck import (broker_top_shares, next_report_date, short_concentrated, short_shares,
                                   within_window)

D = pd.Timestamp


def test_next_report_same_broker_and_stock():
    reports = pd.DataFrame({
        "nid": [1, 2, 3, 4],
        "write_date": [D("2024-01-10"), D("2024-01-10"), D("2024-02-01"), D("2024-03-01")],
        "broker_code": ["a", "a", "b", "a"],
        "item_code": ["X", "X", "X", "Y"]})
    pairs = pd.DataFrame({"r_nid": [1, 2]})
    got = next_report_date(pairs, reports)
    assert got[0] == D("2024-01-10")          # 같은 날 nid가 더 큰 리포트가 다음 리포트
    assert pd.isna(got[1])                   # 다른 증권사(3)·다른 종목(4)은 다음 리포트가 아니다


def test_window_is_calendar_months_inclusive():
    r = pd.Series([D("2024-01-31")] * 3)
    n = pd.Series([D("2024-04-30"), D("2024-05-01"), pd.NaT])
    assert within_window(r, n, 3).tolist() == [True, False, False]   # 1월 31일 + 3개월 = 4월 30일


def test_concentration_numbers():
    grades = pd.Series(["A", "A", "B", "B", "C", "C"])
    top = broker_top_shares(grades, pd.Series(["k", "k", "k", "m", "m", "n"]), 1)
    assert top["A"].to_dict() == {"k": 1.0}
    sh = short_shares(grades, pd.Series([1, 5, 5, 5, 2, 5]), pd.Series([5, 5, 5, 5, 5, 1]), 2)
    assert sh == {"A": 0.5, "B": 0.0, "C": 1.0}


def test_short_concentration_needs_short_pairs():   # 2026-11-19: 짧은 쌍이 하나도 없으면 몰림이 아니다
    assert not short_concentrated({"A": 0.0, "B": 0.0, "C": 0.0}, 2.0)      # v2 기준선: 0 ÷ 0을 몰림으로 보던 경우
    assert short_concentrated({"A": 0.1, "B": 0.0, "C": 0.1}, 2.0)          # B에는 없고 A·C에만 있으면 몰림
    assert short_concentrated({"A": 0.2, "B": 0.1, "C": 0.25}, 2.0)         # 둘 다 2배 이상
    assert not short_concentrated({"A": 0.15, "B": 0.1, "C": 0.3}, 2.0)     # A가 2배 미만
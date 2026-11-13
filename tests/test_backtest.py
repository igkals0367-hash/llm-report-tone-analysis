"""Step 11 엔진 검증 (10장, 11장): 미래 정보 누수 없음, 무작위 라벨이면 수익 ≈ 0, 같은 설정이면 같은 결과.
가짜 데이터만 쓴다. 표류 없이 무작위로 오르내리는 주가라 어떤 라벨도 돈을 벌 수 없어야 한다."""
import numpy as np
import pandas as pd

from src.backtest.engine import run, summarize
from src.strategy.decide import NEXT_REPORT, decide
from src.strategy.store import Store

CFG = {"backtest": {"long_weight": 0.5, "short_weight": 0.5, "holding_days": 60, "cost_roundtrip": 0.003,
                    "borrow_rate_annual": 0.06, "trading_days_per_year": 252},
       "short_ban": {"start": "2023-11-06", "end": "2024-03-29"}}
DAYS = pd.bdate_range("2024-01-01", periods=260)
CODES = [f"{i:03d}" for i in range(40)]


def world(seed=0, n_reports=300):
    rng = np.random.default_rng(seed)
    daily = pd.DataFrame({"date": np.repeat(DAYS, len(CODES)), "item_code": np.tile(CODES, len(DAYS)),
                          "ret": rng.normal(0, 0.02, len(DAYS) * len(CODES))})
    universe = daily[["date", "item_code"]]
    reports = pd.DataFrame({"nid": np.arange(n_reports), "write_date": rng.choice(DAYS[:200], n_reports),
                            "broker_code": rng.choice(list("abc"), n_reports), "item_code": rng.choice(CODES, n_reports)})
    grades = pd.DataFrame({"r_nid": reports["nid"], "grade": rng.choice(list("ABC"), n_reports)})
    return reports, daily, universe, grades


def test_no_lookahead():                         # as_of 뒤의 리포트와 주가를 지워도 그날의 결정은 같다
    reports, daily, universe, grades = world()
    as_of = DAYS[120]
    full = Store(reports, daily, universe).with_grades(grades)
    past = Store(reports[reports["write_date"] < as_of], daily[daily["date"] <= as_of], universe).with_grades(grades)
    for hold in [60, 20, NEXT_REPORT]:
        got = decide(as_of, full, CFG, hold)
        want = decide(as_of, past, CFG, hold)
        assert got.keys() == want.keys() and all(abs(got[c] - want[c]) < 1e-12 for c in got)


def test_future_returns_do_not_change_the_past():
    reports, daily, universe, grades = world()
    changed = daily.assign(ret=np.where(daily["date"] > DAYS[150], 0.05, daily["ret"]))
    a = run(DAYS[0], DAYS[150], CFG, grades, Store(reports, daily, universe))
    b = run(DAYS[0], DAYS[150], CFG, grades, Store(reports, changed, universe))
    pd.testing.assert_frame_equal(a, b)


def test_same_inputs_same_result():
    reports, daily, universe, grades = world()
    a = run(DAYS[0], DAYS[-1], CFG, grades, Store(reports, daily, universe))
    b = run(DAYS[0], DAYS[-1], CFG, grades, Store(reports, daily, universe))
    pd.testing.assert_frame_equal(a, b)


def test_random_labels_earn_about_zero():        # 비용 전 일별 수익의 평균이 0에서 3 표준오차 안
    daily_rets = []
    for seed in range(6):
        reports, daily, universe, grades = world(seed)
        daily_rets.append(run(DAYS[0], DAYS[-1], CFG, grades, Store(reports, daily, universe), enforce_ban=False)["ret_gross"])
    x = pd.concat(daily_rets)
    assert abs(x.mean()) < 3 * x.std() / np.sqrt(len(x))


def test_entry_exit_and_ban():
    reports = pd.DataFrame({"nid": [1, 2], "write_date": [DAYS[70], DAYS[70]], "broker_code": ["a", "a"],
                            "item_code": ["000", "001"]})
    grades = pd.DataFrame({"r_nid": [1, 2], "grade": ["A", "C"]})
    _, daily, universe, _ = world()
    st = Store(reports, daily, universe).with_grades(grades)
    assert decide(DAYS[70], st, CFG) == {}                                   # 게시일에는 아직 안 보인다
    assert decide(DAYS[71], st, CFG) == {"000": 0.5, "001": -0.5}             # 다음 거래일 종가에 진입
    assert decide(DAYS[130], st, CFG) == {"000": 0.5, "001": -0.5}            # 60거래일째까지 보유
    assert decide(DAYS[131], st, CFG) == {}                                   # 그다음 날 정리
    early = pd.DataFrame({"nid": [1, 2], "write_date": [DAYS[10], DAYS[10]], "broker_code": ["a", "a"],
                          "item_code": ["000", "001"]})
    st2 = Store(early, daily, universe).with_grades(grades)
    assert decide(DAYS[11], st2, CFG) == {"000": 0.5}                         # 금지 기간: 숏은 끄고 그 몫은 현금
    assert decide(DAYS[11], st2, CFG, enforce_ban=False) == {"000": 0.5, "001": -0.5}


def test_latest_report_wins_and_b_changes_nothing():
    reports = pd.DataFrame({"nid": [1, 2, 3], "write_date": [DAYS[70], DAYS[80], DAYS[90]],
                            "broker_code": ["a", "b", "c"], "item_code": ["000"] * 3})
    _, daily, universe, _ = world()
    st = Store(reports, daily, universe).with_grades(pd.DataFrame({"r_nid": [1, 2, 3], "grade": ["A", "C", "B"]}))
    assert decide(DAYS[75], st, CFG) == {"000": 0.5}
    assert decide(DAYS[85], st, CFG) == {"000": -0.5}                         # A 뒤에 C가 나오면 숏으로
    assert decide(DAYS[95], st, CFG) == {"000": -0.5}                         # B는 아무것도 바꾸지 않는다
    assert decide(DAYS[141], st, CFG) == {}                                   # C 진입(81) + 60거래일 뒤 정리


def test_hold_until_next_report():
    reports = pd.DataFrame({"nid": [1, 2], "write_date": [DAYS[70], DAYS[100]], "broker_code": ["a", "a"],
                            "item_code": ["000", "000"]})
    _, daily, universe, _ = world()
    st = Store(reports, daily, universe).with_grades(pd.DataFrame({"r_nid": [1], "grade": ["A"]}))
    assert decide(DAYS[100], st, CFG, NEXT_REPORT) == {"000": 0.5}            # 다음 리포트 게시일까지는 보유
    assert decide(DAYS[101], st, CFG, NEXT_REPORT) == {}                      # 게시 다음 거래일 종가에 정리


def test_costs_only_lower_returns():
    reports, daily, universe, grades = world()
    res = run(DAYS[0], DAYS[-1], CFG, grades, Store(reports, daily, universe))
    assert (res["ret_net"] <= res["ret_gross"] + 1e-12).all()
    s = summarize(res, CFG)
    assert s["net"]["total"] < s["gross"]["total"] and s["gross"]["mdd"] <= 0

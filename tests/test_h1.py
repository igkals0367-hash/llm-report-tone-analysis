"""Step 9 H1 (9장): 점수 규칙, 다음 리포트 정의, 고정 전에는 멈춤."""
import numpy as np
import pandas as pd
import pytest

from src.analysis.h1 import next_report, next_report_scores, score_change
from src.analysis.precheck import next_report_date

N = np.nan


def test_score_rules():
    #        R 순위, R 목표가, N 순위, N 목표가 → 점수
    cases = [(4, 100, 4, 100, 0),     # 둘 다 그대로
             (4, 100, 4, 90, -1),     # 목표가 내림
             (4, 100, 4, 110, 1),     # 목표가 올림
             (4, 100, 2, 120, -1),    # 의견 내림이 하나라도 있으면 −1
             (4, 100, 2, N, -1),      # 의견을 내렸고 목표가는 없음 → −1
             (4, 100, N, 120, 1),     # 의견은 없고 목표가 올림 → +1
             (4, 100, 4, N, N),       # 매수 유지, 목표가 없음 → 판단 불가
             (4, 100, N, N, N)]       # 둘 다 없음 → 판단 불가
    got = score_change(*[[c[i] for c in cases] for i in range(4)])
    want = np.array([c[4] for c in cases], dtype=float)
    assert np.array_equal(got, want, equal_nan=True)



def test_locked_rule_excludes_any_missing():     # 잠근 원래 규칙(16장): N에 하나라도 없으면 제외
    got = score_change([4, 4, 4], [100, 100, 100], [2, 4, 4], [np.nan, 90, np.nan], missing="exclude")
    assert np.array_equal(got, [np.nan, -1.0, np.nan], equal_nan=True)


def test_next_report_matches_precheck():         # Step 7과 같은 '다음 리포트' 정의
    reports = pd.DataFrame({"nid": [1, 2, 3, 4], "broker_code": ["a", "a", "b", "a"], "item_code": ["X"] * 4,
                            "write_date": pd.to_datetime(["2024-01-10", "2024-01-10", "2024-02-01", "2024-03-01"])})
    got = next_report([1, 2, 4], reports)
    assert got["n_nid"].tolist()[:2] == [2, 4] and pd.isna(got["n_nid"][2])
    pd.testing.assert_series_equal(got["n_write_date"], next_report_date(pd.DataFrame({"r_nid": [1, 2, 4]}), reports),
                                   check_names=False)


def test_stops_before_lock(tmp_path, monkeypatch):   # 18장: cutoffs.json이 없으면 H1은 돌지 않는다
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError):
        next_report_scores(pd.DataFrame(), pd.DataFrame(), {})
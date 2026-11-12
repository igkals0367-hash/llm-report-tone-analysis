"""Step 6 톤 변화 (7장): 톤(R) − 톤(P), 톤이 없는 쌍 빼기, 다시 실행해도 같은 열."""
import pandas as pd

from src.classify.change import add_tone_change

PAIRS = pd.DataFrame({"r_nid": [3, 4], "p_nid": [1, 2], "period": ["baseline", "baseline"]})
TONE = pd.DataFrame({"nid": [1, 2, 3], "tone": [0.40, 0.10, 0.10]})


def test_change_is_r_minus_p():                  # 7장 Step 6 예: 직전 0.40, 이번 0.10 → −0.30
    out = add_tone_change(PAIRS, TONE)
    assert abs(out.loc[out["r_nid"] == 3, "tone_change"].item() - (-0.30)) < 1e-12


def test_pair_without_tone_is_dropped():         # 7장 Step 4-5: R(4)의 톤이 없으면 그 쌍은 뺀다
    assert add_tone_change(PAIRS, TONE)["r_nid"].tolist() == [3]


def test_rerun_keeps_one_set_of_columns():
    once = add_tone_change(PAIRS, TONE)
    assert list(add_tone_change(once, TONE).columns) == list(once.columns)
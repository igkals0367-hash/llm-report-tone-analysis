"""Step 5 톤 점수: 모델 없이 확인할 수 있는 부분 (7장 Step 5)."""
import pandas as pd

from src.classify.tone import label_order, report_tone


def test_label_order_from_config():              # 세 라벨의 순서는 id2label에서 읽는다
    assert label_order({0: "negative", 1: "neutral", 2: "positive"}) == [2, 1, 0]
    assert label_order({0: "Positive", 1: "Negative", 2: "Neutral"}) == [0, 2, 1]


def test_report_tone_is_mean_of_pos_minus_neg():
    probs = pd.DataFrame({"p_pos": [0.9, 0.1], "p_neu": [0.1, 0.4], "p_neg": [0.0, 0.5]})
    assert abs(report_tone(probs) - 0.25) < 1e-12   # (0.9 − 0.0 + 0.1 − 0.5) / 2
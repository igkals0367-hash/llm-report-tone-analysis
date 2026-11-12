"""Step 8 분류 (8장). 기준선 톤 변화의 두 백분위로 컷오프를 정하고 A·B·C를 붙인다.
lock(cfg)는 Step 8에서 추가한다. Step 7 점검도 이 함수로 임시 등급을 붙여 계산을 하나로 둔다."""
import numpy as np

from src.config import load_config


def fit_cutoffs(baseline_tone_changes, cfg=None):
    """기준선 톤 변화 → (lower, upper) = (33.3 백분위, 66.7 백분위) (8장, 12장)."""
    g = (cfg or load_config())["grade"]
    x = np.asarray(baseline_tone_changes, dtype=float)
    return float(np.percentile(x, g["lower_pct"])), float(np.percentile(x, g["upper_pct"]))


def assign_grade(tone_change, cutoffs):
    """A: upper보다 큼, C: lower보다 작음, B: 그 사이(경계 포함) (8장 등급 표)."""
    lower, upper = cutoffs
    x = np.asarray(tone_change, dtype=float)
    g = np.where(x > upper, "A", np.where(x < lower, "C", "B"))
    return g.item() if g.ndim == 0 else g
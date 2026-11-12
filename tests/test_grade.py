"""Step 8 등급 (8장): 33.3·66.7 백분위 컷오프, 경계는 B."""
from src.classify.grade import assign_grade, fit_cutoffs

CFG = {"grade": {"lower_pct": 33.3, "upper_pct": 66.7}}


def test_cutoffs_are_percentiles():
    lower, upper = fit_cutoffs(range(1, 10), CFG)  # 1~9: 1 + 8 × 0.333 = 3.664, 1 + 8 × 0.667 = 6.336
    assert abs(lower - 3.664) < 1e-9 and abs(upper - 6.336) < 1e-9


def test_grades_and_boundaries():
    assert list(assign_grade([-1.0, -0.1, 0.0, 0.1, 1.0], (-0.1, 0.1))) == ["C", "B", "B", "B", "A"]
    assert assign_grade(0.5, (-0.1, 0.1)) == "A"  # 하나만 넣으면 글자 하나
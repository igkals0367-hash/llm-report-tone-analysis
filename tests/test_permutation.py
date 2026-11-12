"""라벨 섞기 검정 (9장): 효과 없음이면 p = 1, 칸 안에서 완전히 갈리면 p가 작다, 같은 시드면 같은 결과."""
import numpy as np

from src.analysis.permutation import permutation_test


def test_no_effect_gives_p_one():                # 값이 모두 같으면 섞어도 차이는 늘 0 → k = n → p = (n+1)/(n+1)
    actual, p, _ = permutation_test([1.0] * 6, list("AACCAC"), ["x"] * 6, 50, 1)
    assert actual == 0 and p == 1.0


def test_clear_effect_gives_small_p():           # 두 칸 모두 A = +1, C = −1 → 실제 차이 −2
    values = [1, 1, 1, -1, -1, -1] * 2
    labels = list("AAACCC") * 2
    strata = ["x"] * 6 + ["y"] * 6
    actual, p, _ = permutation_test(values, labels, strata, 400, 7)
    assert actual == -2 and p < 0.05


def test_same_seed_same_result():
    args = ([0.1, -0.3, 0.5, 0.2, -1.0, 0.4], list("ACACAC"), list("xxxyyy"), 100)
    assert np.array_equal(permutation_test(*args, 5)[2], permutation_test(*args, 5)[2])
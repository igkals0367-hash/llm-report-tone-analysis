"""라벨 섞기 검정 (9장, H1·H2 공통)."""
import numpy as np
import pandas as pd


def permutation_test(values, labels, strata, n, seed):
    """차이 = C 평균 − A 평균. 같은 칸(strata) 안에서만 A·C 라벨을 n번 섞어 단측 p값을 구한다.
    p = (k + 1) / (n + 1), k는 섞은 차이가 실제 차이 이하인 횟수 (9장, 16장). → (실제 차이, p, 섞은 차이 배열)"""
    v = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    if not set(labels) <= {"A", "C"}:
        raise ValueError("라벨은 A와 C만 넣는다 (B는 비교에서 뺀다).")
    is_c = labels == "C"
    codes = pd.factorize(pd.Series(np.asarray(strata, dtype=object)))[0]
    cells = [np.flatnonzero(codes == k) for k in range(codes.max() + 1)]
    cells = [c for c in cells if 0 < is_c[c].sum() < len(c)]   # A·C가 함께 있는 칸만 섞어도 달라진다

    def diff(c):
        return v[c].mean() - v[~c].mean()

    actual = diff(is_c)
    rng = np.random.default_rng(seed)
    shuffled = np.empty(n)
    for i in range(n):
        c = is_c.copy()
        for cell in cells:                                      # 칸 안에서만 라벨 자리를 바꾼다
            c[cell] = c[cell][rng.permutation(len(cell))]
        shuffled[i] = diff(c)
    k = int(np.sum(shuffled <= actual + 1e-12))                 # 실제만큼 또는 더 낮게 (소수 반올림 여유)
    return float(actual), (k + 1) / (n + 1), shuffled
"""Step 11 엔진 검증 (10장, 13장 1단계 관문): 실제 주가에 무작위 라벨을 넣어 엔진이 없는 수익을 만들지 않는지 본다.
실제 등급의 성과는 계산하지 않는다. 등급 표에서는 A·B·C 개수만 쓰고, 어느 리포트가 무엇인지는 무작위로 섞는다.
사용: python -m src.backtest.verify"""
import json

import numpy as np
import pandas as pd

from src.analysis.guard import require_lock
from src.backtest.common import GRADES, new_run_folder, real_store
from src.backtest.engine import run
from src.config import load_config


SWAP = {"A": "C", "C": "A", "B": "B"}
NEW_SEEDS = 1000                                                # 참고용 무작위 실행은 첫 검증(실패 기록)과 다른 시드를 쓴다


def swap_check(a, b, tol=1e-9):
    """A와 C를 맞바꾼 실행 b가 원래 실행 a의 거울인가: 날마다 Long side(a) = −Short side(b), 그 반대도 (16장).
    → (통과 여부, 가장 큰 어긋남)"""
    gap = max(np.nanmax(np.abs(a["long_side"].to_numpy() + b["short_side"].to_numpy())),
              np.nanmax(np.abs(a["short_side"].to_numpy() + b["long_side"].to_numpy())))
    same_days = bool((a["long_side"].isna() == b["short_side"].isna()).all()
                     and (a["short_side"].isna() == b["long_side"].isna()).all())
    return bool(same_days and gap <= tol), float(gap)


def shuffled(grades, seed):
    """리포트는 그대로 두고 등급 글자만 무작위로 다시 나눠 준다."""
    return grades.assign(grade=np.random.default_rng(seed).permutation(grades["grade"].to_numpy()))


if __name__ == "__main__":
    cfg = load_config()
    require_lock()
    b, seed = cfg["backtest"], cfg["permutation"]["seed"]
    grades = pd.read_parquet(GRADES, columns=["r_nid", "grade"])
    store = real_store()
    start, end = cfg["periods"]["validation_start"], store.days[-1]
    first = shuffled(grades, seed)                               # 판정: 맞바꾸기 (우연이 끼지 않는다)
    res_a = run(start, end, cfg, first, store, enforce_ban=False)
    res_b = run(start, end, cfg, first.assign(grade=first["grade"].map(SWAP)), store, enforce_ban=False)
    swap_ok, swap_gap = swap_check(res_a, res_b)
    spread = lambda r: (r["long_side"] + r["short_side"]).sum()
    print(f"맞바꾸기: 원래 A − C {spread(res_a):+.4%}, 맞바꾼 뒤 {spread(res_b):+.4%}, 일별 최대 어긋남 {swap_gap:.1e}")

    rows = []                                                    # 참고: 무작위 라벨 평균
    for i in range(b["verify_runs"]):
        res = run(start, end, cfg, shuffled(grades, seed + NEW_SEEDS + i), store, enforce_ban=False)
        rows.append({"long_side": res["long_side"].sum(), "short_side": res["short_side"].sum(),
                     "spread": (res["long_side"] + res["short_side"]).sum(), "gross": res["equity_gross"].iloc[-1] - 1})
        if (i + 1) % 10 == 0:
            print(f"무작위 라벨 {i + 1}/{b['verify_runs']}")
    t = pd.DataFrame(rows)
    again = run(start, end, cfg, first, store, enforce_ban=False)
    same = bool(again.equals(res_a))
    mean, se = t["spread"].mean(), t["spread"].std() / np.sqrt(len(t))
    zero = bool(abs(mean) <= 2 * se)
    ok = swap_ok and same                                        # 무작위 평균은 참고 기록이고 판정에 쓰지 않는다

    folder, stamp, head, meta = new_run_folder("engine_check")
    lines = [f"# 엔진 검증 (Step 11b) {stamp}", head, "",
             f"실제 주가, 보유 {b['holding_days']}거래일, 공매도 금지 무시, 비용 전", "",
             "## 판정",
             f"- 맞바꾸기 (무작위 라벨에서 A와 C만 맞바꿔 다시 실행): 원래 A − C {spread(res_a):+.4%}, 맞바꾼 뒤 "
             f"{spread(res_b):+.4%}, 일별 Long side·Short side의 최대 어긋남 {swap_gap:.1e} → **{'통과' if swap_ok else '실패'}**",
             f"- 같은 설정이면 같은 결과 (같은 라벨로 다시 실행, 일별 표 전체 비교): **{'통과' if same else '실패'}**",
             "- 미래 정보 누수: tests/test_backtest.py의 누수 테스트 두 개 (pytest)", "",
             f"엔진 검증: **{'통과' if ok else '실패'}**", "",
             f"## 참고: 무작위 라벨 {len(t)}회 (시드 {seed + NEW_SEEDS}부터, 판정에 쓰지 않음)", "",
             "| 기간 합계 | 평균 | 표준편차 | 최소 | 최대 |", "|---|---|---|---|---|"]
    for k, name in [("spread", "Long side + Short side (= A − C)"), ("long_side", "Long side (A − 시장 평균)"),
                    ("short_side", "Short side (시장 평균 − C)"), ("gross", "계좌 누적 수익")]:
        lines.append(f"| {name} | {t[k].mean():+.2%} | {t[k].std():.2%} | {t[k].min():+.2%} | {t[k].max():+.2%} |")
    lines += ["", f"A − C 평균 {mean:+.2%}, 표준오차 {se:.2%} → 0에서 2 표준오차 {'안' if zero else '밖'}", "",
              "Long side와 Short side 각각은 0이 아닐 수 있다. 표본 종목(유지 리포트가 나온 종목) 자체가 시장 평균과 다르게 "
              "움직인 몫이 양쪽에 반대 부호로 들어가기 때문이다. 둘을 더한 A − C가 무작위 라벨의 기준이다."]
    (folder / "engine_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (folder / "engine_check.json").write_text(json.dumps(
        {"pass": ok, "swap": swap_ok, "swap_gap": swap_gap, "zero": zero, "same": same, "mean_spread": float(mean), "se_spread": float(se),
         "runs": t.round(6).to_dict("records"), **meta}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(lines))
    print("기록:", folder)
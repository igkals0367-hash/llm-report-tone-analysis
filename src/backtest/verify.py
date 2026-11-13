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
    rows = []
    for i in range(b["verify_runs"]):
        res = run(start, end, cfg, shuffled(grades, seed + i), store, enforce_ban=False)
        rows.append({"long_side": res["long_side"].sum(), "short_side": res["short_side"].sum(),
                     "spread": (res["long_side"] + res["short_side"]).sum(), "gross": res["equity_gross"].iloc[-1] - 1})
        print(f"무작위 라벨 {i + 1}/{b['verify_runs']}: spread {rows[-1]['spread']:+.2%}")
    t = pd.DataFrame(rows)
    again = run(start, end, cfg, shuffled(grades, seed), store, enforce_ban=False)
    same = bool(np.isclose(again["equity_gross"].iloc[-1] - 1, t["gross"][0], rtol=0, atol=1e-12))
    mean, se = t["spread"].mean(), t["spread"].std() / np.sqrt(len(t))
    zero = bool(abs(mean) <= 2 * se)
    ok = zero and same

    folder, stamp, head, meta = new_run_folder("engine_check")
    lines = [f"# 엔진 검증 (Step 11b) {stamp}", head, "",
             f"실제 주가, 무작위 라벨 {len(t)}회 (시드 {seed}부터), 보유 {b['holding_days']}거래일, 공매도 금지 무시, 비용 전", "",
             "| 기간 합계 | 평균 | 표준편차 | 최소 | 최대 |", "|---|---|---|---|---|"]
    for k, name in [("spread", "Long side + Short side (= A − C)"), ("long_side", "Long side (A − 시장 평균)"),
                    ("short_side", "Short side (시장 평균 − C)"), ("gross", "계좌 누적 수익")]:
        lines.append(f"| {name} | {t[k].mean():+.2%} | {t[k].std():.2%} | {t[k].min():+.2%} | {t[k].max():+.2%} |")
    lines += ["", f"- 무작위 라벨이면 A − C ≈ 0: 평균 {mean:+.2%}, 표준오차 {se:.2%} → |평균| ≤ 2 × 표준오차? **{'통과' if zero else '실패'}**",
              f"- 같은 설정이면 같은 결과 (첫 시드를 다시 실행): **{'통과' if same else '실패'}**",
              "- 미래 정보 누수: tests/test_backtest.py의 누수 테스트 두 개 (pytest)", "",
              f"엔진 검증: **{'통과' if ok else '실패'}**", "",
              "Long side와 Short side 각각은 0이 아닐 수 있다. 표본 종목(유지 리포트가 나온 종목) 자체가 시장 평균과 다르게 "
              "움직인 몫이 양쪽에 반대 부호로 들어가기 때문이다. 둘을 더한 A − C가 무작위 라벨의 기준이다."]
    (folder / "engine_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (folder / "engine_check.json").write_text(json.dumps(
        {"pass": ok, "zero": zero, "same": same, "mean_spread": float(mean), "se_spread": float(se),
         "runs": t.round(6).to_dict("records"), **meta}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(lines))
    print("기록:", folder)

"""Step 11 백테스트 (10장): 실제 등급으로 돌린다. 분류 고정과 엔진 검증 통과 뒤에만 돈다 (18장, 13장).
공매도 금지 반영(실행 가능)과 금지 무시(가정) 두 버전을 나란히 보고한다 (16장). 판정에는 쓰지 않는다 (2장).
사용: python -m src.backtest.report"""
import json

import pandas as pd

from src.analysis.guard import require_lock
from src.backtest.common import GRADES, RUNS, new_run_folder, real_store
from src.backtest.engine import run, summarize
from src.config import load_config
from src.execution.orders import make_orders
from src.strategy.decide import decide

BAN = {True: "금지 반영 (실행 가능)", False: "금지 무시 (가정)"}


def require_engine_check():
    """가장 최근 엔진 검증 기록이 통과가 아니면 멈춘다 (13장 1단계 관문)."""
    found = sorted(RUNS.glob("*_engine_check/engine_check.json"))
    if not found or not json.loads(found[-1].read_text(encoding="utf-8"))["pass"]:
        raise RuntimeError("엔진 검증(python -m src.backtest.verify)을 통과한 기록이 없습니다.")
    return found[-1].parent


if __name__ == "__main__":
    cfg = load_config()
    lock = require_lock()                                        # 실제 등급은 고정 뒤에만 읽는다
    check = require_engine_check()
    b = cfg["backtest"]
    grades = pd.read_parquet(GRADES, columns=["r_nid", "grade"])
    store = real_store()
    start, end = cfg["periods"]["validation_start"], store.days[-1]
    holds = [b["holding_days"]] + list(b["alt_holding_days"])
    name = lambda h: f"{h}거래일" if isinstance(h, int) else "다음 리포트까지"

    results, summary = {}, {}
    for h in holds:
        for ban in [True, False]:
            key = f"{h}|{'ban' if ban else 'noban'}"
            results[key] = run(start, end, cfg, grades, store, hold=h, enforce_ban=ban)
            summary[key] = summarize(results[key], cfg)
            print("끝:", name(h), BAN[ban])

    folder, stamp, head, meta = new_run_folder("backtest")
    pct = "{:+.2%}".format
    lines = [f"# 백테스트 (Step 11) {stamp}", f"{head} · 고정 커밋 {lock['git_commit'][:7]} · 엔진 검증 {check.name}", "",
             f"기간 {pd.Timestamp(start).date()} ~ {end.date()} ({summary[f'{holds[0]}|ban']['days']}거래일), "
             f"롱 {b['long_weight']:.0%}·숏 {b['short_weight']:.0%}, 왕복 비용 {b['cost_roundtrip']:.1%}, "
             f"공매도 이자 연 {b['borrow_rate_annual']:.1%}. 판정에 쓰지 않는다 (2장).", ""]
    for h in holds:
        lines += [f"## 보유: {name(h)}{' (기본)' if h == holds[0] else ' (비교)'}", "",
                  "| 버전 | 누적 수익 (비용 전) | 누적 수익 (비용 후) | 샤프 (전) | 샤프 (후) | 최대낙폭 (전) | 최대낙폭 (후) | "
                  "평균 롱·숏 종목 수 | 평균 보유일 |", "|---|---|---|---|---|---|---|---|---|"]
        for ban in [True, False]:
            s = summary[f"{h}|{'ban' if ban else 'noban'}"]
            lines.append(f"| {BAN[ban]} | {pct(s['gross']['total'])} | {pct(s['net']['total'])} | {s['gross']['sharpe']:.2f} | "
                         f"{s['net']['sharpe']:.2f} | {pct(s['gross']['mdd'])} | {pct(s['net']['mdd'])} | "
                         f"{s['avg_long']:.1f}·{s['avg_short']:.1f} | {s['avg_hold_days']:.0f} |")
        s = summary[f"{h}|noban"]
        lines += ["", f"Long side (A − 시장 평균) 일별 합 {pct(s['long_side'])}, Short side (시장 평균 − C) 일별 합 "
                      f"{pct(s['short_side'])} (신호 기준, 비용 전, 금지 기간의 Short side는 정보 확인용)", ""]
    (folder / "backtest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    main = results[f"{holds[0]}|ban"]
    curves = {k: {"date": [str(d.date()) for d in r["date"]], "gross": r["equity_gross"].round(6).tolist(),
                  "net": r["equity_net"].round(6).tolist(), "long_side": r["long_side"].fillna(0).cumsum().round(6).tolist(),
                  "short_side": r["short_side"].fillna(0).cumsum().round(6).tolist()} for k, r in results.items()}
    (folder / "backtest.json").write_text(json.dumps({"summary": summary, "curves": curves, "holds": [str(h) for h in holds],
                                                      "engine_check": check.name, **meta}, ensure_ascii=False), encoding="utf-8")
    for ban in [True, False]:
        results[f"{holds[0]}|{'ban' if ban else 'noban'}"].to_csv(folder / f"daily_{'ban' if ban else 'noban'}.csv", index=False)
    st = store.with_grades(grades)                               # execution 직전의 산출물: 마지막 날 주문표 (11장)
    make_orders(decide(end, st, cfg), {}, {}, 1.0).to_csv(folder / "orders_last_day.csv", index=False)
    print("\n".join(lines))
    print("기록:", folder)

"""Step 11 백테스트 (10장): 실제 등급으로 돌린다. 분류 고정과 엔진 검증 통과 뒤에만 돈다 (18장, 13장).
공매도 금지 반영(실행 가능)과 금지 무시(가정) 두 버전을 나란히 보고한다 (16장). 판정에는 쓰지 않는다 (2장).
추가 분석: 한 종목 비중 상한 버전을 같은 조건으로 함께 돌린다. 결과를 본 뒤 추가한 것이라 주 결과를 대신하지 않는다 (16장).
시장 비교: 같은 기간 유니버스 평균 누적 수익을 참고로 적는다 (16장).
v2(text_source: pdf)는 grades_pdf·cutoffs_pdf로 돌려 runs/<시각>_backtest_pdf/에 쓴다. v2의 상한 버전은 미리 정한 것이다.
엔진은 바뀌지 않았으므로 엔진 검증은 v1 기록을 그대로 쓴다 (19장 백테스트).
사용: python -m src.backtest.report"""
import json

import pandas as pd

from src.analysis.guard import CUTOFFS, analysis_mode, require_lock
from src.backtest.common import GRADES, RUNS, new_run_folder, real_store
from src.backtest.engine import run, summarize
from src.config import load_config, source_path
from src.execution.orders import make_orders
from src.strategy.decide import decide

BAN = {True: "금지 반영 (실행 가능)", False: "금지 무시 (가정)"}
CAP_NOTE = {"summary": "추가 분석: 한 종목 상한 {:.0%} (결과를 본 뒤 추가)", "pdf": "한 종목 상한 {:.0%} (미리 정함)"}
CAP_WHY = {"summary": "끝 구간(청산 구간)에 보유 종목이 한두 개로 줄어 계좌가 크게 흔들린 것을 보고 추가했다 (15장, 16장).",
           "pdf": "v1에서 끝 구간(청산 구간) 쏠림을 본 뒤, v2에서는 처음부터 주 결과와 함께 보고하기로 정했다 (19장 백테스트)."}


def require_engine_check(kind="engine_check"):
    """가장 최근 엔진 검증 기록이 통과가 아니면 멈춘다 (13장 1단계 관문). 상한 버전은 kind="engine_check_cap"."""
    found = sorted(RUNS.glob(f"*_{kind}/engine_check.json"))
    if not found or not json.loads(found[-1].read_text(encoding="utf-8"))["pass"]:
        flag = " --cap" if kind.endswith("_cap") else ""
        raise RuntimeError(f"엔진 검증(python -m src.backtest.verify{flag})을 통과한 기록이 없습니다.")
    return found[-1].parent


def key(h, ban, cap):
    return f"{h}|{'ban' if ban else 'noban'}{'|cap' if cap else ''}"


def winddown_start(res):
    """청산 구간의 첫날: 마지막으로 새 포지션을 연 날의 다음 거래일 (16장). 그 뒤로는 종목이 줄기만 한다."""
    opened = res.index[res["n_opened"] > 0]
    nxt = opened[-1] + 1 if len(opened) else None
    return str(res.loc[nxt, "date"].date()) if nxt is not None and nxt < len(res) else None


if __name__ == "__main__":
    cfg = load_config()
    mode = analysis_mode(cfg)                                    # 글 출처: v1이면 꼬리표 없음, v2면 _pdf
    source = cfg.get("text_source", "summary")
    cut_path = source_path(CUTOFFS, cfg)
    lock = require_lock(cut_path)                                # 실제 등급은 고정 뒤에만 읽는다
    check, check_cap = require_engine_check(), require_engine_check("engine_check_cap")
    b = cfg["backtest"]
    cap = b["max_weight"]
    grades = pd.read_parquet(source_path(GRADES, cfg), columns=["r_nid", "grade"])
    cap_note = CAP_NOTE[source]
    store = real_store()
    start, end = cfg["periods"]["validation_start"], store.days[-1]
    holds = [b["holding_days"]] + list(b["alt_holding_days"])
    name = lambda h: f"{h}거래일" if isinstance(h, int) else "다음 리포트까지"

    results, summary = {}, {}
    for h in holds:
        for c in [None, cap]:
            for ban in [True, False]:
                k = key(h, ban, c)
                results[k] = run(start, end, cfg, grades, store, hold=h, enforce_ban=ban, cap=c)
                summary[k] = summarize(results[k], cfg)
                summary[k]["winddown_start"] = winddown_start(results[k])
                print("끝:", name(h), BAN[ban], cap_note.format(c) if c else "")

    first = results[key(holds[0], True, None)]                   # 시장(유니버스 평균)은 어느 실행에서나 같다
    mkt_eq = (1 + first["market"]).cumprod()
    mkt_ret = first["market"]
    n = b["trading_days_per_year"]
    market = {"total": float(mkt_eq.iloc[-1] - 1), "sharpe": float(mkt_ret.mean() / mkt_ret.std() * n ** 0.5),
              "mdd": float((mkt_eq / mkt_eq.cummax() - 1).min())}

    folder, stamp, head, meta = new_run_folder(f"backtest{mode['tag']}", cut_path)
    pct = "{:+.2%}".format
    lines = [f"# 백테스트 (Step 11) {stamp}", mode["label"],
             f"{head} · 고정 커밋 {lock['git_commit'][:7]} · 엔진 검증 {check.name}, {check_cap.name}", "",
             f"기간 {pd.Timestamp(start).date()} ~ {end.date()} ({summary[key(holds[0], True, None)]['days']}거래일), "
             f"롱 {b['long_weight']:.0%}·숏 {b['short_weight']:.0%}, 왕복 비용 {b['cost_roundtrip']:.1%}, "
             f"공매도 이자 연 {b['borrow_rate_annual']:.1%}. 판정에 쓰지 않는다 (2장).", "",
             f"시장 (유니버스 평균, 참고): 누적 {pct(market['total'])}, 샤프 {market['sharpe']:.2f}, "
             f"최대낙폭 {pct(market['mdd'])}. 전략은 롱·숏이라 시장 방향과 직접 비교하는 기준이 아니다.", ""]
    for h in holds:
        lines += [f"## 보유: {name(h)}{' (기본)' if h == holds[0] else ' (비교)'}", "",
                  "| 버전 | 누적 수익 (비용 전) | 누적 수익 (비용 후) | 샤프 (전) | 샤프 (후) | 최대낙폭 (전) | 최대낙폭 (후) | "
                  "평균 롱·숏 종목 수 | 평균 보유일 | 청산 구간 시작 |", "|---|---|---|---|---|---|---|---|---|---|"]
        for c in [None, cap]:
            for ban in [True, False]:
                s = summary[key(h, ban, c)]
                label = BAN[ban] + (f" · {cap_note.format(c)}" if c else "")
                lines.append(f"| {label} | {pct(s['gross']['total'])} | {pct(s['net']['total'])} | {s['gross']['sharpe']:.2f} | "
                             f"{s['net']['sharpe']:.2f} | {pct(s['gross']['mdd'])} | {pct(s['net']['mdd'])} | "
                             f"{s['avg_long']:.1f}·{s['avg_short']:.1f} | {s['avg_hold_days']:.0f} | {s['winddown_start'] or '-'} |")
        s = summary[key(h, False, None)]
        lines += ["", f"Long side (A − 시장 평균) 일별 합 {pct(s['long_side'])}, Short side (시장 평균 − C) 일별 합 "
                      f"{pct(s['short_side'])} (신호 기준, 비용 전, 상한과 무관, 금지 기간의 Short side는 정보 확인용)", ""]
    lines += ["상한 버전은 쪽마다 50%를 종목 수로 나누되 한 종목은 계좌의 "
              f"{cap:.0%}를 넘지 않게 하고 남는 몫은 현금으로 둔다. 쪽마다 종목이 {round(b['long_weight'] / cap)}개 이상이면 주 결과와 같다. "
              + CAP_WHY[source], ""]
    (folder / "backtest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    curves = {k: {"date": [str(d.date()) for d in r["date"]], "gross": r["equity_gross"].round(6).tolist(),
                  "net": r["equity_net"].round(6).tolist(), "long_side": r["long_side"].fillna(0).cumsum().round(6).tolist(),
                  "short_side": r["short_side"].fillna(0).cumsum().round(6).tolist()} for k, r in results.items()}
    curves["market"] = {"date": [str(d.date()) for d in first["date"]], "equity": mkt_eq.round(6).tolist()}
    (folder / "backtest.json").write_text(json.dumps({"summary": summary, "market": market, "curves": curves,
                                                      "holds": [str(h) for h in holds], "cap": cap, "text_source": source,
                                                      "engine_check": check.name, "engine_check_cap": check_cap.name, **meta},
                                                     ensure_ascii=False), encoding="utf-8")
    for c in [None, cap]:
        for ban in [True, False]:
            tag = f"{'ban' if ban else 'noban'}{'_cap' if c else ''}"
            results[key(holds[0], ban, c)].to_csv(folder / f"daily_{tag}.csv", index=False)
    st = store.with_grades(grades)                               # execution 직전의 산출물: 마지막 날 주문표 (11장)
    make_orders(decide(end, st, cfg), {}, {}, 1.0).to_csv(folder / "orders_last_day.csv", index=False)
    print("\n".join(lines))
    print("기록:", folder)
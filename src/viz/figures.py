"""Step 12 결과 그림 5종 (10장). runs/의 가장 최근 H1·H2·백테스트 기록만 읽어 그린다.
결과를 새로 계산하지 않으므로 그림의 숫자가 기록과 어긋날 수 없다. 고정 뒤에만 돈다 (18장).
사용: python -m src.viz.figures  →  reports/figures/fig1~fig5 *.png"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from src.analysis.guard import require_lock
from src.config import load_config

RUNS = pathlib.Path("runs")
OUT = pathlib.Path("reports/figures")
A, C, N = "#4C72B0", "#C44E52", "#7B7485"                       # 부드러운 파랑·빨강, 중립은 회색
INK, GRID, BAN = "#3C4A66", "#E2E6F0", "#F0F4FA"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False, "font.size": 9,
                     "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.edgecolor": "#9AA3B5", "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "axes.titlecolor": "#01037B"})


def latest(kind):
    found = sorted(RUNS.glob(f"*_{kind}/{kind}.json"))
    if not found:
        raise FileNotFoundError(f"runs/*_{kind}/{kind}.json 기록이 없습니다.")
    return json.loads(found[-1].read_text(encoding="utf-8")), found[-1].parent.name


def style(ax, ygrid=True):
    ax.spines[["top", "right"]].set_visible(False)
    if ygrid:
        ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name, source):
    fig.canvas.draw()                                            # 출처는 그림의 맨 아래(범례 밑)에 붙인다
    bottom = fig.get_tightbbox(fig.canvas.get_renderer()).y0 / fig.get_figheight()
    fig.text(0.01, bottom - 0.01, f"Source: {source}. Naver Pay Securities research summaries, KRX prices. "
             "Validation period 2024-07-01 to 2026-08-11.", fontsize=7, color="#9AA3B5", va="top")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return OUT / name


def titles(ax, title, sub):
    """제목과 그 아래 한 줄 설명. 결론 문장은 코드에 박지 않고 기록의 숫자로만 쓴다."""
    ax.set_title(title, pad=26 if sub.count("\n") else 16)
    ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=7.5, va="bottom", color=INK)


def legend_below(ax, ncol):
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -0.2), ncol=ncol)


def label_end(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8, color=color)


def fig1_car(h2, src):
    """그림 1 (H2): 등급별 진입 후 누적 초과수익, 중립 참고선, 20거래일 지점."""
    cv, m, aux = h2["curves"], h2["main"], h2["aux_days"]
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    style(ax)
    for key, label, color, ls in [("A", "A (tone warmed)", A, "-"), ("C", "C (tone cooled)", C, "-"),
                                  ("N", "Explicit neutral reports (reference)", N, "--")]:
        y = cv[key]
        ax.plot(range(1, len(y) + 1), y, color=color, lw=2, ls=ls, label=label)
        label_end(ax, len(y), y[-1], f"{y[-1]:+.2%}", color)
    ax.axhline(0, color="#9AA3B5", lw=0.8)
    for d in (20, 60):
        ax.axvline(d, color=GRID, lw=1, zorder=0)
    ax.set_xlim(1, 66)
    ax.set_xlabel("Trading days after entry (next close after publication)")
    ax.set_ylabel("Mean cumulative abnormal return")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
    legend_below(ax, 3)
    verdict = "supported" if h2["verdict"] == "지지" else "not supported"
    titles(ax, "H2: Cumulative abnormal return after A vs C reports",
           f"Day 60: C − A = {m['diff']:+.2%}, permutation p = {m['p']:.2f} ({verdict}).   "
           f"Day 20 (auxiliary): C − A = {aux['diff']:+.2%}, p = {aux['p']:.2f}.\n"
           f"A {h2['counts']['A']['kept']}, C {h2['counts']['C']['kept']} reports after deduplication; "
           "each day winsorized at 1% across A and C.")
    return save(fig, "fig1_h2_car.png", src)


def fig2_h1(h1, src):
    """그림 2 (H1): 등급별 다음 리포트의 하향·유지·상향 비율과 Wilson 95% 구간."""
    sm = h1["summary"]
    kinds = [("down", "Cut target or opinion"), ("same", "Kept both"), ("up", "Raised (no cut)")]
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    style(ax)
    for j, (g, color) in enumerate([("A", A), ("C", C)]):
        xs = np.arange(3) + (j - 0.5) * 0.36
        ys = [sm[g][k] for k, _ in kinds]
        err = [[y - sm[g][k + "_ci"][0] for y, (k, _) in zip(ys, kinds)],
               [sm[g][k + "_ci"][1] - y for y, (k, _) in zip(ys, kinds)]]
        ax.bar(xs, ys, width=0.34, color=color, yerr=err, ecolor=INK, capsize=3,
               label=f"{g}  (n = {sm[g]['점수 매김']}, mean score {sm[g]['mean']:+.3f})")
        for x, y in zip(xs, ys):
            ax.text(x, 0.012, f"{y:.1%}", ha="center", fontsize=8, color="white", fontweight="bold")
    ax.set_xticks(range(3), [n for _, n in kinds])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Share of next reports (same broker, within 3 months)")
    legend_below(ax, 2)
    lr = h1.get("locked_rule")
    verdict = "supported" if h1["verdict"] == "지지" else "not supported"
    note = f"\nSensitivity (locked missing-value rule): C − A = {lr['diff']:+.3f}, p = {lr['p']:.2f}." if lr else ""
    titles(ax, "H1: How the same broker's next report changed (score −1 / 0 / +1)",
           f"Mean score C − A = {h1['diff']:+.3f}, permutation p = {h1['p']:.2f} ({verdict}). Error bars: Wilson 95%.{note}")
    return save(fig, "fig2_h1_next_report.png", src)


def fig3_permutation(h1, h2, src):
    """그림 3 (H1·H2): 같은 달·같은 업종 안에서 라벨을 1,000번 섞었을 때의 차이와 실제 차이."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, sh, actual, p, title, fmt in [
            (axes[0], h1["shuffled"], h1["diff"], h1["p"], "H1: mean next-report score, C − A", "{:+.3f}"),
            (axes[1], h2["shuffled"], h2["main"]["diff"], h2["main"]["p"], "H2: 60-day CAR, C − A", "{:+.2%}")]:
        style(ax)
        ax.hist(sh, bins=40, color="#C9D6EA", edgecolor="white")
        ax.axvline(actual, color=C, lw=2)
        ax.axvline(np.median(sh), color=N, lw=1, ls="--")
        top = ax.get_ylim()[1]
        ax.annotate(f"actual {fmt.format(actual)}\np = {p:.2f}", (actual, top * 0.9), xytext=(6, 0),
                    textcoords="offset points", color=C, fontsize=8)
        ax.annotate(f"shuffled median\n{fmt.format(np.median(sh))}", (np.median(sh), top * 0.65), xytext=(-6, 0),
                    textcoords="offset points", ha="right", color=N, fontsize=7.5)
        ax.set_title(title)
        ax.set_xlabel("Difference with labels shuffled within month × sector (1,000 times)")
        if fmt.endswith("%}"):
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    axes[0].set_ylabel("Count")
    fig.suptitle("Permutation tests: actual difference vs 1,000 within-cell shuffles", x=0.01, ha="left",
                 fontsize=10, fontweight="bold", color="#01037B")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return save(fig, "fig3_permutation.png", src)


def fig4_backtest(bt, chk, cfg, src):
    """그림 4: 롱숏 백테스트 계좌(비용 전후, 공매도 금지 반영·무시), 낙폭, Long side·Short side."""
    base = bt["holds"][0]
    days = pd.to_datetime(bt["curves"][f"{base}|ban"]["date"])
    ban0 = max(pd.Timestamp(cfg["short_ban"]["start"]), days[0])
    ban1 = min(pd.Timestamp(cfg["short_ban"]["end"]), days[-1])
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.6, 2.4]})
    lines = [("ban", "gross", "Short ban enforced (executable), before costs", A, "-"),
             ("ban", "net", "Short ban enforced (executable), after costs", A, ":"),
             ("noban", "gross", "Short ban ignored (hypothetical), before costs", C, "-"),
             ("noban", "net", "Short ban ignored (hypothetical), after costs", C, ":")]
    for ax in axes:
        style(ax)
        ax.axvspan(ban0, ban1, color=BAN, zorder=0)
    for ban, kind, label, color, ls in lines:
        c = bt["curves"][f"{base}|{ban}"]
        x, eq = pd.to_datetime(c["date"]), pd.Series(c[kind])
        axes[0].plot(x, eq - 1, color=color, ls=ls, lw=1.8, label=label)
        label_end(axes[0], x[-1], eq.iloc[-1] - 1, f"{eq.iloc[-1] - 1:+.1%}", color)
        if kind == "net":
            axes[1].plot(x, eq / eq.cummax() - 1, color=color, ls=ls, lw=1.5)
    c = bt["curves"][f"{base}|noban"]
    x = pd.to_datetime(c["date"])
    for key, label, color in [("long_side", "Long side: A − market average", A), ("short_side", "Short side: market average − C", C)]:
        axes[2].plot(x, c[key], color=color, lw=1.8, label=label)
        label_end(axes[2], x[-1], c[key][-1], f"{c[key][-1]:+.1%}", color)
    for ax in (axes[0], axes[2]):
        ax.axhline(0, color="#9AA3B5", lw=0.8)
    b = cfg["backtest"]
    titles(axes[0], f"Backtest: long A / short C, {b['long_weight']:.0%} each, {base}-day holding",
           f"Shaded: short-selling ban (shorts off in the executable version). Round-trip cost "
           f"{b['cost_roundtrip']:.1%}, borrow {b['borrow_rate_annual']:.0%}/yr. Not used for the verdict.")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
    axes[0].set_ylabel("Cumulative return")
    axes[1].set_ylabel("Drawdown (after costs)")
    axes[2].set_ylabel("Cumulative daily sum")
    axes[2].legend(frameon=False, fontsize=7.5, loc="upper left")
    runs = pd.DataFrame(chk["runs"])
    titles(axes[2], "Long side and Short side (signal-based, before costs)",
           f"Random-label baseline ({len(runs)} runs): Long side {runs['long_side'].mean():+.1%}, Short side "
           f"{runs['short_side'].mean():+.1%} on average, because the sample stocks as a group differed from the market.")
    for ax in axes:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    fig.tight_layout()
    return save(fig, "fig4_backtest.png", src)


def fig5_checks(h2, src):
    """그림 5 (점검, 판정에 쓰지 않음): 직전 20거래일 초과수익, 공매도 금지·가능 기간 곡선."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), gridspec_kw={"width_ratios": [1, 2]})
    ax = axes[0]
    style(ax)
    ys = [h2["pre"]["A"], h2["pre"]["C"]]
    bars = ax.bar([0, 1], ys, color=[A, C], width=0.6)
    ax.bar_label(bars, labels=[f"{y:+.2%}" for y in ys], padding=3, fontsize=8)
    ax.margins(y=0.2)
    ax.axhline(0, color="#9AA3B5", lw=0.8)
    ax.set_xticks([0, 1], ["A", "C"])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
    ax.set_title("Before publication (20 trading days)", pad=10)
    ax.set_ylabel("Cumulative abnormal return")
    ax = axes[1]
    style(ax)
    cv = h2["curves"]
    for key, label, color, ls in [("ban_A", "A, published during short ban", A, "-"), ("ban_C", "C, during short ban", C, "-"),
                                  ("after_A", "A, after ban", A, ":"), ("after_C", "C, after ban", C, ":")]:
        y = cv[key]
        ax.plot(range(1, len(y) + 1), y, color=color, ls=ls, lw=1.8, label=label)
        label_end(ax, len(y), y[-1], f"{y[-1]:+.1%}", color)
    ax.axhline(0, color="#9AA3B5", lw=0.8)
    ax.set_xlim(1, 68)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
    ax.set_xlabel("Trading days after entry")
    legend_below(ax, 2)
    ax.set_title("After entry, split by short-ban period (not stratified, not tested)", pad=10)
    fig.suptitle("Checks (not used for the verdict)", x=0.01, ha="left", fontsize=10, fontweight="bold", color="#01037B")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return save(fig, "fig5_checks.png", src)


if __name__ == "__main__":
    cfg = load_config()
    require_lock()                                               # 결과 그림은 고정 뒤에만 (18장)
    h1, s1 = latest("h1")
    h2, s2 = latest("h2")
    bt, s3 = latest("backtest")
    chk, s4 = latest("engine_check")
    for path in [fig1_car(h2, f"runs/{s2}"), fig2_h1(h1, f"runs/{s1}"), fig3_permutation(h1, h2, f"runs/{s1}, runs/{s2}"),
                 fig4_backtest(bt, chk, cfg, f"runs/{s3}, runs/{s4}"), fig5_checks(h2, f"runs/{s2}")]:
        print("그림:", path)
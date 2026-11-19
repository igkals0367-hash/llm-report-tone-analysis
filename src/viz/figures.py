"""Step 12 결과 그림 5종과 추가 분석 그림 1종 (10장). runs/의 가장 최근 H1·H2·백테스트 기록만 읽어 그린다.
결과를 새로 계산하지 않으므로 그림의 숫자가 기록과 어긋날 수 없다. 고정 뒤에만 돈다 (18장).
v2(text_source: pdf)는 runs/*_h1_pdf 등 v2 기록을 읽고 파일 이름에 _pdf, 제목에 v2 표시를 붙인다 (19장 Step 12).
사용: python -m src.viz.figures  →  reports/figures/fig1~fig6 *.png (v2는 *_pdf.png)"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from src.analysis.guard import CUTOFFS, analysis_mode, require_lock
from src.config import load_config, source_path

RUNS = pathlib.Path("runs")
OUT = pathlib.Path("reports/figures")
A, C, N = "#4C72B0", "#C44E52", "#7B7485"                       # 부드러운 파랑·빨강, 중립은 회색
INK, GRID, BAN, WIND = "#3C4A66", "#E2E6F0", "#EEF2F8", "#F7F1E6"   # 음영: 공매도 금지, 청산 구간
LONG, SHORT = "#4DAF6E", "#B2373D"                                    # 백테스트의 롱(A)·숏(C): 초록·빨강, 밝기 차이로 색약에도 구분
VERSION = {"ban": ("Short ban enforced (executable)", "#3A6EA5"),      # 백테스트 버전은 롱·숏 색과 겹치지 않게
           "noban": ("Short ban ignored (hypothetical)", "#D98A2B")}
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False, "font.size": 9,
                     "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.edgecolor": "#9AA3B5", "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "axes.titlecolor": "#01037B"})
SRC = {"tag": "", "title": "", "text": "research summaries"}         # 글 출처별 이름표. main에서 set_source로 정한다


def set_source(tag):
    """v1이면 그대로, v2(tag "_pdf")면 파일 이름에 _pdf, 제목 앞에 v2 표시, 출처 문구를 PDF 본문으로 (19장)."""
    pdf = tag == "_pdf"
    SRC.update(tag=tag, title="v2 (PDF body) · " if pdf else "",
               text="research report titles and PDF bodies" if pdf else "research summaries")


def latest(kind, tag=""):
    found = sorted(RUNS.glob(f"*_{kind}{tag}/{kind}.json"))
    if not found:
        raise FileNotFoundError(f"runs/*_{kind}{tag}/{kind}.json 기록이 없습니다.")
    return json.loads(found[-1].read_text(encoding="utf-8")), found[-1].parent.name


def style(ax, ygrid=True):
    ax.spines[["top", "right"]].set_visible(False)
    if ygrid:
        ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name, source):
    fig.canvas.draw()                                            # 출처는 그림의 맨 아래(범례 밑)에 붙인다
    bottom = fig.get_tightbbox(fig.canvas.get_renderer()).y0 / fig.get_figheight()
    fig.text(0.01, bottom - 0.01, f"Source: {source}. Naver Pay Securities {SRC['text']}, KRX prices. "
             "Validation period 2024-07-01 to 2026-08-11.", fontsize=7, color="#9AA3B5", va="top")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{pathlib.Path(name).stem}{SRC['tag']}.png"     # v1 그림을 덮어쓰지 않게 v2는 _pdf
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def verdict_text(rec, p):
    """판정 문구. v1 기록(alpha 없음, 0.05)은 예전 그대로, v2는 판정 기준을 함께 적는다."""
    v = "supported" if rec["verdict"] == "지지" else "not supported"
    a = rec.get("alpha")
    return f"permutation p = {p:.2f} ({v})" if a in (None, 0.05) else f"permutation p = {p:.3f} ({v} at p < {a:g})"


def titles(ax, title, sub, top=True):
    """제목과 그 아래 한 줄 설명. 결론 문장은 코드에 박지 않고 기록의 숫자로만 쓴다. v2 표시는 맨 위 제목에만(top)."""
    ax.set_title((SRC["title"] if top else "") + title, pad=26 if sub.count("\n") else 16)
    ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=7.5, va="bottom", color=INK)


def legend_below(ax, ncol):
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -0.2), ncol=ncol)


def label_end(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8, color=color)


def label_ends(ax, x, items, gap=0.08):
    """선 끝 숫자들이 겹치지 않게 위아래로 벌려 단다. items: [(y, 글자, 색)], gap: 축 높이 대비 최소 간격."""
    lo, hi = ax.get_ylim()
    step, placed = (hi - lo) * gap, []
    for y, text, color in sorted(items, key=lambda t: t[0]):
        y = max(y, placed[-1] + step) if placed else y
        placed.append(y)
        ax.annotate(text, (x, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8,
                    color=color, fontweight="bold")


def fig1_car(h2, src):
    """그림 1 (H2): 등급별 진입 후 누적 초과수익, 중립 참고선, 20거래일 지점."""
    cv, m, aux = h2["curves"], h2["main"], h2["aux_days"]
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    style(ax)
    for key, label, color, ls in [("A", "A (tone warmed)", A, "-"), ("C", "C (tone cooled)", C, "-"),
                                  ("N", "Explicit neutral reports (reference)", N, "-")]:
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
    titles(ax, "H2: Cumulative abnormal return after A vs C reports",
           f"Day 60: C − A = {m['diff']:+.2%}, {verdict_text(h2, m['p'])}.   "
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
    note = f"\nSensitivity (locked missing-value rule): C − A = {lr['diff']:+.3f}, p = {lr['p']:.2f}." if lr else ""
    titles(ax, "H1: How the same broker's next report changed (score −1 / 0 / +1)",
           f"Mean score C − A = {h1['diff']:+.3f}, {verdict_text(h1, h1['p'])}. Error bars: Wilson 95%.{note}")
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
    fig.suptitle(SRC["title"] + "Permutation tests: actual difference vs 1,000 within-cell shuffles", x=0.01, ha="left",
                 fontsize=10, fontweight="bold", color="#01037B")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return save(fig, "fig3_permutation.png", src)


def shade(ax, ban0, ban1, wind0, end, text=False, top=False):
    """공매도 금지 기간과 청산 구간(마지막 신규 진입 다음 날부터 끝까지)을 칠한다 (16장)."""
    ax.axvspan(ban0, ban1, color=BAN, zorder=0)
    if wind0 is not None:
        ax.axvspan(wind0, end, color=WIND, zorder=0)
    if text:
        for x0, x1, t in [(ban0, ban1, "Short-selling ban"), (wind0, end, "Wind-down")]:
            if x0 is not None:
                ax.text(x0 + (x1 - x0) / 2, 0.96 if top else 0.04, t, transform=ax.get_xaxis_transform(), ha="center",
                        va="top" if top else "bottom",
                        fontsize=7, color="#7B7485", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1})


def fig4_backtest(bt, chk, cfg, src):
    """그림 4: 시장(유니버스 평균), 비용 전·후 누적 수익(공매도 금지 반영·무시), Long side·Short side (10장, 16장)."""
    base, b = bt["holds"][0], cfg["backtest"]
    cv, sm = bt["curves"], bt["summary"]
    days = pd.to_datetime(cv[f"{base}|ban"]["date"])
    ban0 = max(pd.Timestamp(cfg["short_ban"]["start"]), days[0])
    ban1 = min(pd.Timestamp(cfg["short_ban"]["end"]), days[-1])
    w = sm[f"{base}|ban"].get("winddown_start")
    wind0 = pd.Timestamp(w) if w else None
    fig, axes = plt.subplots(4, 1, figsize=(8.5, 11), sharex=True, gridspec_kw={"height_ratios": [1.4, 2.2, 2.2, 2.2]})
    for i, ax in enumerate(axes):
        style(ax)
        shade(ax, ban0, ban1, wind0, days[-1], text=(i == 0))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))

    mk = cv["market"]                                            # 시장: 유니버스 평균 (참고)
    x, eq = pd.to_datetime(mk["date"]), pd.Series(mk["equity"]) - 1
    axes[0].plot(x, eq, color=N, lw=1.8)
    label_end(axes[0], x[-1], eq.iloc[-1], f"{eq.iloc[-1]:+.1%}", N)
    titles(axes[0], f"Backtest: long A / short C, {b['long_weight']:.0%} each, {base}-day holding",
           "Top: market (universe average, top 200 equal-weighted) for reference only; the long-short strategy is not a "
           "bet on market direction.\nShaded: short-selling ban (shorts off in the executable version) and wind-down "
           "(no new signals after the validation period, holdings shrink). Not used for the verdict.")
    axes[0].set_ylabel("Market")

    for ax, kind, title in [(axes[1], "gross", "Before costs"), (axes[2], "net", "After costs")]:
        items = []
        for ver, (label, color) in VERSION.items():
            c = cv[f"{base}|{ver}"]
            x, eq = pd.to_datetime(c["date"]), pd.Series(c[kind]) - 1
            ax.plot(x, eq, color=color, lw=1.8, label=label)
            items.append((eq.iloc[-1], f"{label.split('(')[1][:-1]} {eq.iloc[-1]:+.1%}", color))
        ax.axhline(0, color="#9AA3B5", lw=0.8)
        label_ends(ax, x[-1], items)
        ax.set_ylabel("Cumulative return")
        if kind == "gross":                                      # 범례는 위 칸에만, 그래프 바깥 오른쪽 위에 둔다
            ax.set_title(title, pad=8)
            ax.legend(frameon=False, fontsize=7.5, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)
        else:
            mdd = {v: sm[f"{base}|{v}"]["net"]["mdd"] for v in VERSION}
            titles(ax, f"{title} (round-trip {b['cost_roundtrip']:.1%}, borrow {b['borrow_rate_annual']:.0%}/yr)",
                   f"Max drawdown: executable {mdd['ban']:+.1%}, hypothetical {mdd['noban']:+.1%}.", top=False)

    c = cv[f"{base}|noban"]
    x = pd.to_datetime(c["date"])
    items = []
    for key, label, color, name in [("long_side", "Long side: A − market average", LONG, "Long"),
                                    ("short_side", "Short side: market average − C", SHORT, "Short")]:
        axes[3].plot(x, c[key], color=color, lw=1.8, label=label)
        items.append((c[key][-1], f"{name} {c[key][-1]:+.1%}", color))
    axes[3].axhline(0, color="#9AA3B5", lw=0.8)
    label_ends(axes[3], x[-1], items)
    axes[3].set_ylabel("Cumulative daily sum")
    axes[3].legend(frameon=False, fontsize=7.5, loc="best")
    runs = pd.DataFrame(chk["runs"])
    titles(axes[3], "Long side and Short side (signal-based, before costs)",
           f"Random-label baseline ({'v1 engine check, ' if SRC['tag'] else ''}{len(runs)} runs): Long side {runs['long_side'].mean():+.1%}, Short side "
           f"{runs['short_side'].mean():+.1%} on average, because the sample stocks as a group differed from the market.",
           top=False)
    fig.tight_layout(h_pad=1.2)
    return save(fig, "fig4_backtest.png", src)


def fig5_checks(h2, cfg, src):
    """그림 5 (점검, 판정에 쓰지 않음): 직전 20거래일 초과수익, 공매도 금지 기간과 그 뒤로 나눈 곡선 (두 칸)."""
    fig = plt.figure(figsize=(11, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.45, 1.45], wspace=0.35)
    ax = fig.add_subplot(gs[0])
    style(ax)
    ys = [h2["pre"]["A"], h2["pre"]["C"]]
    bars = ax.bar([0, 1], ys, color=[A, C], width=0.6)
    ax.bar_label(bars, labels=[f"{y:+.2%}" for y in ys], padding=3, fontsize=8)
    ax.margins(y=0.2)
    ax.axhline(0, color="#9AA3B5", lw=0.8)
    ax.set_xticks([0, 1], ["A", "C"])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
    ax.set_title("Before publication\n(20 trading days)", pad=10)
    ax.set_ylabel("Cumulative abnormal return")
    cv = h2["curves"]
    start, end = cfg["periods"]["validation_start"], cfg["short_ban"]["end"]
    first = fig.add_subplot(gs[1])
    second = fig.add_subplot(gs[2], sharey=first)
    for ax, pre, title in [(first, "ban", f"Published during the short-selling ban\n({start} to {end})"),
                           (second, "after", "Published after the ban")]:
        style(ax)
        items = []
        for g, color in [("A", A), ("C", C)]:
            y = cv[f"{pre}_{g}"]
            ax.plot(range(1, len(y) + 1), y, color=color, lw=1.8, label=g)
            items.append((y[-1], f"{g} {y[-1]:+.1%}", color))
        ax.axhline(0, color="#9AA3B5", lw=0.8)
        ax.set_xlim(1, 72)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
        ax.set_xlabel("Trading days after entry")
        ax.set_title(title, pad=10)
    for ax in (first, second):                                   # 두 칸이 같은 세로축을 쓴 뒤에 끝 숫자를 단다
        label_ends(ax, len(cv["ban_A"]), [(cv[f"{p}_{g}"][-1], f"{g} {cv[f'{p}_{g}'][-1]:+.1%}", col)
                                          for p in (["ban"] if ax is first else ["after"]) for g, col in [("A", A), ("C", C)]])
    first.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(SRC["title"] + "Checks (not used for the verdict)", x=0.01, y=1.06, ha="left", fontsize=10, fontweight="bold", color="#01037B")
    fig.text(0.01, 0.985, "Right two panels: cumulative abnormal return after entry, split by publication date. Reference only: "
             "labels not grouped by month × sector, no permutation test, smaller samples.", fontsize=7.5, color=INK, va="bottom")
    return save(fig, "fig5_checks.png", src)


def fig6_cap(bt, folder, cfg, src):
    """그림 6 (추가 분석, 결과를 본 뒤 추가): 한 종목 비중 상한 버전. 보유 종목 수, 비용 전·후 누적 수익,
    주 결과(상한 없음)의 실행 가능 버전을 회색 참고선으로 (10장, 16장)."""
    base, b = bt["holds"][0], cfg["backtest"]
    cap, cv, sm = bt["cap"], bt["curves"], bt["summary"]
    days = pd.to_datetime(cv[f"{base}|ban|cap"]["date"])
    ban0 = max(pd.Timestamp(cfg["short_ban"]["start"]), days[0])
    ban1 = min(pd.Timestamp(cfg["short_ban"]["end"]), days[-1])
    w = sm[f"{base}|ban|cap"].get("winddown_start")
    wind0 = pd.Timestamp(w) if w else None
    daily = pd.read_csv(RUNS / folder / "daily_noban_cap.csv", parse_dates=["date"])
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.6), sharex=True, gridspec_kw={"height_ratios": [1.6, 2.2, 2.2]})
    for i, ax in enumerate(axes):
        style(ax)
        shade(ax, ban0, ban1, wind0, days[-1], text=(i == 0))

    ax = axes[0]                                                 # 쪽마다 보유 종목 수 (신호 기준, 상한과 무관)
    for col, label, color in [("n_long", "Long side (A)", LONG), ("n_short", "Short side (C)", SHORT)]:
        ax.plot(daily["date"], daily[col], color=color, lw=1.5, label=label)
    limit = round(b["long_weight"] / cap)
    ax.axhline(limit, color=N, lw=1)
    ax.annotate(f"{limit} names: below this, the cap leaves part of the side in cash", (ban1, limit), xytext=(6, 3),
                textcoords="offset points", fontsize=7, color=N)                # 종목 수가 충분한 구간에 글자를 둔다
    ax.set_ylabel("Names held")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)                   # 위쪽에 범례 자리를 남긴다
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=2)
    head = (f"Per-stock cap of {cap:.0%} (pre-specified for v2)" if SRC["tag"] else
            f"Additional analysis (added after seeing the results): per-stock cap of {cap:.0%}")
    titles(ax, f"{head}, {base}-day holding",
           f"Each side still gets up to {b['long_weight']:.0%}, but no single stock above {cap:.0%} of the account. "
           "The main result (fig4) has no cap.\nTop: names held per side (short-side count includes the ban period, "
           "when the executable version holds no shorts). Shaded: short-selling ban and wind-down.")

    ref = ("ban", "No cap, executable (main result, fig4)", "#B8BECB")
    for ax, kind, title in [(axes[1], "gross", "Before costs"), (axes[2], "net", "After costs")]:
        items = []
        for ver, label, color, key in [(ref[0], ref[1], ref[2], f"{base}|ban")] + \
                [(v, f"{VERSION[v][0].split(' (')[1][:-1].capitalize()}, {cap:.0%} cap", VERSION[v][1], f"{base}|{v}|cap")
                 for v in VERSION]:
            c = cv[key]
            x, eq = pd.to_datetime(c["date"]), pd.Series(c[kind]) - 1
            ax.plot(x, eq, color=color, lw=1.4 if key == f"{base}|ban" else 1.8, label=label, zorder=1 if color == ref[2] else 2)
            name = "no cap" if key == f"{base}|ban" else VERSION[ver][0].split(" (")[1][:-1]
            items.append((eq.iloc[-1], f"{name} {eq.iloc[-1]:+.1%}", "#8C96AA" if color == ref[2] else color))
        ax.axhline(0, color="#9AA3B5", lw=0.8)
        label_ends(ax, x[-1], items)
        ax.set_ylabel("Cumulative return")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
        if kind == "gross":
            ax.set_title(title, pad=8)
            ax.legend(frameon=False, fontsize=7.5, loc="best")
        else:
            mdd = {v: sm[f"{base}|{v}|cap"]["net"]["mdd"] for v in VERSION}
            titles(ax, f"{title} (round-trip {b['cost_roundtrip']:.1%}, borrow {b['borrow_rate_annual']:.0%}/yr)",
                   f"Max drawdown with the cap: executable {mdd['ban']:+.1%}, hypothetical {mdd['noban']:+.1%} "
                   f"(no cap, executable: {sm[f'{base}|ban']['net']['mdd']:+.1%}).", top=False)
    fig.tight_layout(h_pad=1.2)
    return save(fig, "fig6_backtest_cap.png", src)


if __name__ == "__main__":
    cfg = load_config()
    tag = analysis_mode(cfg)["tag"]                              # v1 "", v2 "_pdf"
    set_source(tag)
    require_lock(source_path(CUTOFFS, cfg))                      # 결과 그림은 고정 뒤에만 (18장)
    h1, s1 = latest("h1", tag)
    h2, s2 = latest("h2", tag)
    bt, s3 = latest("backtest", tag)
    chk, s4 = latest("engine_check")                             # 엔진 검증은 v1 기록 그대로 (19장 백테스트)
    for path in [fig1_car(h2, f"runs/{s2}"), fig2_h1(h1, f"runs/{s1}"), fig3_permutation(h1, h2, f"runs/{s1}, runs/{s2}"),
                 fig4_backtest(bt, chk, cfg, f"runs/{s3}, runs/{s4}"), fig5_checks(h2, cfg, f"runs/{s2}"),
                 fig6_cap(bt, s3, cfg, f"runs/{s3}")]:
        print("그림:", path)
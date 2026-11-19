"""개발용 대시보드 (16장). data/를 읽기만 하고 data/dashboard.html 한 장을 만든다. 결과 계산에 쓰지 않는다.
원문(제목·요약)은 넣지 않고 건수와 분포만 보여 준다. 분류 고정(Step 8) 전에는 결과 데이터를 읽는 칸이 없다.
실행: python -m tools.dev_dashboard  →  data/dashboard.html을 브라우저로 연다."""
import base64
import datetime as dt
import html
import io
import json
import pathlib
import subprocess
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq
from matplotlib import font_manager

OUT = pathlib.Path("data/dashboard.html")
NAVER = pathlib.Path("data/raw/naver")
KRX = pathlib.Path("data/raw/krx/daily")
F = {k: pathlib.Path(v) for k, v in {
    "reports": "data/processed/reports.parquet",
    "universe": "data/processed/universe.parquet",
    "pairs": "data/processed/pairs.parquet",
    "funnel": "data/processed/pairs_funnel.csv",
    "sentences": "data/interim/sentences.parquet",
    "probs": "data/interim/sentence_probs.parquet",
    "tone": "data/processed/tone.parquet",
}.items()}

# KRISS 발표 자료의 색: 남색 띠·표 머리. 데이터 색은 결과 그림(src/viz/figures.py)과 같다
NAVY, BLUE, RED, SLATE = "#01037B", "#4C72B0", "#C44E52", "#3C4A66"   # BLUE = A, RED = C
LONG, SHORT, GRAY = "#4DAF6E", "#B2373D", "#9AA3B5"                   # 백테스트의 롱·숏, 참고선
VERSION = {"ban": ("금지 반영 (실행 가능)", "#3A6EA5"), "noban": ("금지 무시 (가정)", "#D98A2B")}
LINE, CARD, LIGHT, SURFACE = "#E2E6F0", "#F8F9FC", "#DCEDFF", "#FFFFFF"
BAN_SHADE, WIND_SHADE = "#EEF2F8", "#F7F1E6"                          # 음영: 공매도 금지 기간, 청산 구간
PERIOD_COLOR = {"baseline": BLUE, "validation": RED}
_fonts = {f.name for f in font_manager.fontManager.ttflist}
for _name in ["Pretendard", "Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"]:
    if _name in _fonts:                                          # 한글 글꼴 (Windows는 Malgun Gothic)
        plt.rcParams["font.family"] = _name
        break
plt.rcParams.update({"axes.unicode_minus": False,                # 한글 글꼴에서 − 기호가 깨지지 않게
                     "text.color": SLATE, "axes.labelcolor": SLATE, "xtick.color": SLATE, "ytick.color": SLATE,
                     "axes.edgecolor": "#9AA3B5", "axes.titlecolor": NAVY, "axes.titleweight": "bold"})


# ---------- 그림·표 도우미 ----------
def axes(w=8.0, h=3.0):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=LINE, lw=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def img(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def table(df):
    df = df.copy()
    for c in df.columns:                                         # 정수는 1,234처럼, 소수는 끝의 0을 떼고
        if pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].map({True: "✓", False: ""})
        elif pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].map("{:,}".format)
    df = df.reset_index().rename(columns={"index": ""})
    return '<div class="tw">' + df.to_html(classes="t", border=0, index=False,
                                          float_format=lambda x: f"{x:,.3f}".rstrip("0").rstrip(".")) + "</div>"


def monthly_bar(dates, title, by=None):
    """월별 건수 막대. by(기간 등)를 주면 색을 나눠 쌓는다."""
    m = pd.to_datetime(dates).dt.to_period("M")
    df = pd.crosstab(m, by if by is not None else "건수").sort_index()
    fig, ax = axes()
    x, bottom = range(len(df)), 0
    for col in df.columns:
        ax.bar(x, df[col], bottom=bottom, color=PERIOD_COLOR.get(col, BLUE), label=str(col),
               edgecolor=SURFACE, linewidth=0.5)
        bottom = bottom + df[col]
    step = max(1, len(df) // 8)
    ax.set_xticks(list(x)[::step], [str(p) for p in df.index[::step]], rotation=0, fontsize=8)
    ax.set_title(title, loc="left", fontsize=10)
    if by is not None:
        ax.legend(frameon=False, fontsize=8)
    return img(fig)


def hist(values, title, xlabel, bins=40, range_=None, vlines=()):
    fig, ax = axes()
    ax.hist(values, bins=bins, range=range_, color=BLUE, edgecolor=SURFACE, linewidth=0.8)
    for i, (x, label) in enumerate(vlines):                      # (위치, 이름): 기준선. 두 개면 첫째 이름은 왼쪽에
        left = i == 0 and len(vlines) > 1
        ax.axvline(x, color=RED, lw=1.5)
        ax.annotate(label, (x, ax.get_ylim()[1] * 0.92), xytext=(-5 if left else 5, 0), textcoords="offset points",
                    color=RED, fontsize=9, ha="right" if left else "left")
    ax.set_title(f"{title} (n={len(values):,})", loc="left", fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    return img(fig)


def barh(counts, title, top=15):
    s = counts.head(top)[::-1]
    fig, ax = axes(8, 0.28 * len(s) + 0.8)
    ax.barh([str(i) for i in s.index], s.values, color=BLUE)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=LINE, lw=0.8)
    for y, v in enumerate(s.values):
        ax.text(v, y, f" {v:,}", va="center", fontsize=8, color=SLATE)
    ax.set_title(title, loc="left", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    return img(fig)


def rows(path):
    return pq.ParquetFile(path).metadata.num_rows


def end_labels(ax, x, items, gap=0.08):
    """선 끝 숫자가 겹치지 않게 위아래로 벌려 단다. items: [(y, 글자, 색)]"""
    lo, hi = ax.get_ylim()
    placed = []
    for y, text, color in sorted(items, key=lambda t: t[0]):
        y = max(y, placed[-1] + (hi - lo) * gap) if placed else y
        placed.append(y)
        ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", fontsize=8, color=color,
                    va="center", fontweight="bold")


def shade(ax, ban, wind0, end):
    """공매도 금지 기간과 청산 구간(마지막 신규 진입 다음 날부터 끝까지)을 칠한다 (16장)."""
    ax.axvspan(*ban, color=BAN_SHADE, zorder=0)
    if wind0 is not None:
        ax.axvspan(wind0, end, color=WIND_SHADE, zorder=0)


# ---------- 칸: Step마다 하나. 읽을 파일이 없으면 None을 돌려 건너뛴다 ----------
def section_status():
    """파이프라인 현황: Step마다 출력 파일이 있는지, 몇 줄인지, 언제 만들었는지."""
    items = [("1", "data/raw/naver/*.jsonl", sorted(NAVER.glob("*.jsonl"))),
             ("1", "data/raw/krx/daily/*.parquet", sorted(KRX.glob("*.parquet")))]
    out = []
    for step, name, files in items:
        last = max((f.stat().st_mtime for f in files), default=None)
        out.append({"Step": step, "파일": name, "있음": bool(files), "줄 수": f"{len(files):,}개 파일",
                    "만든 때": dt.datetime.fromtimestamp(last).strftime("%m-%d %H:%M") if last else ""})
    steps = {"reports": "2", "universe": "3", "pairs": "3·6", "funnel": "3", "sentences": "4", "probs": "5", "tone": "5"}
    for k, path in F.items():
        ok = path.exists()
        n = (rows(path) if path.suffix == ".parquet" else len(pd.read_csv(path))) if ok else None
        out.append({"Step": steps[k], "파일": str(path).replace("\\", "/"), "있음": ok,
                    "줄 수": f"{n:,}" if ok else "",
                    "만든 때": dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%m-%d %H:%M") if ok else ""})
    return "현황", table(pd.DataFrame(out).set_index("Step"))


def section_step1():
    """Step 1 수집: 월별 네이버 리포트 수, 거래일마다 받은 종목 수."""
    files = sorted(NAVER.glob("*.jsonl"))
    if not files:
        return None
    naver = pd.Series({f.stem: sum(1 for _ in open(f, encoding="utf-8")) for f in files})
    fig, ax = axes()
    ax.bar(range(len(naver)), naver.values, color=BLUE)
    step = max(1, len(naver) // 8)
    ax.set_xticks(range(0, len(naver), step), naver.index[::step], fontsize=8)
    ax.set_title(f"네이버 리포트, 월별 (합계 {naver.sum():,}건)", loc="left", fontsize=10)
    parts = [img(fig)]
    krx = sorted(KRX.glob("*.parquet"))
    if krx:
        n = pd.Series({pd.Timestamp(f.stem): rows(f) for f in krx})
        fig, ax = axes()
        ax.plot(n.index, n.values, color=BLUE, lw=1.5)
        ax.set_title(f"KRX 거래일마다 받은 종목 수 ({len(n):,}거래일, {n.index[0].date()} ~ {n.index[-1].date()})",
                     loc="left", fontsize=10)
        parts.append(img(fig))
    return "Step 1 수집", "".join(parts)


def section_step2():
    """Step 2 리포트 정리: 표시별 건수, 월별 분포, 의견, 증권사."""
    if not F["reports"].exists():
        return None
    r = pd.read_parquet(F["reports"], columns=["write_date", "broker_name", "opinion_group", "has_opinion",
                                                "has_goal_price", "has_summary", "is_neutral_ref"])
    raw = sum(sum(1 for _ in open(f, encoding="utf-8")) for f in NAVER.glob("*.jsonl"))
    counts = pd.Series({"원본(네이버)": raw, "정리 후([AI] 제외)": len(r), "의견 있음": int(r["has_opinion"].sum()),
                        "목표가 있음": int(r["has_goal_price"].sum()), "요약 있음": int(r["has_summary"].sum()),
                        "매수": int((r["opinion_group"] == "buy").sum()), "중립 참고": int(r["is_neutral_ref"].sum())},
                       name="건수").to_frame()
    return "Step 2 리포트 정리", (table(counts)
                                + monthly_bar(r["write_date"], "정리된 리포트, 월별")
                                + barh(r["opinion_group"].value_counts(), "의견 묶음")
                                + barh(r["broker_name"].value_counts(), "증권사 상위 15곳"))


def section_step3():
    """Step 3 표본: 거르는 단계마다 남은 수, 월별·업종·증권사별 표본 쌍, 직전 리포트까지 걸린 날."""
    if not F["pairs"].exists():
        return None
    parts = []
    if F["funnel"].exists():
        f = pd.read_csv(F["funnel"], index_col="stage")
        parts.append(table(f))
        fig, ax = axes(8, 3.2)
        y = range(len(f))[::-1]
        for i, col in enumerate(["baseline", "validation"]):
            ax.barh([v + (0.2 if i == 0 else -0.2) for v in y], f[col], height=0.38,
                    color=PERIOD_COLOR[col], label=col)
        ax.set_yticks(list(y), f.index, fontsize=8)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=LINE, lw=0.8)
        ax.legend(frameon=False, fontsize=8)
        ax.set_title("거르는 단계마다 남은 리포트 수", loc="left", fontsize=10)
        parts.append(img(fig))
    p = pd.read_parquet(F["pairs"], columns=["write_date", "period", "sector", "broker_code", "gap_rp_days"])
    if F["reports"].exists():
        names = (pd.read_parquet(F["reports"], columns=["broker_code", "broker_name"])
                 .drop_duplicates("broker_code").set_index("broker_code")["broker_name"])
        p["broker"] = p["broker_code"].map(names).fillna(p["broker_code"])
    else:
        p["broker"] = p["broker_code"]
    parts.append(table(p["period"].value_counts().rename("표본 쌍").to_frame()))
    parts.append(monthly_bar(p["write_date"], "표본 쌍, R 게시 월별", by=p["period"]))
    parts.append(barh(p["sector"].value_counts(), "업종 큰 분류별 표본 쌍"))
    parts.append(barh(p["broker"].value_counts(), "증권사 상위 15곳 표본 쌍"))
    parts.append(hist(p["gap_rp_days"], "직전 리포트(P)까지 걸린 날", "일", bins=36, range_=(0, 186)))
    if F["universe"].exists():
        u = pd.read_parquet(F["universe"], columns=["date", "item_code"])
        parts.append(table(pd.Series({"거래일": u["date"].nunique(), "하루 종목 수(최소~최대)":
                                      f"{u.groupby('date').size().min()}~{u.groupby('date').size().max()}",
                                      "한 번이라도 들어온 종목": u["item_code"].nunique()},
                                     name="유니버스").to_frame()))
    return "Step 3 표본", "".join(parts)


def section_step4():
    """Step 4 전처리: 리포트당 문장 수."""
    if not F["sentences"].exists():
        return None
    s = pd.read_parquet(F["sentences"], columns=["nid", "sent_idx"])
    per = s.groupby("nid").size()
    return "Step 4 전처리", (table(per.describe().round(1).rename("리포트당 문장 수").to_frame())
                           + hist(per, "리포트당 문장 수", "문장", bins=int(per.max()), range_=(0.5, per.max() + 0.5)))


def section_step5():
    """Step 5 톤: 문장별 가장 큰 라벨 비율, 리포트 톤 분포."""
    if not F["tone"].exists():
        return None
    parts = []
    if F["probs"].exists():
        pr = pd.read_parquet(F["probs"], columns=["p_pos", "p_neu", "p_neg"])
        lab = pr.idxmax(axis=1).map({"p_pos": "긍정", "p_neu": "중립", "p_neg": "부정"}).value_counts()
        parts.append(barh(lab, f"문장별 가장 큰 라벨 (문장 {len(pr):,}개)"))
    t = pd.read_parquet(F["tone"], columns=["tone", "model_revision"])
    parts.append(table(t["tone"].describe().round(3).rename("리포트 톤").to_frame()))
    parts.append(hist(t["tone"], "리포트 톤 = 문장별 (긍정 - 부정)의 평균", "톤", range_=(-1, 1)))
    parts.append(f'<p class="note">모델 리비전: {html.escape(", ".join(t["model_revision"].unique()))}</p>')
    return "Step 5 톤 점수", "".join(parts)


def section_step6():
    """Step 6 톤 변화: 기준선 기간 분포 (7장 Step 6 확인). 결과 데이터는 읽지 않는다."""
    if not F["pairs"].exists():
        return None
    p = pd.read_parquet(F["pairs"])
    if "tone_change" not in p.columns:
        return None
    b = p.loc[p["period"] == "baseline", "tone_change"]
    stats = b.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(3)
    stats["정확히 0 비율"] = round((b == 0).mean(), 3)
    return "Step 6 톤 변화", (table(stats.rename("기준선 톤 변화").to_frame())
                            + hist(b, "기준선 톤 변화 = 톤(R) - 톤(P)", "톤 변화", range_=(-2, 2)))


def section_step7():
    """Step 7 고정 전 점검: 가장 최근 runs/*_precheck의 판정과 다음 리포트까지 걸린 날."""
    found = sorted(pathlib.Path("runs").glob("*_precheck/precheck.json"))
    if not found:
        return None
    r = json.loads(found[-1].read_text(encoding="utf-8"))
    w = r["h1_window_months"]
    t = pd.DataFrame({"값": [f"{r['h1_share_validation']:.1%}",
                            f"{r['broker_top_sum']['A']:.1%} / {r['broker_top_sum']['C']:.1%}",
                            " / ".join(f"{r['short_shares'][g]:.1%}" for g in "ABC")],
                      "판정": [f"{w}개월로 고정" if r["h1_window_ok"] else "기간을 넓힘",
                             "쏠림 있음" if r["broker_flag"] else "쏠림 없음",
                             "몰림 있음" if r["short_flag"] else "몰림 없음"]},
                     index=[f"검증 R 중 {w}개월 안에 다음 리포트", "A / C 상위 증권사 비중 합",
                            "짧은 요약 쌍 비율 A / B / C"])
    parts = [f'<p class="note">기록: {html.escape(str(found[-1].parent))} · 임시 컷오프 '
             f'{r["cutoffs"][0]:.4f} / {r["cutoffs"][1]:.4f} (Step 8에서 고정)</p>', table(t)]
    if F["pairs"].exists() and F["reports"].exists():
        from src.analysis.precheck import next_report_date             # 다음 리포트는 날짜만 읽는다
        p = pd.read_parquet(F["pairs"], columns=["r_nid", "write_date", "period"])
        rep = pd.read_parquet(F["reports"], columns=["nid", "write_date", "broker_code", "item_code"])
        gap = (next_report_date(p, rep) - p["write_date"]).dt.days[p["period"] == "validation"].dropna()
        parts.append(hist(gap, "검증 R: 다음 리포트까지 걸린 날", "일", bins=40, range_=(0, 400),
                          vlines=[(w * 30.4, f"{w}개월")]))
    return "Step 7 고정 전 점검", "".join(parts)


def section_step8():
    """Step 8 분류 고정: 잠긴 컷오프와 검증 표본의 등급 (고정 뒤에만 보인다)."""
    from src.analysis.guard import require_lock                    # 등급 파일은 고정 뒤에만 읽는다 (18장)
    try:
        rec = require_lock()
    except RuntimeError:
        return None
    lock_t = pd.Series({k: rec[k] for k in ["lower", "upper", "baseline_n", "h1_window_months", "locked_at",
                                            "git_commit", "model_revision"]}, name="cutoffs.json").astype(str)
    parts = [table(lock_t.to_frame())]
    g = pd.read_parquet("data/processed/grades.parquet")
    n = g["grade"].value_counts().reindex(["A", "B", "C"], fill_value=0)
    parts.append(table(pd.DataFrame({"쌍": n, "비율": (n / n.sum()).map("{:.1%}".format)}).rename_axis("등급")))
    cuts = [(rec["lower"], f"lower {rec['lower']:.4f}"), (rec["upper"], f"upper {rec['upper']:.4f}")]
    parts.append(hist(g["tone_change"], "검증 톤 변화와 잠긴 컷오프", "톤 변화", range_=(-2, 2), vlines=cuts))
    return "Step 8 분류 고정", "".join(parts)


def section_step9():
    """Step 9 H1: 다음 리포트의 하향·유지·상향 비율과 라벨 섞기 검정 (가장 최근 runs/*_h1)."""
    from src.analysis.guard import require_lock                    # 결과는 고정 뒤에만 (18장)
    try:
        require_lock()
    except RuntimeError:
        return None
    found = sorted(pathlib.Path("runs").glob("*_h1/h1.json"))
    if not found:
        return None
    r = json.loads(found[-1].read_text(encoding="utf-8"))
    sm = r["summary"]
    rows = {g: {"R": sm[g]["R"], "점수 매김": sm[g]["점수 매김"], "다음 리포트 없음": sm[g]["no_next"],
                "기준가 조정": sm[g]["adjusted"], "비교 불가": sm[g]["not_comparable"],
                "하향": f"{sm[g]['down']:.1%}", "유지": f"{sm[g]['same']:.1%}", "상향": f"{sm[g]['up']:.1%}",
                "평균 점수": f"{sm[g]['mean']:+.4f}"} for g in ["A", "C"]}
    parts = [f'<p class="note">기록: {html.escape(str(found[-1].parent))}</p>',
             table(pd.DataFrame(rows).T.rename_axis("등급")),
             table(pd.Series({"차이 (C − A)": f"{r['diff']:+.4f}", "p": f"{r['p']:.4f}",
                              "판정": f"H1 {r['verdict']}"}, name="H1").to_frame())]
    if "locked_rule" in r:                                       # 민감도: 잠근 원래 규칙 (16장)
        lr = r["locked_rule"]
        parts.append(table(pd.Series({"점수 매김 A / C": f"{lr['summary']['A']['점수 매김']:,} / "
                                                         f"{lr['summary']['C']['점수 매김']:,}",
                                      "차이 (C − A)": f"{lr['diff']:+.4f}", "p": f"{lr['p']:.4f}"},
                                     name="민감도: 잠근 원래 규칙 (판정에 안 씀)").to_frame()))
    fig, ax = axes(8, 3.2)                                       # 하향·유지·상향 비율과 Wilson 95% 구간
    kinds = [("down", "하향"), ("same", "유지"), ("up", "상향")]
    for j, (g, color) in enumerate([("A", BLUE), ("C", RED)]):
        xs = [i + (j - 0.5) * 0.36 for i in range(3)]
        ys = [sm[g][k] for k, _ in kinds]
        err = [[y - sm[g][k + "_ci"][0] for y, (k, _) in zip(ys, kinds)],
               [sm[g][k + "_ci"][1] - y for y, (k, _) in zip(ys, kinds)]]
        ax.bar(xs, ys, width=0.34, color=color, label=g, yerr=err, ecolor=SLATE, capsize=3)
    ax.set_xticks(range(3), [n for _, n in kinds])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("다음 리포트의 변화 (점수 매긴 리포트 중 비율, 95% 구간)", loc="left", fontsize=10)
    parts.append(img(fig))
    parts.append(hist(pd.Series(r["shuffled"]), "라벨을 섞었을 때의 차이 (C - A)", "차이", bins=40,
                      vlines=[(r["diff"], f"실제 {r['diff']:+.3f}")]))
    return "Step 9 H1", "".join(parts)


def section_step10():
    """Step 10 H2: 등급별 누적 초과수익 곡선과 라벨 섞기 검정 (가장 최근 runs/*_h2)."""
    from src.analysis.guard import require_lock                    # 결과는 고정 뒤에만 (18장)
    try:
        require_lock()
    except RuntimeError:
        return None
    found = sorted(pathlib.Path("runs").glob("*_h2/h2.json"))
    if not found:
        return None
    r = json.loads(found[-1].read_text(encoding="utf-8"))
    pct = "{:+.2%}".format
    rows = {name: {"A 평균": pct(x["A"]), "C 평균": pct(x["C"]), "차이 C − A": pct(x["diff"]), "p": f"{x['p']:.4f}"}
            for name, x in [("판정: 60거래일, 1% 누름", r["main"]), ("보조: 60거래일, 5% 누름", r["aux_winsor"]),
                            ("보조: 20거래일, 1% 누름", r["aux_days"])]}
    cn = r["counts"]
    parts = [f'<p class="note">기록: {html.escape(str(found[-1].parent))} · 판정: H2 {html.escape(r["verdict"])} · '
             f'중복 제거 뒤 A {cn["A"]["kept"]:,} · C {cn["C"]["kept"]:,} · 중립 참고 {cn["N"]["kept"]:,}</p>',
             table(pd.DataFrame(rows).T)]

    def curves(title, series, ylim=None):
        fig, ax = axes(8, 3.4)
        items, n = [], 0
        for key, label, color in series:
            y = r["curves"].get(key) or []
            if y:
                ax.plot(range(1, len(y) + 1), y, color=color, lw=2, label=label)
                items.append((y[-1], f"{label} {y[-1]:+.1%}", color))
                n = len(y)
        if ylim:
            ax.set_ylim(*ylim)
        end_labels(ax, n, items)
        ax.axhline(0, color=GRAY, lw=0.8)
        ax.set_xlim(1, 68)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
        ax.set_xlabel("진입 후 거래일", fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.set_title(title, loc="left", fontsize=10)
        return img(fig)

    parts.append(curves("평균 누적 초과수익 (날마다 1% 누름)",
                        [("A", "A", BLUE), ("C", "C", RED), ("N", "중립 참고", GRAY)]))
    split = [v for k in ["ban_A", "ban_C", "after_A", "after_C"] for v in (r["curves"].get(k) or [])]
    if split:                                                    # 점검 1: 두 칸이 같은 세로축을 쓴다
        pad = (max(split) - min(split)) * 0.1
        ylim = (min(split) - pad, max(split) + pad)
        parts.append(curves("점검 1: 게시일이 공매도 금지 기간인 리포트", [("ban_A", "A", BLUE), ("ban_C", "C", RED)], ylim))
        parts.append(curves("점검 1: 게시일이 금지가 풀린 뒤인 리포트", [("after_A", "A", BLUE), ("after_C", "C", RED)], ylim))
    parts.append(table(pd.Series({"A": pct(r["pre"]["A"]), "C": pct(r["pre"]["C"])},
                                 name="점검 2: 리포트 직전 20거래일 초과수익").to_frame()))
    parts.append(hist(pd.Series(r["shuffled"]), "라벨을 섞었을 때의 60거래일 차이 (C - A)", "차이", bins=40,
                      vlines=[(r["main"]["diff"], f"실제 {r['main']['diff']:+.2%}")]))
    return "Step 10 H2", "".join(parts)


def section_step11():
    """Step 11 백테스트: 주 결과(상한 없음), 추가 분석(한 종목 상한, 결과를 본 뒤 추가), 시장 참고 (가장 최근 runs/*_backtest). 판정에 안 씀."""
    from src.analysis.guard import require_lock                    # 결과는 고정 뒤에만 (18장)
    from src.config import load_config
    try:
        require_lock()
    except RuntimeError:
        return None
    parts = []
    for kind, name in [("engine_check", "엔진 검증 (주 결과)"), ("engine_check_cap", "엔진 검증 (상한, 추가 분석)")]:
        chk = sorted(pathlib.Path("runs").glob(f"*_{kind}/engine_check.json"))
        if chk:
            c = json.loads(chk[-1].read_text(encoding="utf-8"))
            parts.append(f'<p class="note">{name} {html.escape(chk[-1].parent.name)}: {"통과" if c["pass"] else "실패"} · '
                         f'무작위 라벨 A − C 평균 {c["mean_spread"]:+.2%} (표준오차 {c["se_spread"]:.2%})</p>')
    found = sorted(pathlib.Path("runs").glob("*_backtest/backtest.json"))
    if not found:
        return ("Step 11 백테스트", "".join(parts)) if parts else None
    r = json.loads(found[-1].read_text(encoding="utf-8"))
    pct, cap = "{:+.2%}".format, r.get("cap")
    hold_name = lambda h: f"{h}거래일" if h.isdigit() else "다음 리포트까지"
    parts.append(f'<p class="note">기록: {html.escape(str(found[-1].parent))}</p>')
    if "market" in r:
        m = r["market"]
        parts.append(f'<p class="note">시장 (유니버스 평균, 참고): 누적 {pct(m["total"])}, 샤프 {m["sharpe"]:.2f}, '
                     f'최대낙폭 {m["mdd"]:.1%}</p>')
    rows = {}
    for key, sm in r["summary"].items():
        h, ban, *capped = key.split("|")                         # 상한 버전의 키는 "60|ban|cap"
        label = f"{hold_name(h)} · {VERSION[ban][0]}" + (f" · 추가 분석: 상한 {cap:.0%}" if capped else "")
        rows[label] = {"누적 (비용 전)": pct(sm["gross"]["total"]), "누적 (비용 후)": pct(sm["net"]["total"]),
                       "샤프 (전/후)": f"{sm['gross']['sharpe']:.2f} / {sm['net']['sharpe']:.2f}",
                       "최대낙폭 (전/후)": f"{sm['gross']['mdd']:.1%} / {sm['net']['mdd']:.1%}",
                       "Long side": pct(sm["long_side"]), "Short side": pct(sm["short_side"]),
                       "청산 구간 시작": sm.get("winddown_start") or "-"}
    parts.append(table(pd.DataFrame(rows).T))

    cfg, base = load_config(), r["holds"][0]
    cv, sm = r["curves"], r["summary"]
    days = pd.to_datetime(cv[f"{base}|ban"]["date"])
    ban = (max(pd.Timestamp(cfg["short_ban"]["start"]), days[0]), min(pd.Timestamp(cfg["short_ban"]["end"]), days[-1]))
    w = sm[f"{base}|ban"].get("winddown_start")
    wind0 = pd.Timestamp(w) if w else None
    note = f" · 음영: 공매도 금지(파랑), 청산 구간(베이지)" if wind0 is not None else " · 음영: 공매도 금지"

    def chart(title, lines, h=3.2, ylabel=None):
        """lines: [(x, y, 이름, 색, 굵기, 끝 숫자 이름)]. 모두 실선."""
        fig, ax = axes(8, h)
        shade(ax, ban, wind0, days[-1])
        items = []
        for x, y, label, color, lw, tag in lines:
            ax.plot(x, y, color=color, lw=lw, label=label)
            if tag:
                items.append((y.iloc[-1], f"{tag} {y.iloc[-1]:+.1%}", color))
        ax.axhline(0, color=GRAY, lw=0.8)
        end_labels(ax, days[-1], items)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.legend(frameon=False, fontsize=8, loc="best")
        ax.set_title(title, loc="left", fontsize=10)
        return img(fig)

    def eq(key, kind):
        c = cv[key]
        return pd.to_datetime(c["date"]), pd.Series(c[kind]) - 1

    if "market" in cv:                                           # 시장은 크기가 달라 따로 그린다
        x, y = pd.to_datetime(cv["market"]["date"]), pd.Series(cv["market"]["equity"]) - 1
        parts.append(chart("시장: 유니버스 평균 누적 수익 (참고)" + note, [(x, y, "유니버스 평균", GRAY, 1.8, "시장")], h=2.6))
    for kind, title in [("gross", "비용 전"), ("net", "비용 후")]:
        lines = [(*eq(f"{base}|{v}", kind), VERSION[v][0], VERSION[v][1], 1.8, VERSION[v][0].split(" (")[1][:-1])
                 for v in VERSION]
        parts.append(chart(f"누적 수익 {title} (보유 {hold_name(base)}, 주 결과)", lines))
    if cap and f"{base}|ban|cap" in cv:                          # 추가 분석: 상한 버전과 주 결과(회색)
        lines = [(*eq(f"{base}|ban", "net"), "상한 없음, 실행 가능 (주 결과)", "#B8BECB", 1.4, "상한 없음")]
        lines += [(*eq(f"{base}|{v}|cap", "net"), f"{VERSION[v][0]}, 상한 {cap:.0%}", VERSION[v][1], 1.8,
                   VERSION[v][0].split(" (")[1][:-1]) for v in VERSION]
        parts.append(chart(f"추가 분석 (결과를 본 뒤 추가): 한 종목 상한 {cap:.0%}, 비용 후", lines))
        daily = found[-1].parent / "daily_noban_cap.csv"
        if daily.exists():                                       # 쪽마다 보유 종목 수 (신호 기준)
            d = pd.read_csv(daily, parse_dates=["date"])
            fig, ax = axes(8, 2.6)
            shade(ax, ban, wind0, days[-1])
            ax.plot(d["date"], d["n_long"], color=LONG, lw=1.5, label="롱 (A)")
            ax.plot(d["date"], d["n_short"], color=SHORT, lw=1.5, label="숏 (C)")
            ax.axhline(round(0.5 / cap), color=GRAY, lw=1)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
            ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
            ax.set_title(f"쪽마다 보유 종목 수 (회색 선 {round(0.5 / cap)}개 아래면 상한 때문에 일부가 현금)", loc="left", fontsize=10)
            parts.append(img(fig))
    c = cv[f"{base}|noban"]
    x = pd.to_datetime(c["date"])
    parts.append(chart("Long side와 Short side (일별 합, 신호 기준, 비용 전)",
                       [(x, pd.Series(c["long_side"]), "Long side (A - 시장 평균)", LONG, 1.8, "Long"),
                        (x, pd.Series(c["short_side"]), "Short side (시장 평균 - C)", SHORT, 1.8, "Short")]))
    return "Step 11 백테스트", "".join(parts)


def last_run(kind, base):
    """runs/*_<kind>/<base>.json 중 가장 최근 기록. 없으면 None."""
    found = sorted(pathlib.Path("runs").glob(f"*_{kind}/{base}.json"))
    return json.loads(found[-1].read_text(encoding="utf-8")) if found else None


def section_v2():
    """v2 결과: 제목 + PDF 본문으로 같은 질문을 다시 검정한 결과와 그림 (19장). v2 분류 고정 뒤에만 보인다."""
    from src.analysis.guard import require_lock                    # 결과는 고정 뒤에만 (18장)
    try:
        lock = require_lock(pathlib.Path("models/cutoffs_pdf.json"))
    except RuntimeError:
        return None
    parts = ['<p class="note">판정 기준: v1은 p &lt; 0.05, v2는 p &lt; 0.025 (같은 결과 데이터에 두 번째로 도전하므로 '
             '둘로 나눔). 같은 쌍 v1은 v2에 남은 쌍만으로 v1 등급을 다시 계산한 참고 비교로, 판정하지 않는다. '
             '구성 몫 = 같은 달·같은 업종 안에서 라벨을 섞은 차이의 중간값, 종목 몫 = 실제 차이 − 구성 몫.</p>']

    data = {}                                                      # v2 데이터 단계 숫자 (건수만)
    man = pathlib.Path("data/raw/pdf_manifest.csv")
    if man.exists():
        st = pd.read_csv(man).drop_duplicates("nid", keep="last")["status"].value_counts()
        data["PDF 상태"] = " · ".join(f"{k} {v:,}" for k, v in st.items())
    pp = pathlib.Path("data/processed/pairs_pdf.parquet")
    if pp.exists():
        per = pd.read_parquet(pp, columns=["period"])["period"].value_counts()
        data["남은 쌍 (기준선 / 검증)"] = f"{per.get('baseline', 0):,} / {per.get('validation', 0):,}"
    sp = pathlib.Path("data/interim/sentences_pdf.parquet")
    if sp.exists():
        data["정리한 문장"] = f"{rows(sp):,}"
    data["컷오프 (lower / upper)"] = f"{lock['lower']:+.4f} / {lock['upper']:+.4f}"
    parts.append(table(pd.Series(data, name="v2 데이터").to_frame()))

    cols = {"v1 (제목·요약)": "", "같은 쌍 v1": "_v1same", "v2 (제목·PDF 본문)": "_pdf"}
    out = {}
    for name, tag in cols.items():                                 # v1·같은 쌍 v1·v2 결과 비교
        col = {}
        for h, kind in [("H1", "h1"), ("H2", "h2")]:
            r = last_run(f"{kind}{tag}", kind)
            if r is None:
                continue
            m = r if h == "H1" else r["main"]
            mid = float(pd.Series(r["shuffled"]).median())
            fmt = "{:+.4f}" if h == "H1" else "{:+.2%}"
            col[f"{h} 차이 (C − A)"] = fmt.format(m["diff"])
            col[f"{h} 구성 몫"] = fmt.format(mid)
            col[f"{h} 종목 몫"] = fmt.format(m["diff"] - mid)
            col[f"{h} p"] = f"{m['p']:.4f}"
            col[f"{h} 판정"] = (f"{r['verdict']} (p < {r.get('alpha') or 0.05:g})" if r.get("verdict") else "판정 안 함")
        bt = last_run(f"backtest{tag}", "backtest") if tag != "_v1same" else None
        if bt is not None:
            s = bt["summary"][f"{bt['holds'][0]}|ban"]
            col["백테스트 60일 실행 가능 (비용 전 / 후)"] = f"{s['gross']['total']:+.2%} / {s['net']['total']:+.2%}"
        out[name] = col
    parts.append(table(pd.DataFrame(out).fillna("-")))

    figs = sorted(pathlib.Path("reports/figures").glob("fig*_pdf.png"))   # python -m src.viz.figures (v2)가 만든 그림
    if not figs:
        parts.append('<p class="note">v2 그림이 아직 없습니다 (python -m src.viz.figures).</p>')
    for f in figs:
        parts.append(f'<p class="note">{html.escape(f.name)}</p>'
                     f'<img src="data:image/png;base64,{base64.b64encode(f.read_bytes()).decode()}">')
    return "v2 PDF 본문 결과", "".join(parts)


SECTIONS = [section_status, section_step1, section_step2, section_step3, section_step4, section_step5, section_step6,
            section_step7, section_step8, section_step9, section_step10, section_step11, section_v2]


def kpis():
    """맨 위 숫자 카드: 있는 파일에서만 센다."""
    out = []
    raw = sorted(NAVER.glob("*.jsonl"))
    if raw:
        out.append((f"{sum(sum(1 for _ in open(f, encoding='utf-8')) for f in raw):,}", "수집한 리포트"))
    if F["pairs"].exists():
        per = pd.read_parquet(F["pairs"], columns=["period"])["period"].value_counts()
        out.append((f"{per.get('baseline', 0):,} / {per.get('validation', 0):,}", "표본 쌍 (기준선 / 검증)"))
    if F["sentences"].exists():
        out.append((f"{rows(F['sentences']):,}", "전처리한 문장"))
    if F["pairs"].exists():
        p = pd.read_parquet(F["pairs"])
        if "tone_change" in p.columns:
            out.append((f"{p.loc[p['period'] == 'baseline', 'tone_change'].mean():+.3f}", "기준선 톤 변화 평균"))
    return "".join(f'<div class="kpi"><b>{html.escape(v)}</b><span>{html.escape(k)}</span></div>' for v, k in out)


CSS = """
*{box-sizing:border-box}
body{margin:0;background:#fff;color:#3C4A66;font-family:Pretendard,'Malgun Gothic',Arial,sans-serif;}
header.top{background:#01037B;color:#fff;padding:20px 16px 18px;border-bottom:4px solid #DCEDFF;}
header.top div,main{max-width:900px;margin:0 auto;}
.kicker{font-size:11px;font-weight:700;color:#DCEDFF;letter-spacing:.02em;}
header.top h1{font-size:22px;margin:4px 0 0;font-weight:800;}
main{padding:0 16px 40px;}
.sub{font-style:italic;font-weight:700;color:#01037B;font-size:13px;margin:12px 0;}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:14px 0;}
.kpi{background:#F8F9FC;border:1px solid #E2E6F0;padding:12px 14px;}
.kpi b{display:block;color:#01037B;font-size:22px;font-weight:800;}
.kpi span{font-size:12px;}
html{scroll-behavior:smooth;}
nav{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 4px;position:sticky;top:0;z-index:10;background:#fff;
    padding:8px 0;border-bottom:1px solid #E2E6F0;}
@media (max-width:640px){nav{flex-wrap:nowrap;overflow-x:auto;}}
nav a{background:#DCEDFF;border:1px solid #01037B;color:#01037B;font-size:12px;font-weight:700;
      padding:5px 10px;text-decoration:none;}
nav a i{font-style:normal;font-weight:400;margin-right:6px;}
section{margin-top:28px;scroll-margin-top:96px;}
section > h2{background:#01037B;color:#fff;margin:0;padding:8px 14px 9px;font-size:16px;font-weight:800;
             border-bottom:3px solid #DCEDFF;}
section > h2 small{display:block;font-size:10px;font-weight:700;color:#DCEDFF;margin-bottom:2px;}
img{max-width:100%;display:block;margin:10px 0;}
.tw{overflow-x:auto;}
table.t{border-collapse:collapse;font-size:12px;margin:10px 0;min-width:240px;}
table.t th{background:#01037B;color:#fff;font-weight:700;padding:5px 12px;text-align:right;}
table.t td{padding:4px 12px;border-bottom:1px solid #E2E6F0;text-align:right;}
table.t tr:nth-child(even) td{background:#F8F9FC;}
table.t td:first-child,table.t th:first-child{text-align:left;}
table.t td:first-child{color:#01037B;font-weight:700;}
.note{font-size:12px;color:#5A6478;}
.err{background:#FFF1EC;border-left:4px solid #C44E52;color:#C44E52;white-space:pre-wrap;font-size:12px;padding:10px;}
footer{max-width:900px;margin:30px auto 0;font-size:11px;color:#9AA3B5;padding:0 16px 20px;}
"""


def build():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    blocks, nav = [], []
    for i, fn in enumerate(SECTIONS):
        try:
            got = fn()
        except Exception:                                    # 칸 하나가 깨져도 나머지는 보이게
            got = (fn.__name__, f'<p class="err">{html.escape(traceback.format_exc())}</p>')
        if got is None:
            continue
        title, body = got
        kicker, _, name = title.partition(" ") if title.startswith("Step") else ("Overview", "", title)
        if title.startswith("Step"):
            num, _, name = name.partition(" ")
            kicker = f"Step {num}"
        if title.startswith("v2 "):                                 # v2 칸 (19장)
            kicker, name = "v2", title[3:]
        sub = (fn.__doc__ or "").split("\n")[0].split(":", 1)[-1].strip()
        nav.append(f'<a href="#s{i}"><i>{html.escape(kicker)}</i>{html.escape(name)}</a>')
        blocks.append(f'<section id="s{i}"><h2><small>{html.escape(kicker)}</small>{html.escape(name)}</h2>'
                      f'<p class="sub">{html.escape(sub)}</p>{body}</section>')
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    page = (f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>개발 대시보드</title>'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><style>{CSS}</style></head><body>'
            f'<header class="top"><div><span class="kicker">KRISS Quant · 개발 대시보드</span>'
            f'<h1>증권사 리포트 숨은 매도 의견 탐지</h1></div></header>'
            f'<main><p class="sub">만든 때 {now} · 커밋 {commit or "?"} · 원문 없음, 건수와 분포만</p>'
            f'<div class="kpis">{kpis()}</div><nav>{"".join(nav)}</nav>{"".join(blocks)}</main>'
            f'<footer>tools/dev_dashboard.py · data/를 읽기만 함 · 분류 고정 전에는 결과 데이터 칸 없음</footer>'
            f'</body></html>')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    return OUT


if __name__ == "__main__":
    print("대시보드:", build().resolve())
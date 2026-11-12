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

# KRISS 발표 자료의 색: 남색 띠·표 머리, 파란색 데이터, 주황색은 두 번째 계열과 경고
NAVY, BLUE, ORANGE, SLATE = "#01037B", "#006DCE", "#C0562B", "#3C4A66"
LINE, CARD, LIGHT, SURFACE = "#E2E6F0", "#F8F9FC", "#DCEDFF", "#FFFFFF"
PERIOD_COLOR = {"baseline": BLUE, "validation": ORANGE}
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
        ax.axvline(x, color=ORANGE, lw=1.5, ls="--")
        ax.annotate(label, (x, ax.get_ylim()[1] * 0.92), xytext=(-5 if left else 5, 0), textcoords="offset points",
                    color=ORANGE, fontsize=9, ha="right" if left else "left")
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
    fig, ax = axes(8, 3.2)                                       # 하향·유지·상향 비율과 Wilson 95% 구간
    kinds = [("down", "하향"), ("same", "유지"), ("up", "상향")]
    for j, (g, color) in enumerate([("A", BLUE), ("C", ORANGE)]):
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


SECTIONS = [section_status, section_step1, section_step2, section_step3, section_step4, section_step5, section_step6,
            section_step7, section_step8, section_step9]


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
nav{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 4px;}
nav a{background:#DCEDFF;border:1px solid #01037B;color:#01037B;font-size:12px;font-weight:700;
      padding:5px 10px;text-decoration:none;}
nav a i{font-style:normal;font-weight:400;margin-right:6px;}
section{margin-top:28px;}
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
.err{background:#FFF1EC;border-left:4px solid #C0562B;color:#C0562B;white-space:pre-wrap;font-size:12px;padding:10px;}
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
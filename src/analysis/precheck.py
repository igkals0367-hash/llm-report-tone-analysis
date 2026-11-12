"""Step 7 고정 전 점검 (8장). 결과(다음 리포트의 목표가 방향, 수익률)는 읽지 않는다.
쓰는 정보는 18장이 정한 것뿐이다: 게시일, 의견, 목표가, 증권사, 문장 수, 톤 변화, 다음 리포트까지 걸린 날 수."""
import datetime as dt
import json
import pathlib
import shutil
import subprocess

import pandas as pd

from src.classify.grade import assign_grade, fit_cutoffs
from src.config import load_config

PAIRS = pathlib.Path("data/processed/pairs.parquet")
REPORTS = pathlib.Path("data/processed/reports.parquet")
TONE = pathlib.Path("data/processed/tone.parquet")
CONFIG = pathlib.Path("configs/default.yaml")
RUNS = pathlib.Path("runs")


def next_report_date(pairs, reports):
    """표본 R마다 같은 증권사가 같은 종목에 R 다음에 낸 리포트의 게시일 (없으면 NaT).
    다음 리포트는 날짜만 읽는다 (18장 분류 고정 장치). Step 9도 같은 순서 정의를 쓴다."""
    r = reports[["nid", "write_date", "broker_code", "item_code"]].sort_values(["write_date", "nid"])
    nxt = r.groupby(["broker_code", "item_code"])["write_date"].shift(-1)
    return pairs["r_nid"].map(pd.Series(nxt.values, index=r["nid"].values))


def within_window(r_date, n_date, months):
    """다음 리포트가 R 게시일 + months개월(달력 기준, 같은 날 포함) 안에 있나 (8장 Step 7, 9장 Step 9)."""
    return n_date.notna() & (n_date <= r_date + pd.DateOffset(months=months))


def broker_top_shares(grades, brokers, top_k):
    """등급별로 비중이 큰 증권사 top_k곳과 그 비중."""
    return {g: brokers[grades == g].value_counts(normalize=True).head(top_k) for g in ["A", "B", "C"]}


def short_shares(grades, n_r, n_p, max_sentences):
    """등급별 '아주 짧은 요약' 쌍(R이나 P의 문장 수가 max_sentences 이하)의 비율."""
    short = (n_r <= max_sentences) | (n_p <= max_sentences)
    return {g: float(short[grades == g].mean()) for g in ["A", "B", "C"]}


def md_table(df):
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for i, row in df.iterrows():
        lines.append("| " + " | ".join([str(i)] + [str(v) for v in row]) + " |")
    return "\n".join(lines)


def precheck(cfg):
    """네 점검을 계산해 runs/<실행 시각>_precheck/에 기록하고, 판정을 돌려준다 (8장 Step 7)."""
    pc = cfg["precheck"]
    pairs = pd.read_parquet(PAIRS)
    reports = pd.read_parquet(REPORTS, columns=["nid", "write_date", "broker_code", "broker_name", "item_code"])
    n_sent = pd.read_parquet(TONE, columns=["nid", "n_sentences"]).set_index("nid")["n_sentences"]
    names = reports.drop_duplicates("broker_code").set_index("broker_code")["broker_name"]
    out = []

    # ① H1 기간: 판정은 검증 기간 R
    n_date = next_report_date(pairs, reports)
    gap = (n_date - pairs["write_date"]).dt.days
    win = cfg["h1"]["window_months"]
    rows = {}
    for period in ["baseline", "validation"]:
        m = pairs["period"] == period
        rows[period] = {"표본 R": f"{m.sum():,}", "다음 리포트 있음": f"{n_date[m].notna().sum():,}",
                        f"{win}개월 안": f"{within_window(pairs['write_date'][m], n_date[m], win).mean():.1%}",
                        f"{pc['h1_alt_window_months']}개월 안":
                            f"{within_window(pairs['write_date'][m], n_date[m], pc['h1_alt_window_months']).mean():.1%}",
                        "걸린 날 중간값": f"{gap[m].median():.0f}"}
    h1_share = float(within_window(pairs["write_date"], n_date, win)[pairs["period"] == "validation"].mean())
    h1_ok = h1_share >= pc["h1_min_share"]
    out += ["## ① H1 기간", md_table(pd.DataFrame(rows).T.rename_axis("기간")),
            f"\n판정 (검증 R 기준): {win}개월 안 {h1_share:.1%} → "
            + (f"{pc['h1_min_share']:.0%} 이상이라 **{win}개월로 고정**" if h1_ok else
               f"{pc['h1_min_share']:.0%} 미만이라 **{pc['h1_alt_window_months']}개월로 넓히고 검증 끝을 한 달 당김**")]

    # ② 쏠림: 기준선 임시 등급 (Step 8과 같은 계산, 아직 고정 아님)
    base = pairs[pairs["period"] == "baseline"].reset_index(drop=True)
    cut = fit_cutoffs(base["tone_change"], cfg)
    grades = pd.Series(assign_grade(base["tone_change"], cut))
    brokers = base["broker_code"].map(names).fillna(base["broker_code"])
    top = broker_top_shares(grades, brokers, pc["broker_top_k"])
    bt = pd.DataFrame({g: [f"{n} {v:.1%}" for n, v in s.items()] + [f"{s.sum():.1%}"] for g, s in top.items()},
                      index=[f"{i + 1}위" for i in range(pc["broker_top_k"])] + ["합"]).rename_axis("")
    broker_flag = any(top[g].sum() > pc["broker_max_share"] for g in ["A", "C"])
    sh = short_shares(grades, base["r_nid"].map(n_sent), base["p_nid"].map(n_sent), pc["short_max_sentences"])
    ratio = {g: (sh[g] / sh["B"] if sh["B"] > 0 else float("inf")) for g in ["A", "C"]}
    short_flag = all(ratio[g] >= pc["short_ratio"] for g in ["A", "C"])
    st = pd.DataFrame({g: [f"{sh[g]:.1%}", f"{ratio[g]:.2f}" if g in ratio else "1"] for g in ["A", "B", "C"]},
                      index=["짧은 요약 쌍 비율", "B 대비 배수"]).rename_axis("")
    sizes = grades.value_counts().reindex(["A", "B", "C"]).to_dict()
    out += ["\n## ② 증권사·문장 수 쏠림 (기준선 임시 등급)",
            f"임시 컷오프 lower {cut[0]:.4f}, upper {cut[1]:.4f} | A {sizes['A']}, B {sizes['B']}, C {sizes['C']}",
            "\n### 증권사", md_table(bt),
            f"\n판정: A나 C에서 상위 {pc['broker_top_k']}곳 합이 {pc['broker_max_share']:.0%} 초과? → "
            + ("**쏠림 있음: 증권사별 표준화**" if broker_flag else "**쏠림 없음**"),
            f"\n### 문장 수 ('아주 짧은 요약' = R이나 P의 문장이 {pc['short_max_sentences']}개 이하)", md_table(st),
            f"\n판정: A와 C 각각 B의 {pc['short_ratio']}배 이상? → "
            + (f"**몰림 있음: R·P 모두 {pc['short_max_sentences'] + 1}문장 이상만 남김**" if short_flag else "**몰림 없음**")]

    # ③ 표본 수, ④ 전처리
    per = pairs["period"].value_counts()
    out += ["\n## ③ 표본 수",
            f"기준선 {per.get('baseline', 0):,}쌍 (어림 약 1,200), 검증 {per.get('validation', 0):,}쌍 (어림 약 2,800). "
            "검증이 적은 이유는 14장·15장에 기록.",
            "\n## ④ 전처리",
            "Step 4 확인에서 고친 것: 네이버 종목명 자동 추가, 'TP'는 단독일 때만, 문장 중간 줄바꿈 잇기, "
            "글자 없는 문장 버리기 (16장 '전처리 보완'). 무작위 30건 원문 대조 이상 없음."]

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = RUNS / f"{stamp}_precheck"
    folder.mkdir(parents=True, exist_ok=True)
    head = [f"# 고정 전 점검 (Step 7) {stamp}", f"커밋 {commit or '?'}{' (커밋 안 된 변경 있음)' if dirty else ''} · "
            f"모델 리비전 {cfg['model']['revision']}", ""]
    (folder / "precheck.md").write_text("\n".join(head + out) + "\n", encoding="utf-8")
    shutil.copy(CONFIG, folder / "config.yaml")
    result = {"h1_window_months": win, "h1_share_validation": h1_share, "h1_window_ok": h1_ok, "cutoffs": cut,
              "broker_flag": broker_flag, "short_flag": short_flag, "short_shares": sh,
              "broker_top_sum": {g: float(top[g].sum()) for g in "ABC"}, "commit": commit, "dirty": dirty}
    (folder / "precheck.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return folder, result


if __name__ == "__main__":
    folder, r = precheck(load_config())
    print((folder / "precheck.md").read_text(encoding="utf-8"))
    print("기록:", folder)
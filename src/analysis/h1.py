"""Step 9 H1 (9장): C 리포트 뒤에 다음 리포트의 목표가·의견이 A 뒤보다 더 나빠지는가.
분류 고정 뒤에만 돈다 (18장). 점수는 모델이 아니라 네이버의 목표가·의견 값으로만 매긴다.
v2(text_source: pdf)는 _pdf 등급·컷오프로 p < 0.025 판정, --v1same은 v1 등급으로 v2에 남은 쌍만 계산하는 참고 비교 (19장).
사용: python -m src.analysis.h1            (설정의 글 출처로 판정)
      python -m src.analysis.h1 --v1same   (같은 쌍 v1 비교, v2 고정 뒤에만)"""
import datetime as dt
import json
import math
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from src.analysis.guard import CUTOFFS, analysis_mode, require_lock
from src.analysis.permutation import permutation_test
from src.analysis.precheck import within_window
from src.config import load_config, source_path
from src.preprocess.pairs import adjusted_between, adjustment_days

PAIRS = pathlib.Path("data/processed/pairs.parquet")
REPORTS = pathlib.Path("data/processed/reports.parquet")
GRADES = pathlib.Path("data/processed/grades.parquet")
OUT = pathlib.Path("data/processed/h1.parquet")
CONFIG = pathlib.Path("configs/default.yaml")
RUNS = pathlib.Path("runs")


def next_report(r_nids, reports):
    """R마다 같은 증권사가 같은 종목에 R 다음에 낸 리포트의 nid와 게시일 (Step 7과 같은 순서 정의)."""
    r = reports[["nid", "write_date", "broker_code", "item_code"]].sort_values(["write_date", "nid"])
    g = r.groupby(["broker_code", "item_code"])
    nxt = pd.DataFrame({"n_nid": g["nid"].shift(-1).astype("Int64").to_numpy(),
                        "n_write_date": g["write_date"].shift(-1).to_numpy()}, index=r["nid"].to_numpy())
    return nxt.reindex(np.asarray(r_nids)).reset_index(drop=True)


def _f(x):
    return pd.Series(x).astype("float64").to_numpy()


def score_change(r_rank, r_tp, n_rank, n_tp, missing="known"):
    """9장 점수표. 하나라도 내림 −1, 내림 없이 하나라도 올림 +1, 둘 다 같음 0, 판단 불가 NaN.
    missing="known"(판정, 2026-11-13): 빠진 값이 있어도 아는 것만으로 판단한다. 빈값과의 비교는 거짓이라
    모르는 값은 내림으로도 올림으로도 세지 않는다.
    missing="exclude"(잠근 원래 규칙, 16장, 민감도 분석): N의 목표가나 의견이 하나라도 없으면 제외한다."""
    r_rank, r_tp, n_rank, n_tp = map(_f, (r_rank, r_tp, n_rank, n_tp))
    down = (n_rank < r_rank) | (n_tp < r_tp)
    up = (n_rank > r_rank) | (n_tp > r_tp)
    same = (n_rank == r_rank) & (n_tp == r_tp)
    score = np.select([down, up, same], [-1.0, 1.0, 0.0], default=np.nan)
    if missing == "exclude":
        score = np.where(np.isnan(n_rank) | np.isnan(n_tp), np.nan, score)
    return score


def next_report_scores(pairs, reports, cfg, keep=None):
    """검증 A·C 리포트마다 다음 리포트를 찾아 점수를 매긴다 (9장 Step 9). → 리포트별 표
    등급·컷오프는 cfg의 글 출처를 따른다(v2는 _pdf). keep을 주면 그 r_nid만 남긴다(같은 쌍 v1 비교, 19장)."""
    lock = require_lock(source_path(CUTOFFS, cfg))              # 분류 고정 전이면 여기서 멈춘다
    grades = pd.read_parquet(source_path(GRADES, cfg))
    if keep is not None:
        grades = grades[grades["r_nid"].isin(keep)]
    ac = (grades[grades["grade"].isin(["A", "C"])][["r_nid", "grade"]]
          .merge(pairs[["r_nid", "write_date", "item_code", "sector"]], on="r_nid", how="left")
          .reset_index(drop=True))
    ac = pd.concat([ac, next_report(ac["r_nid"], reports)], axis=1)

    info = reports.set_index("nid")[["opinion_rank", "goal_price"]]
    r_info = info.reindex(ac["r_nid"].to_numpy()).reset_index(drop=True)
    n_info = info.reindex(ac["n_nid"].to_numpy()).reset_index(drop=True)
    args = (r_info["opinion_rank"], r_info["goal_price"], n_info["opinion_rank"], n_info["goal_price"])
    score = score_change(*args)
    score_locked = score_change(*args, missing="exclude")

    in_win = within_window(ac["write_date"], ac["n_write_date"], lock["h1_window_months"]).to_numpy()
    adj = adjustment_days(cfg)
    adjusted = np.array([w and adjusted_between(adj, c, s, e) for w, c, s, e in
                         zip(in_win, ac["item_code"], ac["write_date"], ac["n_write_date"])], dtype=bool)
    status = np.select([~in_win, adjusted, np.isnan(score)], ["no_next", "adjusted", "not_comparable"], "scored")
    ac["score"] = np.where(status == "scored", score, np.nan)
    ac["score_locked"] = np.where(np.isin(status, ["no_next", "adjusted"]), np.nan, score_locked)
    ac["status"] = status
    return ac


def wilson(k, n, z=1.96):
    """비율 k/n의 95% 구간 (Wilson). 그림용이고 판정에는 쓰지 않는다 (9장)."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (mid - half, mid + half)


def summarize(h, col="score"):
    """등급별 건수, 하향·유지·상향 비율과 95% 구간, 평균 점수. col: score(판정) 또는 score_locked(민감도)."""
    out = {}
    for g in ["A", "C"]:
        x = h[h["grade"] == g]
        sc = x[col].dropna()
        n = len(sc)
        row = {"R": len(x), "점수 매김": n}
        for s in ["no_next", "adjusted"]:
            row[s] = int((x["status"] == s).sum())
        row["not_comparable"] = row["R"] - n - row["no_next"] - row["adjusted"]   # 빠진 값 등으로 판단 불가
        for name, v in [("down", -1), ("same", 0), ("up", 1)]:
            k = int((sc == v).sum())
            row[name] = k / n if n else math.nan
            row[name + "_ci"] = wilson(k, n)
        row["mean"] = float(sc.mean()) if n else math.nan
        out[g] = row
    return out


if __name__ == "__main__":
    mode = analysis_mode(load_config(), v1same="--v1same" in sys.argv[1:])
    cfg, tag, alpha = mode["cfg"], mode["tag"], mode["alpha"]  # v1same이면 cfg는 v1(summary) 설정
    cut_path = source_path(CUTOFFS, cfg)
    lock = require_lock(cut_path)
    pairs = pd.read_parquet(source_path(PAIRS, cfg))
    reports = pd.read_parquet(REPORTS, columns=["nid", "write_date", "broker_code", "item_code",
                                                "opinion_rank", "goal_price"])
    h = next_report_scores(pairs, reports, cfg, mode["keep"])
    out = OUT.with_name(f"{OUT.stem}{tag}{OUT.suffix}")         # h1.parquet, h1_pdf.parquet, h1_v1same.parquet
    out.parent.mkdir(parents=True, exist_ok=True)
    h[["r_nid", "grade", "n_nid", "n_write_date", "score", "status", "score_locked"]].to_parquet(out, index=False)
    pm = cfg["permutation"]

    def test(col):                                              # 같은 칸·같은 시드로 라벨 섞기
        s = h[h[col].notna()]
        strata = s["write_date"].dt.strftime("%Y-%m") + "|" + s["sector"].astype(str)   # 게시 연월 × 업종 큰 분류
        return permutation_test(s[col], s["grade"], strata, pm["n"], pm["seed"])

    diff, p, shuffled = test("score")                           # 판정
    verdict = None if alpha is None else ("지지" if p < alpha else "미지지")
    verdict_line = ("같은 쌍 v1 비교는 참고라 판정하지 않는다 (19장)" if alpha is None
                    else f"판정 (p < {alpha}): **H1 {verdict}**")
    summ = summarize(h)
    diff_l, p_l, _ = test("score_locked")                       # 민감도: 잠근 원래 규칙 (16장)
    summ_l = summarize(h, "score_locked")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = RUNS / f"{stamp}_h1{tag}"
    folder.mkdir(parents=True, exist_ok=True)
    pct = lambda r, k: f"{r[k]:.1%} [{r[k + '_ci'][0]:.1%}, {r[k + '_ci'][1]:.1%}]"
    lines = [f"# H1 (Step 9) {stamp}", mode["label"],
             f"커밋 {commit}{' (커밋 안 된 변경 있음)' if dirty else ''} · 고정 커밋 {lock['git_commit'][:7]} · "
             f"컷오프 {lock['lower']:.5f} / {lock['upper']:.5f} · 기간 {lock['h1_window_months']}개월", "",
             "| 등급 | R | 점수 매김 | 다음 리포트 없음 | 기준가 조정 | 비교 불가 | 하향 | 유지 | 상향 | 평균 점수 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for g, r in summ.items():
        lines.append(f"| {g} | {r['R']:,} | {r['점수 매김']:,} | {r['no_next']:,} | {r['adjusted']:,} | "
                     f"{r['not_comparable']:,} | {pct(r, 'down')} | {pct(r, 'same')} | {pct(r, 'up')} | {r['mean']:+.4f} |")
    lines += ["", f"차이 = C 평균 − A 평균 = **{diff:+.4f}**",
              f"라벨 섞기 {pm['n']:,}회 (칸: 게시 연월 × 업종, 시드 {pm['seed']}): p = **{p:.4f}** "
              f"(섞은 차이 중간값 {np.median(shuffled):+.4f}, 5% 지점 {np.quantile(shuffled, 0.05):+.4f})",
              verdict_line, "", "구간은 Wilson 95%이고 판정에 쓰지 않는다.", "",
              "## 민감도 분석: 잠근 원래 규칙 (16장, N의 목표가나 의견이 하나라도 없으면 제외, 판정에 쓰지 않음)",
              "| 등급 | 점수 매김 | 빠진 값 등으로 제외 | 평균 점수 |", "|---|---|---|---|"]
    for g, r in summ_l.items():
        lines.append(f"| {g} | {r['점수 매김']:,} | {r['not_comparable']:,} | {r['mean']:+.4f} |")
    lines += ["", f"차이 = **{diff_l:+.4f}**, p = **{p_l:.4f}**"]
    (folder / "h1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy(CONFIG, folder / "config.yaml")
    shutil.copy(cut_path, folder / cut_path.name)
    (folder / "h1.json").write_text(json.dumps(
        {"mode": tag or "_v1", "label": mode["label"], "alpha": alpha,
         "diff": diff, "p": p, "verdict": verdict, "summary": summ, "shuffled": shuffled.round(6).tolist(),
         "locked_rule": {"diff": diff_l, "p": p_l, "summary": summ_l},
         "commit": commit, "dirty": dirty, "n_perm": pm["n"], "seed": pm["seed"]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(lines))
    print("기록:", folder)
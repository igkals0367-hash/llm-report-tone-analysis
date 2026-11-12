"""Step 9 H1 (9장): C 리포트 뒤에 다음 리포트의 목표가·의견이 A 뒤보다 더 나빠지는가.
분류 고정 뒤에만 돈다 (18장). 점수는 모델이 아니라 네이버의 목표가·의견 값으로만 매긴다."""
import datetime as dt
import json
import math
import pathlib
import shutil
import subprocess

import numpy as np
import pandas as pd

from src.analysis.guard import CUTOFFS, require_lock
from src.analysis.permutation import permutation_test
from src.analysis.precheck import within_window
from src.config import load_config
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


def score_change(r_rank, r_tp, n_rank, n_tp):
    """9장 점수표와 빠진 값 규칙 (16장). 하나라도 내림 −1, 내림 없이 하나라도 올림 +1, 둘 다 같음 0, 판단 불가 NaN.
    빈값과의 비교는 거짓이라, 모르는 값은 내림으로도 올림으로도 세지 않는다."""
    r_rank, r_tp, n_rank, n_tp = map(_f, (r_rank, r_tp, n_rank, n_tp))
    down = (n_rank < r_rank) | (n_tp < r_tp)
    up = (n_rank > r_rank) | (n_tp > r_tp)
    same = (n_rank == r_rank) & (n_tp == r_tp)
    return np.select([down, up, same], [-1.0, 1.0, 0.0], default=np.nan)


def next_report_scores(pairs, reports, cfg):
    """검증 A·C 리포트마다 다음 리포트를 찾아 점수를 매긴다 (9장 Step 9). → 리포트별 표"""
    lock = require_lock()                                       # 분류 고정 전이면 여기서 멈춘다
    grades = pd.read_parquet(GRADES)
    ac = (grades[grades["grade"].isin(["A", "C"])][["r_nid", "grade"]]
          .merge(pairs[["r_nid", "write_date", "item_code", "sector"]], on="r_nid", how="left")
          .reset_index(drop=True))
    ac = pd.concat([ac, next_report(ac["r_nid"], reports)], axis=1)

    info = reports.set_index("nid")[["opinion_rank", "goal_price"]]
    r_info = info.reindex(ac["r_nid"].to_numpy()).reset_index(drop=True)
    n_info = info.reindex(ac["n_nid"].to_numpy()).reset_index(drop=True)
    score = score_change(r_info["opinion_rank"], r_info["goal_price"], n_info["opinion_rank"], n_info["goal_price"])

    in_win = within_window(ac["write_date"], ac["n_write_date"], lock["h1_window_months"]).to_numpy()
    adj = adjustment_days(cfg)
    adjusted = np.array([w and adjusted_between(adj, c, s, e) for w, c, s, e in
                         zip(in_win, ac["item_code"], ac["write_date"], ac["n_write_date"])], dtype=bool)
    status = np.select([~in_win, adjusted, np.isnan(score)], ["no_next", "adjusted", "not_comparable"], "scored")
    ac["score"] = np.where(status == "scored", score, np.nan)
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


def summarize(h):
    """등급별 건수, 하향·유지·상향 비율과 95% 구간, 평균 점수."""
    out = {}
    for g in ["A", "C"]:
        x = h[h["grade"] == g]
        sc = x.loc[x["status"] == "scored", "score"]
        n = len(sc)
        row = {"R": len(x), "점수 매김": n}
        for s in ["no_next", "adjusted", "not_comparable"]:
            row[s] = int((x["status"] == s).sum())
        for name, v in [("down", -1), ("same", 0), ("up", 1)]:
            k = int((sc == v).sum())
            row[name] = k / n if n else math.nan
            row[name + "_ci"] = wilson(k, n)
        row["mean"] = float(sc.mean()) if n else math.nan
        out[g] = row
    return out


if __name__ == "__main__":
    cfg = load_config()
    lock = require_lock()
    pairs = pd.read_parquet(PAIRS)
    reports = pd.read_parquet(REPORTS, columns=["nid", "write_date", "broker_code", "item_code",
                                                "opinion_rank", "goal_price"])
    h = next_report_scores(pairs, reports, cfg)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    h[["r_nid", "grade", "n_nid", "n_write_date", "score", "status"]].to_parquet(OUT, index=False)

    s = h[h["status"] == "scored"]
    strata = s["write_date"].dt.strftime("%Y-%m") + "|" + s["sector"].astype(str)   # 게시 연월 × 업종 큰 분류
    pm = cfg["permutation"]
    diff, p, shuffled = permutation_test(s["score"], s["grade"], strata, pm["n"], pm["seed"])
    verdict = "지지" if p < pm["alpha"] else "미지지"
    summ = summarize(h)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = RUNS / f"{stamp}_h1"
    folder.mkdir(parents=True, exist_ok=True)
    pct = lambda r, k: f"{r[k]:.1%} [{r[k + '_ci'][0]:.1%}, {r[k + '_ci'][1]:.1%}]"
    lines = [f"# H1 (Step 9) {stamp}",
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
              f"판정 (p < {pm['alpha']}): **H1 {verdict}**", "", "구간은 Wilson 95%이고 판정에 쓰지 않는다."]
    (folder / "h1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy(CONFIG, folder / "config.yaml")
    shutil.copy(CUTOFFS, folder / "cutoffs.json")
    (folder / "h1.json").write_text(json.dumps(
        {"diff": diff, "p": p, "verdict": verdict, "summary": summ, "shuffled": shuffled.round(6).tolist(),
         "commit": commit, "dirty": dirty, "n_perm": pm["n"], "seed": pm["seed"]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(lines))
    print("기록:", folder)
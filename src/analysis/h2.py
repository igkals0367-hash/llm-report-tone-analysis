"""Step 10 H2 (9장): C 리포트 종목의 60거래일 누적 초과수익이 A 리포트 종목보다 낮은가.
분류 고정 뒤에만 돈다 (18장). 판정은 60거래일·양 끝 1% 누르기·라벨 섞기로만 한다.
v2(text_source: pdf)는 _pdf 등급·컷오프로 p < 0.025 판정, --v1same은 v1 등급으로 v2에 남은 쌍만 계산하는 참고 비교 (19장).
사용: python -m src.analysis.h2            (설정의 글 출처로 판정)
      python -m src.analysis.h2 --v1same   (같은 쌍 v1 비교, v2 고정 뒤에만)"""
import datetime as dt
import json
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from src.analysis.guard import CUTOFFS, analysis_mode, require_lock
from src.analysis.permutation import permutation_test
from src.config import load_config, source_path
from src.preprocess.pairs import validation_end
from src.preprocess.universe import load_daily

PAIRS = pathlib.Path("data/processed/pairs.parquet")
REPORTS = pathlib.Path("data/processed/reports.parquet")
UNIVERSE = pathlib.Path("data/processed/universe.parquet")
GRADES = pathlib.Path("data/processed/grades.parquet")
OUT = pathlib.Path("data/processed/h2.parquet")
CONFIG = pathlib.Path("configs/default.yaml")
RUNS = pathlib.Path("runs")
PRE_DAYS = 20                                                   # 점검 2: 리포트 직전 거래일 수 (9장)


def abnormal_cumsum(daily, universe):
    """일별 초과수익(종목 수익률 − 유니버스 평균)을 날짜순으로 더해 둔 표 (4장, 9장 Step 10).
    유니버스 평균은 전 거래일 명단의 단순 평균이다. 데이터가 없는 날의 수익률은 0으로 둔다 (16장).
    → (거래일, 종목코드→열 번호, cum[날짜, 종목])"""
    T = pd.DatetimeIndex(np.sort(daily["date"].unique()))
    ret = daily.pivot(index="date", columns="item_code", values="ret").reindex(T).fillna(0.0)
    nxt = pd.Series(T[1:], index=T[:-1])                        # 명단 날짜 → 그 명단을 쓰는 날(다음 거래일)
    u = universe[["date", "item_code"]].assign(use=universe["date"].map(nxt)).dropna(subset=["use"])
    u = u.merge(daily[["date", "item_code", "ret"]].rename(columns={"date": "use"}), on=["use", "item_code"], how="left")
    um = u.assign(ret=u["ret"].fillna(0.0)).groupby("use")["ret"].mean().reindex(T).fillna(0.0)
    cum = (ret.to_numpy() - um.to_numpy()[:, None]).cumsum(axis=0)
    return T, {c: i for i, c in enumerate(ret.columns)}, cum


def event_cars(T, col, cum, dates, codes, horizon):
    """리포트마다 진입일 위치, 1~horizon일 누적 초과수익, 직전 20거래일 초과수익.
    진입 = 게시일 다음 거래일 종가, 누적 = 진입 다음 거래일부터 (9장). 직전 = 게시일 전 거래일까지 (16장)."""
    d = pd.DatetimeIndex(dates).to_numpy()
    e = np.searchsorted(T.to_numpy(), d, side="right")          # 게시일보다 뒤인 첫 거래일
    q = np.searchsorted(T.to_numpy(), d, side="left") - 1       # 게시일보다 앞인 마지막 거래일
    if (e + horizon >= len(T)).any() or (q - PRE_DAYS < 0).any():
        raise ValueError("주가 데이터 범위를 벗어나는 리포트가 있습니다.")
    j = np.array([col[c] for c in codes])
    car = np.stack([cum[e + k, j] - cum[e, j] for k in range(1, horizon + 1)], axis=1)
    pre = cum[q, j] - cum[q - PRE_DAYS, j]
    return e, car, pre


def dedup(entry_idx, keys, nids, horizon):
    """같은 키(종목·등급)에서 마지막으로 남긴 리포트의 진입일보다 horizon거래일 이상 뒤일 때만 남긴다 (9장, 16장)."""
    df = pd.DataFrame({"e": entry_idx, "k": keys, "nid": nids}).sort_values(["k", "e", "nid"])
    keep = np.zeros(len(df), dtype=bool)
    last_key, last_e = None, None
    for pos, e, k in zip(df.index, df["e"], df["k"]):
        if k != last_key or e - last_e >= horizon:
            keep[pos] = True
            last_key, last_e = k, e
    return keep


def winsorize(x, pct):
    """양 끝 pct%를 경계값으로 누른다. 버리지 않는다 (9장). 2차원이면 열(날짜)마다 따로."""
    lo, hi = np.percentile(x, [pct, 100 - pct], axis=0)
    return np.clip(x, lo, hi)


def in_universe(dates, codes, universe, T):
    """게시일 명단(휴장일이면 직전 거래일 명단)에 들었는지 (4장, 16장)."""
    i = np.searchsorted(T.to_numpy(), pd.DatetimeIndex(dates).to_numpy(), side="right") - 1
    key = pd.MultiIndex.from_arrays([T[np.maximum(i, 0)], codes])
    return (i >= 0) & key.isin(pd.MultiIndex.from_frame(universe[["date", "item_code"]]))


def cumulative_abnormal_returns(pairs, daily, universe, cfg, keep=None):
    """검증 A·C 리포트의 누적 초과수익 표 (9장 Step 10). → (리포트별 표, car 행렬, 거래일, (T, col, cum))
    등급·컷오프는 cfg의 글 출처를 따른다(v2는 _pdf). keep을 주면 그 r_nid만 남긴다(같은 쌍 v1 비교, 19장)."""
    require_lock(source_path(CUTOFFS, cfg))                     # 분류 고정 전이면 여기서 멈춘다
    h = cfg["h2"]["horizon_days"]
    grades = pd.read_parquet(source_path(GRADES, cfg))
    if keep is not None:
        grades = grades[grades["r_nid"].isin(keep)]
    ac = (grades.query("grade in ['A', 'C']")[["r_nid", "grade"]]
          .merge(pairs[["r_nid", "write_date", "item_code", "sector"]], on="r_nid", how="left").reset_index(drop=True))
    T, col, cum = abnormal_cumsum(daily, universe)
    e, car, pre = event_cars(T, col, cum, ac["write_date"], ac["item_code"], h)
    ac["entry_date"], ac["pre"] = T[e], pre
    ac["kept"] = dedup(e, ac["item_code"] + "|" + ac["grade"], ac["r_nid"], h)
    return ac, car, (T, col, cum)


def to_long(meta, car, group):
    n, h = car.shape
    return pd.DataFrame({"r_nid": np.repeat(meta["r_nid"].to_numpy(), h), "grade": group if isinstance(group, str)
                         else np.repeat(np.asarray(group), h), "entry_date": np.repeat(meta["entry_date"].to_numpy(), h),
                         "day": np.tile(np.arange(1, h + 1), n), "car": car.ravel(),
                         "kept": np.repeat(meta["kept"].to_numpy(), h)})


if __name__ == "__main__":
    mode = analysis_mode(load_config(), v1same="--v1same" in sys.argv[1:])
    cfg, tag, alpha = mode["cfg"], mode["tag"], mode["alpha"]  # v1same이면 cfg는 v1(summary) 설정
    cut_path = source_path(CUTOFFS, cfg)
    lock = require_lock(cut_path)
    h2, pm = cfg["h2"], cfg["permutation"]
    H, AUX = h2["horizon_days"], h2["aux_horizon_days"]
    pairs = pd.read_parquet(source_path(PAIRS, cfg))
    daily = load_daily(["date", "item_code", "ret"])
    universe = pd.read_parquet(UNIVERSE)
    ac, car, (T, col, cum) = cumulative_abnormal_returns(pairs, daily, universe, cfg, mode["keep"])

    # 중립 참고선: 검증 기간 중립 의견 리포트 중 게시일 유니버스 (5장, 16장). 검정에는 쓰지 않는다
    rep = pd.read_parquet(REPORTS, columns=["nid", "write_date", "item_code", "is_neutral_ref"])
    v0, v1 = pd.Timestamp(cfg["periods"]["validation_start"]), validation_end(T, cfg)
    nt = rep[rep["is_neutral_ref"] & rep["write_date"].between(v0, v1)]
    nt = nt[in_universe(nt["write_date"], nt["item_code"], universe, T)].rename(columns={"nid": "r_nid"}).reset_index(drop=True)
    ne, ncar, _ = event_cars(T, col, cum, nt["write_date"], nt["item_code"], H)
    nt["entry_date"] = T[ne]
    nt["kept"] = dedup(ne, nt["item_code"], nt["r_nid"], H)

    out = OUT.with_name(f"{OUT.stem}{tag}{OUT.suffix}")         # h2.parquet, h2_pdf.parquet, h2_v1same.parquet
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([to_long(ac, car, ac["grade"]), to_long(nt, ncar, "N")]).to_parquet(out, index=False)

    k = ac["kept"].to_numpy()
    s, c = ac[k].reset_index(drop=True), car[k]
    is_c = (s["grade"] == "C").to_numpy()
    strata = s["write_date"].dt.strftime("%Y-%m") + "|" + s["sector"].astype(str)
    W = winsorize(c, h2["winsor_pct"])                           # 날짜마다 A·C 합쳐 누르기, 섞기 전에 한 번 (16장)

    def test(values):
        d, p, sh = permutation_test(values, s["grade"], strata, pm["n"], pm["seed"])
        return {"A": float(values[~is_c].mean()), "C": float(values[is_c].mean()), "diff": d, "p": p}, sh

    main, shuffled = test(W[:, H - 1])                           # 판정: 60거래일, 1%
    verdict = None if alpha is None else ("지지" if main["p"] < alpha else "미지지")
    verdict_line = ("같은 쌍 v1 비교는 참고라 판정하지 않는다 (19장)" if alpha is None
                    else f"판정 (p < {alpha}): **H2 {verdict}**")
    aux_w, _ = test(winsorize(c[:, H - 1], h2["aux_winsor_pct"]))   # 보조: 60거래일, 5%
    aux_d, _ = test(W[:, AUX - 1])                               # 보조: 20거래일, 1%
    raw = {"A": float(c[~is_c, H - 1].mean()), "C": float(c[is_c, H - 1].mean())}

    ban = (s["write_date"] <= pd.Timestamp(cfg["short_ban"]["end"])).to_numpy()   # 점검 1
    curve = lambda m: W[m].mean(axis=0).round(6).tolist() if m.any() else []
    curves = {"A": curve(~is_c), "C": curve(is_c),
              "N": winsorize(ncar[nt["kept"].to_numpy()], h2["winsor_pct"]).mean(axis=0).round(6).tolist(),
              "ban_A": curve(ban & ~is_c), "ban_C": curve(ban & is_c),
              "after_A": curve(~ban & ~is_c), "after_C": curve(~ban & is_c)}
    pre_w = winsorize(s["pre"].to_numpy(), h2["winsor_pct"])     # 점검 2
    pre = {"A": float(pre_w[~is_c].mean()), "C": float(pre_w[is_c].mean())}
    counts = {g: {"R": int((ac["grade"] == g).sum()), "kept": int(((ac["grade"] == g) & ac["kept"]).sum()),
                  "ban": int((ban & (is_c == (g == "C"))).sum())} for g in ["A", "C"]}
    counts["N"] = {"R": len(nt), "kept": int(nt["kept"].sum())}

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = RUNS / f"{stamp}_h2{tag}"
    folder.mkdir(parents=True, exist_ok=True)
    row = lambda name, r: f"| {name} | {r['A']:+.2%} | {r['C']:+.2%} | {r['diff']:+.2%} | {r['p']:.4f} |"
    lines = [f"# H2 (Step 10) {stamp}", mode["label"],
             f"커밋 {commit}{' (커밋 안 된 변경 있음)' if dirty else ''} · 고정 커밋 {lock['git_commit'][:7]} · "
             f"컷오프 {lock['lower']:.5f} / {lock['upper']:.5f}", "",
             "| 등급 | 검증 R | 중복 제거 뒤 | 그중 게시일이 공매도 금지 기간 |", "|---|---|---|---|"]
    lines += [f"| {g} | {counts[g]['R']:,} | {counts[g]['kept']:,} | {counts[g]['ban']:,} |" for g in ["A", "C"]]
    lines += [f"| 중립 참고 | {counts['N']['R']:,} | {counts['N']['kept']:,} | |", "",
              "| 누적 초과수익 | A 평균 | C 평균 | 차이 C − A | p |", "|---|---|---|---|---|",
              row(f"**판정: {H}거래일, 양 끝 {h2['winsor_pct']}% 누름**", main),
              row(f"보조: {H}거래일, {h2['aux_winsor_pct']}% 누름", aux_w),
              row(f"보조: {AUX}거래일, {h2['winsor_pct']}% 누름", aux_d), "",
              f"누르기 전 {H}거래일 평균: A {raw['A']:+.2%}, C {raw['C']:+.2%}",
              f"라벨 섞기 {pm['n']:,}회 (칸: 게시 연월 × 업종, 시드 {pm['seed']}): 섞은 차이 중간값 "
              f"{np.median(shuffled):+.2%}, 5% 지점 {np.quantile(shuffled, 0.05):+.2%}",
              verdict_line, "", "## 점검 (판정에 쓰지 않음)",
              f"- 점검 1, {H}거래일 평균: 금지 기간 게시 A {curves['ban_A'][-1]:+.2%} · C {curves['ban_C'][-1]:+.2%}, "
              f"그 뒤 게시 A {curves['after_A'][-1]:+.2%} · C {curves['after_C'][-1]:+.2%}",
              f"- 점검 2, 리포트 직전 {PRE_DAYS}거래일 초과수익(게시일 전날까지, {h2['winsor_pct']}% 누름): "
              f"A {pre['A']:+.2%}, C {pre['C']:+.2%}",
              f"- 중립 참고선 {H}거래일: {curves['N'][-1]:+.2%}"]
    (folder / "h2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy(CONFIG, folder / "config.yaml")
    shutil.copy(cut_path, folder / cut_path.name)
    (folder / "h2.json").write_text(json.dumps(
        {"mode": tag or "_v1", "label": mode["label"], "alpha": alpha, "main": main, "verdict": verdict, "aux_winsor": aux_w, "aux_days": aux_d, "raw": raw, "pre": pre,
         "counts": counts, "curves": curves, "shuffled": shuffled.round(6).tolist(), "commit": commit, "dirty": dirty},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(lines))
    print("기록:", folder)
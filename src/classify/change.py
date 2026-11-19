"""Step 6: 톤 변화 (7장). 톤 변화 = 톤(R) − 톤(P)를 쌍 표에 열로 붙인다.
쌍 목록은 언제나 pairs.parquet에서 읽는다. 결과는 v1(text_source: summary)이면 pairs.parquet에 다시 쓰고,
v2(pdf)면 pairs_pdf.parquet에 따로 쓴다. 톤도 tone.parquet / tone_pdf.parquet 중 글 출처에 맞는 것을 읽는다 (19장 코드)."""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import load_config, source_path

PAIRS = pathlib.Path("data/processed/pairs.parquet")
TONE = pathlib.Path("data/processed/tone.parquet")
FIG = pathlib.Path("reports/figures/tone_change_baseline.png")
COLS = ["tone_r", "tone_p", "tone_change"]


def add_tone_change(pairs, tone):
    """pairs에 tone_r, tone_p, tone_change(= tone_r − tone_p)를 붙인다.
    R·P 중 톤이 없는 쌍은 뺀다 (7장 Step 4-5, Step 6)."""
    t = tone.set_index("nid")["tone"]
    out = pairs.drop(columns=COLS, errors="ignore")             # 다시 실행해도 열이 두 벌 생기지 않게
    out = out.assign(tone_r=out["r_nid"].map(t), tone_p=out["p_nid"].map(t))
    out = out.dropna(subset=["tone_r", "tone_p"]).reset_index(drop=True)
    out["tone_change"] = out["tone_r"] - out["tone_p"]
    return out


def save_baseline_hist(baseline, path):
    """Step 6 확인: 기준선 톤 변화 히스토그램을 저장한다 (7장 Step 6, 2026-11-13 결정)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(baseline, bins=40, color="#3A6EA5", edgecolor="white")
    ax.axvline(0, color="#555555", linewidth=1)
    ax.set_xlabel("Tone change = tone(R) − tone(P)")
    ax.set_ylabel("Pairs")
    ax.set_title(f"Baseline tone change (n = {len(baseline):,}, mean {baseline.mean():+.3f})", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    cfg = load_config()
    tone_path, out_path, fig_path = (source_path(p, cfg) for p in (TONE, PAIRS, FIG))
    tone = pd.read_parquet(tone_path)
    if set(tone["model_revision"]) != {cfg["model"]["revision"]}:     # 고정한 리비전으로 매긴 톤만 쓴다
        raise ValueError(f"{tone_path.name}의 리비전 {set(tone['model_revision'])}이 설정과 다름")
    pairs = pd.read_parquet(PAIRS)                       # 쌍 목록은 v1과 같다. 예전 톤 열은 add_tone_change가 지우고 새로 붙인다
    out = add_tone_change(pairs, tone)
    out.to_parquet(out_path, index=False)
    print(f"글 출처: {cfg.get('text_source', 'summary')} → {out_path}")
    print(f"표본 쌍 {len(pairs):,} → 톤 변화가 붙은 쌍 {len(out):,} (톤이 없어 뺀 쌍 {len(pairs) - len(out)})")
    print("기간별:", out.groupby("period").size().to_dict())
    b = out.loc[out["period"] == "baseline", "tone_change"]
    stats = b.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(3).to_dict()
    print("기준선 톤 변화:", {k: stats[k] for k in ["mean", "std", "min", "5%", "25%", "50%", "75%", "95%", "max"]},
          f"정확히 0인 비율 {(b == 0).mean():.3f}")
    save_baseline_hist(b, fig_path)
    print("그림:", fig_path)
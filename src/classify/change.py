"""Step 6: 톤 변화 (7장). 톤 변화 = 톤(R) − 톤(P)를 pairs.parquet에 열로 붙인다."""
import pathlib

import pandas as pd

from src.config import load_config

PAIRS = pathlib.Path("data/processed/pairs.parquet")
TONE = pathlib.Path("data/processed/tone.parquet")
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


if __name__ == "__main__":
    cfg = load_config()
    tone = pd.read_parquet(TONE)
    if set(tone["model_revision"]) != {cfg["model"]["revision"]}:     # 고정한 리비전으로 매긴 톤만 쓴다
        raise ValueError(f"tone.parquet의 리비전 {set(tone['model_revision'])}이 설정과 다름")
    pairs = pd.read_parquet(PAIRS)
    out = add_tone_change(pairs, tone)
    out.to_parquet(PAIRS, index=False)
    print(f"표본 쌍 {len(pairs):,} → 톤 변화가 붙은 쌍 {len(out):,} (톤이 없어 뺀 쌍 {len(pairs) - len(out)})")
    print("기간별:", out.groupby("period").size().to_dict())
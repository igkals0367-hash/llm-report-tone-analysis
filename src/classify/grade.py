"""Step 8 분류 고정 (8장). 기준선 톤 변화의 두 백분위로 컷오프를 정해 잠그고, 검증 표본에 A·B·C를 붙인다.
Step 7 점검도 같은 fit_cutoffs·assign_grade로 임시 등급을 붙여 계산을 하나로 둔다.
v2(text_source: pdf)는 pairs_pdf·tone_pdf를 읽고 cutoffs_pdf.json, grades_pdf.parquet에 쓴다. v1 파일은 건드리지 않는다 (19장)."""
import datetime as dt
import json
import pathlib
import subprocess

import numpy as np
import pandas as pd

from src.analysis.guard import CUTOFFS, require_lock
from src.config import load_config, source_path

PAIRS = pathlib.Path("data/processed/pairs.parquet")
TONE = pathlib.Path("data/processed/tone.parquet")
GRADES = pathlib.Path("data/processed/grades.parquet")


def fit_cutoffs(baseline_tone_changes, cfg=None):
    """기준선 톤 변화 → (lower, upper) = (33.3 백분위, 66.7 백분위) (8장, 12장)."""
    g = (cfg or load_config())["grade"]
    x = np.asarray(baseline_tone_changes, dtype=float)
    return float(np.percentile(x, g["lower_pct"])), float(np.percentile(x, g["upper_pct"]))


def assign_grade(tone_change, cutoffs):
    """A: upper보다 큼, C: lower보다 작음, B: 그 사이(경계 포함) (8장 등급 표)."""
    lower, upper = cutoffs
    x = np.asarray(tone_change, dtype=float)
    g = np.where(x > upper, "A", np.where(x < lower, "C", "B"))
    return g.item() if g.ndim == 0 else g


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def lock(cfg, path=None):
    """컷오프를 계산해 cutoffs.json(v2는 cutoffs_pdf.json)에 봉인한다(회원가입처럼: 없어야 만든다) (8장, 18장).
    파일이 이미 있거나, 커밋 안 된 변경이 있거나, 톤의 모델 리비전이 설정과 다르면 멈춘다."""
    path = pathlib.Path(path or source_path(CUTOFFS, cfg))
    if path.exists():
        raise FileExistsError(f"{path}이 이미 있습니다. 다시 고정하려면 파일을 직접 지우고 "
                              "PROGRESS.md에 이유를 적으세요 (18장 덮어쓰기 금지).")
    if _git("status", "--porcelain"):
        raise RuntimeError("커밋 안 된 변경이 있습니다. 코드를 먼저 커밋한 뒤 고정하세요 "
                           f"({path.name}의 git_commit이 실제로 돌린 코드를 가리키게).")
    tone_path = source_path(TONE, cfg)
    revs = set(pd.read_parquet(tone_path, columns=["model_revision"])["model_revision"])
    if revs != {cfg["model"]["revision"]}:
        raise ValueError(f"{tone_path.name}의 리비전 {revs}이 설정({cfg['model']['revision']})과 다릅니다.")

    pairs = pd.read_parquet(source_path(PAIRS, cfg), columns=["period", "tone_change"])
    base = pairs.loc[pairs["period"] == "baseline", "tone_change"]
    lower, upper = fit_cutoffs(base, cfg)
    record = {"model": cfg["model"]["name"], "model_revision": cfg["model"]["revision"],
              "text_source": cfg.get("text_source", "summary"),
              "lower": lower, "upper": upper, "baseline_n": int(len(base)),
              "h1_window_months": cfg["h1"]["window_months"],
              "locked_at": dt.datetime.now().isoformat(timespec="seconds"),
              "git_commit": _git("rev-parse", "HEAD")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def grade_validation(cfg, path=None):
    """잠긴 컷오프로 검증 표본에 등급을 붙여 grades.parquet(v2는 grades_pdf.parquet)에 저장한다 (8장 '고정 뒤에 하는 것')."""
    rec = require_lock(path or source_path(CUTOFFS, cfg))       # 고정 뒤에만 지나간다
    if rec["model_revision"] != cfg["model"]["revision"]:
        raise ValueError("설정의 모델 리비전이 고정할 때와 다릅니다.")
    pairs = pd.read_parquet(source_path(PAIRS, cfg), columns=["r_nid", "period", "tone_change"])
    v = pairs[pairs["period"] == "validation"]
    grades = pd.DataFrame({"r_nid": v["r_nid"].to_numpy(), "tone_change": v["tone_change"].to_numpy(),
                           "grade": assign_grade(v["tone_change"], (rec["lower"], rec["upper"]))})
    out = source_path(GRADES, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    grades.to_parquet(out, index=False)
    return grades


if __name__ == "__main__":
    cfg = load_config()
    path = source_path(CUTOFFS, cfg)
    record = lock(cfg, path)
    print(f"{path}:", json.dumps(record, ensure_ascii=False, indent=2))
    g = grade_validation(cfg, path)
    print("등급 파일:", source_path(GRADES, cfg))
    n = g["grade"].value_counts().reindex(["A", "B", "C"], fill_value=0)
    print(f"검증 {len(g):,}쌍 등급:", {k: f"{v:,} ({v / len(g):.1%})" for k, v in n.items()})
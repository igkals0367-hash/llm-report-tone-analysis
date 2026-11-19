"""백테스트 실행에 함께 쓰는 것: 실제 데이터로 Store 만들기, runs/ 기록."""
import datetime as dt
import pathlib
import shutil
import subprocess

import pandas as pd

from src.analysis.guard import CUTOFFS
from src.preprocess.universe import load_daily
from src.strategy.store import Store

REPORTS = pathlib.Path("data/processed/reports.parquet")
UNIVERSE = pathlib.Path("data/processed/universe.parquet")
GRADES = pathlib.Path("data/processed/grades.parquet")
CONFIG = pathlib.Path("configs/default.yaml")
RUNS = pathlib.Path("runs")


def real_store():
    """모든 리포트(날짜·증권사·종목만), 일별 등락률, 유니버스 명단. 등급은 여기서 붙이지 않는다."""
    reports = pd.read_parquet(REPORTS, columns=["nid", "write_date", "broker_code", "item_code"])
    return Store(reports, load_daily(["date", "item_code", "ret"]), pd.read_parquet(UNIVERSE, columns=["date", "item_code"]))


def new_run_folder(kind, cutoffs=CUTOFFS):
    """runs/<실행 시각>_<kind>/ 를 만들고 설정·컷오프 사본을 넣는다 (11장 재현성). → (폴더, 머리말 줄)
    cutoffs: 사본으로 넣을 봉인 파일. v2는 models/cutoffs_pdf.json (19장)."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = RUNS / f"{stamp}_{kind}"
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(CONFIG, folder / "config.yaml")
    cutoffs = pathlib.Path(cutoffs)
    shutil.copy(cutoffs, folder / cutoffs.name)
    return folder, stamp, f"커밋 {commit or '?'}{' (커밋 안 된 변경 있음)' if dirty else ''}", {"commit": commit, "dirty": dirty}
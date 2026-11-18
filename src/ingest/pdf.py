"""v2 Step 1b (19장): 표본 쌍의 R·P 리포트 PDF를 받아 두고, 글자를 덩어리(칸)별로 뽑아 줄 단위로 저장한다.
리포트마다 상세 API → attachUrl → data/raw/pdf/<nid>.pdf → 글자 → data/raw/pdf_text/<nid>.txt.
진행은 data/raw/pdf_manifest.csv에 남는다. 다시 실행하면 끝난 리포트(ok, no_url, no_text)는 건너뛰고
download_failed는 다시 시도한다. PDF는 Step 4b 확인이 끝날 때까지 지우지 않는다(2026-11-18 변경).
글자 정리(표·면책 고지 빼기)는 여기서 하지 않는다 (19장 Step 4b).
사용: python -m src.ingest.pdf              (전체)
      python -m src.ingest.pdf 20           (아직 안 끝난 것 중 앞의 20개만, 시험용)
      python -m src.ingest.pdf --reextract  (받아 둔 PDF에서 텍스트만 다시 뽑는다. 네트워크를 쓰지 않는다)"""
import csv
import json
import pathlib
import re
import sys
import time
import urllib.request

import pandas as pd
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams

from src.config import load_config

API = "https://stock.naver.com/api/stockSecurity/researches/v2/company/{nid}"
HEADERS = {"User-Agent": "Mozilla/5.0"}
PAIRS = pathlib.Path("data/processed/pairs.parquet")
TEXT = pathlib.Path("data/raw/pdf_text")
PDF = pathlib.Path("data/raw/pdf")
MANIFEST = pathlib.Path("data/raw/pdf_manifest.csv")
FIELDS = ["nid", "attach_url", "status", "pages", "hangul_chars"]
DONE = {"ok", "no_url", "no_text"}                     # 다시 실행할 때 건너뛰는 상태 (19장 ③)
HANGUL = re.compile(r"[가-힣]")


def hangul_count(text):
    """한글 글자 수. 스캔 이미지나 글꼴이 깨진 PDF는 한글이 거의 안 나온다 (19장 ②)."""
    return len(HANGUL.findall(text))


def todo(nids, manifest):
    """아직 끝나지 않은 nid만 순서대로 남긴다. 같은 nid가 여러 번 적혀 있으면 마지막 줄을 따른다."""
    last = {int(row["nid"]): row["status"] for row in manifest}
    return [n for n in nids if last.get(int(n)) not in DONE]


def fetch(url, cfg, as_json=False):
    """요청 하나. 실패하면 간격을 2배씩 늘려 가며 retries번까지 다시 시도한다 (19장 ④)."""
    p = cfg["pdf"]
    for attempt in range(p["retries"] + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=p["timeout_sec"]) as r:
                body = r.read()
            return json.loads(body.decode("utf-8")) if as_json else body
        except Exception:
            if attempt == p["retries"]:
                raise
            time.sleep(2 * p["sleep_sec"] * 2 ** attempt)


def extract_lines(path, cfg):
    """pdfminer.six 레이아웃 분석으로 서로 가까운 글자를 덩어리로 묶어 덩어리마다 읽는다 (19장 ①).
    나란히 놓인 표와 본문이 한 줄로 섞이지 않고, 덩어리 사이에는 빈 줄이 들어간다. → (줄 목록, 페이지 수)"""
    text = extract_text(str(path), laparams=LAParams(line_margin=cfg["pdf"]["line_margin"]))
    return text.replace("\x0c", "\n").splitlines(), text.count("\x0c")    # \x0c = 페이지 끝 표시


def text_row(nid, path, cfg, row):
    """받아 둔 PDF에서 글자를 뽑아 텍스트 파일로 저장하고 상태를 정한다 (19장 ②)."""
    try:
        lines, pages = extract_lines(path, cfg)
    except Exception:                                    # 열리지 않는 PDF는 글자 없음과 같게 본다
        lines, pages = [], 0
    text = "\n".join(lines)
    row = {**row, "pages": pages, "hangul_chars": hangul_count(text)}
    out = TEXT / f"{nid}.txt"
    if row["hangul_chars"] < cfg["pdf"]["min_hangul"]:
        out.unlink(missing_ok=True)
        return {**row, "status": "no_text"}
    out.write_text(text, encoding="utf-8")
    return {**row, "status": "ok"}


def process(nid, cfg):
    """리포트 하나: 상세 API → PDF 받아 두기 → 글자 뽑기 → 텍스트 저장. → manifest 한 줄"""
    row = {"nid": nid, "attach_url": "", "status": "", "pages": "", "hangul_chars": ""}
    try:
        detail = fetch(API.format(nid=nid), cfg, as_json=True)
    except Exception:
        return {**row, "status": "download_failed"}
    url = detail.get("attachUrl") if isinstance(detail, dict) else None
    if not url:
        return {**row, "status": "no_url"}
    row["attach_url"] = url
    time.sleep(cfg["pdf"]["sleep_sec"])
    path = PDF / f"{nid}.pdf"
    try:
        path.write_bytes(fetch(url, cfg))
    except Exception:
        return {**row, "status": "download_failed"}
    return text_row(nid, path, cfg, row)


def read_manifest():
    if not MANIFEST.exists():
        return []
    with open(MANIFEST, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def latest(manifest):
    """nid마다 마지막 줄만 남긴다 (처음 나온 순서대로)."""
    out = {}
    for row in manifest:
        out[int(row["nid"])] = row
    return list(out.values())


def reextract(cfg):
    """받아 둔 PDF가 있는 리포트는 텍스트만 다시 뽑아 상태를 고치고 manifest를 새로 쓴다 (네트워크 없음)."""
    rows = []
    for row in latest(read_manifest()):
        path = PDF / f"{row['nid']}.pdf"
        rows.append(text_row(int(row["nid"]), path, cfg, row) if path.exists() else row)
    with open(MANIFEST, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in FIELDS} for r in rows)
    return len(rows)


def summary():
    """상태별 리포트 수와, R·P 중 하나라도 ok가 아니어서 빠질 쌍의 수를 기간별로 센다 (19장 빠지는 쌍)."""
    last = {int(r["nid"]): r["status"] for r in read_manifest()}
    p = pd.read_parquet(PAIRS, columns=["r_nid", "p_nid", "period"])
    ok = p["r_nid"].map(last).eq("ok") & p["p_nid"].map(last).eq("ok")
    done = p["r_nid"].map(last).notna() & p["p_nid"].map(last).notna()
    print("리포트 상태:", pd.Series(last).value_counts().to_dict())
    t = pd.DataFrame({"쌍": p.groupby("period").size(), "R·P 모두 처리됨": done.groupby(p["period"]).sum(),
                      "빠질 쌍": (done & ~ok).groupby(p["period"]).sum()})
    t["빠질 비율"] = (t["빠질 쌍"] / t["R·P 모두 처리됨"]).round(4)
    print(t.to_string())


if __name__ == "__main__":
    cfg = load_config()
    TEXT.mkdir(parents=True, exist_ok=True)
    PDF.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] == ["--reextract"]:
        print("다시 뽑은 리포트:", reextract(cfg))
        summary()
        sys.exit()
    pairs = pd.read_parquet(PAIRS, columns=["r_nid", "p_nid"])
    nids = todo(sorted(set(pairs["r_nid"]) | set(pairs["p_nid"])), read_manifest())
    if len(sys.argv) > 1:
        nids = nids[:int(sys.argv[1])]
    new = not MANIFEST.exists()
    with open(MANIFEST, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for i, nid in enumerate(nids, 1):
            row = process(int(nid), cfg)
            w.writerow(row)
            f.flush()                                    # 끊겨도 여기까지는 남는다
            if i % 50 == 0 or i == len(nids):
                print(f"{i}/{len(nids)} 처리 (마지막: {row['status']})")
            time.sleep(cfg["pdf"]["sleep_sec"])
    summary()
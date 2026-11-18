"""v2 Step 1b (19장): 표본 쌍의 R·P 리포트 PDF를 받아 글자만 줄 단위로 뽑아 저장하고 PDF는 지운다.
리포트마다 상세 API → attachUrl → PDF → 글자 → data/raw/pdf_text/<nid>.txt. 진행은 data/raw/pdf_manifest.csv에 남는다.
다시 실행하면 이미 끝난 리포트(ok, no_url, no_text)는 건너뛰고 download_failed는 다시 시도한다.
글자 정리(표·면책 고지 빼기)는 여기서 하지 않는다 (19장 Step 4b).
사용: python -m src.ingest.pdf        (전체)
      python -m src.ingest.pdf 20     (아직 안 끝난 것 중 앞의 20개만, 시험용)"""
import csv
import json
import pathlib
import re
import sys
import time
import urllib.request

import pandas as pd
import pdfplumber

from src.config import load_config

API = "https://stock.naver.com/api/stockSecurity/researches/v2/company/{nid}"
HEADERS = {"User-Agent": "Mozilla/5.0"}
PAIRS = pathlib.Path("data/processed/pairs.parquet")
TEXT = pathlib.Path("data/raw/pdf_text")
TMP = pathlib.Path("data/raw/pdf_tmp")
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


def extract_lines(path):
    """페이지마다 글자를 뽑아 줄 단위 목록으로 돌려준다. → (줄 목록, 페이지 수)"""
    lines = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            lines += (page.extract_text() or "").splitlines()
        return lines, len(pdf.pages)


def process(nid, cfg):
    """리포트 하나: 상세 API → PDF 받기 → 글자 뽑기 → 텍스트 저장 → PDF 지우기. → manifest 한 줄"""
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
    tmp = TMP / f"{nid}.pdf"
    try:
        tmp.write_bytes(fetch(url, cfg))
    except Exception:
        return {**row, "status": "download_failed"}
    try:
        lines, pages = extract_lines(tmp)
    except Exception:                                    # 열리지 않는 PDF는 글자 없음과 같게 본다
        lines, pages = [], 0
    finally:
        tmp.unlink(missing_ok=True)                      # PDF는 남기지 않는다 (19장, Hamin 결정)
    text = "\n".join(lines)
    row.update(pages=pages, hangul_chars=hangul_count(text))
    if row["hangul_chars"] < cfg["pdf"]["min_hangul"]:
        return {**row, "status": "no_text"}
    (TEXT / f"{nid}.txt").write_text(text, encoding="utf-8")
    return {**row, "status": "ok"}


def read_manifest():
    if not MANIFEST.exists():
        return []
    with open(MANIFEST, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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
    TMP.mkdir(parents=True, exist_ok=True)
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
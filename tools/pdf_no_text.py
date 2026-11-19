"""v2 Step 1b 점검 (19장 빠지는 쌍): no_text 리포트에서 왜 한글이 안 나왔는지 건수로만 본다.
받아 둔 PDF를 다시 읽어 글자 수를 센다. 원문은 출력하지 않는다. 결과 계산에 쓰지 않는다.
사용: python -m tools.pdf_no_text"""
import re

import pandas as pd
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams

from src.config import load_config
from src.ingest.pdf import PDF, hangul_count, latest, read_manifest
from src.preprocess.text import REPORTS

CID = re.compile(r"\(cid:\d+\)")       # 글꼴에 글자 대응표가 없을 때 pdfminer가 글자 대신 내놓는 표시
LATIN = re.compile(r"[A-Za-z]")


def diagnose(path, cfg):
    """PDF 하나 → 이유와 숫자. 이유는 위에서부터 처음 맞는 것 하나."""
    row = {"size_kb": round(path.stat().st_size / 1024) if path.exists() else 0,
           "chars": 0, "cid": 0, "latin": 0, "hangul": 0}
    if not path.exists():
        return {**row, "reason": "PDF 없음"}
    try:
        text = extract_text(str(path), laparams=LAParams(line_margin=cfg["pdf"]["line_margin"]))
    except Exception:
        return {**row, "reason": "열리지 않음"}            # 받다가 잘린 파일이면 여기로 온다
    row["cid"] = len(CID.findall(text))
    text = CID.sub("", text)
    row.update(chars=len(re.sub(r"\s", "", text)), latin=len(LATIN.findall(text)), hangul=hangul_count(text))
    if row["cid"] >= 100:
        reason = "글꼴 대응표 없음"                          # 글자는 있는데 무슨 글자인지 모름
    elif row["chars"] < 200:
        reason = "글자 거의 없음"                            # 스캔 이미지
    elif row["latin"] >= row["chars"] / 2:
        reason = "영문 위주"                                 # 영문 리포트
    else:
        reason = "그 밖"
    return {**row, "reason": reason}


if __name__ == "__main__":
    cfg = load_config()
    manifest = latest(read_manifest())
    bad = [r for r in manifest if r["status"] == "no_text"]
    d = pd.DataFrame([{"nid": int(r["nid"]), "pages": int(r["pages"] or 0),
                       **diagnose(PDF / f"{r['nid']}.pdf", cfg)} for r in bad])
    print(f"no_text 리포트 {len(d)}개\n")
    print(d.groupby("reason").agg(리포트=("nid", "size"), 쪽=("pages", "median"), 크기KB=("size_kb", "median"),
                                  글자=("chars", "median"), cid=("cid", "median"), 영문=("latin", "median"),
                                  한글=("hangul", "median")).to_string())
    print("(쪽부터는 중앙값)\n")

    r = pd.read_parquet(REPORTS, columns=["nid", "write_date", "broker_name"])
    r = r[r["nid"].isin({int(x["nid"]) for x in manifest})].copy()
    r["reason"] = r["nid"].map(d.set_index("nid")["reason"])
    t = pd.crosstab(r["broker_name"], r["reason"])
    t["no_text"] = t.sum(axis=1)
    t["전체"] = r.groupby("broker_name").size()
    t["비율"] = (t["no_text"] / t["전체"]).round(3)
    print("증권사별 (no_text가 있는 곳만)")
    print(t[t["no_text"] > 0].sort_values("no_text", ascending=False).to_string(), "\n")
    y = r.groupby(pd.to_datetime(r["write_date"]).dt.year)["reason"].agg(no_text="count", 전체="size")
    y["비율"] = (y["no_text"] / y["전체"]).round(3)
    print("연도별")
    print(y.to_string())
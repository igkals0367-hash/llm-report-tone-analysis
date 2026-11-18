"""v2 Step 4b (19장): PDF에서 뽑은 글자를 문장으로 정리한다.
(a) 문장다운 줄만 남기기 → v1 규칙으로 잇고 나누기 → (b) 여러 리포트에 반복되는 정형 문구 빼기 → v1 전처리.
입력: data/raw/pdf_text/<nid>.txt (pdf_manifest.csv에서 ok인 리포트). 출력: data/interim/sentences_pdf.parquet
사용: python -m src.preprocess.pdf_text                (전체)
      python -m src.preprocess.pdf_text --check 74859  (한 리포트의 남김·뺌 이유를 data/interim/pdf_check_<nid>.txt에 쓴다.
                                                       원문이 들어 있으니 직접 열어 보기만 하고 대화창에는 붙이지 않는다)"""
import pathlib
import re
import sys
from collections import Counter

import pandas as pd

from src.config import load_config
from src.ingest.pdf import TEXT, read_manifest
from src.preprocess.text import ALIASES, PAIRS, REPORTS, clean_sentences, split_sentences

OUT = pathlib.Path("data/interim/sentences_pdf.parquet")
LETTER = re.compile(r"[가-힣A-Za-z]")
HANGUL = re.compile(r"[가-힣]")
SPACE_DIGIT = re.compile(r"[\s\d]")
SENT_END = re.compile(r"[.?!]$")


def line_reason(line, cfg):
    """(a) 줄 하나를 남기면 None, 빼면 이유('empty', 'symbol', 'short'). 숫자를 지우기 전에 판정한다.
    숫자·기호 = 공백을 뺀 글자 중 한글·영문이 아닌 것 (19장 Step 4b-2).
    한글이 적어도 . ? !로 끝나는 줄은 앞 줄에서 이어진 문장의 끝일 수 있어 남긴다."""
    chars = re.sub(r"\s", "", line)
    if not chars:
        return "empty"
    if 1 - len(LETTER.findall(chars)) / len(chars) >= cfg["max_symbol_ratio"]:
        return "symbol"
    if len(HANGUL.findall(chars)) < cfg["min_line_hangul"] and not SENT_END.search(chars):
        return "short"
    return None


def split_pdf(title, lines, cfg):
    """(a)를 통과한 줄을 v1 규칙으로 잇고 나눈다 (19장 Step 4b-3). 뺀 줄이나 빈 줄은 문장 경계로 본다:
    표를 사이에 둔 앞뒤 글이 한 문장으로 붙지 않게. → (제목 문장 목록, 본문 문장 목록, 줄별 이유)"""
    reasons = [line_reason(line, cfg) for line in lines]
    blocks, cur = [], []
    for line, r in zip(lines, reasons):
        if r is None:
            cur.append(line)
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    body = [s for b in blocks for s in split_sentences("", "\n".join(b))]
    return split_sentences(title, ""), body, reasons


def key(sentence):
    """(b)에서 같은 문장인지 볼 때 쓰는 열쇠: 공백과 숫자를 지운다 (날짜·번호만 다른 고지도 같게 본다)."""
    return SPACE_DIGIT.sub("", sentence)


def boilerplate(bodies, cfg):
    """(b) 서로 다른 리포트 N개 이상에 나오는 문장 열쇠의 집합 (19장 Step 4b-4). bodies: {nid: 본문 문장 목록}"""
    count = Counter(k for sents in bodies.values() for k in {key(s) for s in sents})
    return {k for k, n in count.items() if n >= cfg["boilerplate_min_reports"]}


def load_aliases(r):
    """종목코드별 별명. v1 build_sentences와 같은 방식 (16장)."""
    aliases = (pd.read_csv(ALIASES, dtype={"item_code": str}).groupby("item_code")["alias"].apply(set).to_dict())
    for code, name in r[["item_code", "item_name"]].drop_duplicates().itertuples(index=False):
        aliases.setdefault(code, set()).add(name)
    return aliases


def split_all(cfg):
    """manifest에서 ok인 리포트를 모두 읽어 (a)와 문장 나누기까지 한다. → (리포트 표, {nid: (제목, 본문)}, 줄 이유 개수)"""
    ok = {int(row["nid"]) for row in read_manifest() if row["status"] == "ok"}
    r = pd.read_parquet(REPORTS, columns=["nid", "item_code", "item_name", "title"])
    split, lines_stat = {}, Counter()
    for nid, title in r.loc[r["nid"].isin(ok), ["nid", "title"]].itertuples(index=False):
        lines = (TEXT / f"{nid}.txt").read_text(encoding="utf-8").splitlines()
        head, body, reasons = split_pdf(title, lines, cfg)
        split[nid] = (head, body)
        lines_stat.update(reason or "kept" for reason in reasons)
    return r, split, lines_stat


def build(cfg):
    """19장 Step 4b 전체. sentences_pdf.parquet를 저장하고 확인용 숫자를 돌려준다."""
    pc = cfg["pdf"]
    r, split, lines_stat = split_all(pc)
    bp = boilerplate({nid: body for nid, (_, body) in split.items()}, pc)
    aliases = load_aliases(r)
    codes = dict(zip(r["nid"], r["item_code"]))
    rows, sent_stat = [], Counter()
    for nid, (head, body) in split.items():
        kept = [s for s in body if key(s) not in bp]
        sent_stat.update(body=len(body), boilerplate=len(body) - len(kept))
        cleaned = clean_sentences(head + kept, aliases.get(codes[nid], set()))   # v1 전처리 그대로 (19장 Step 4b-5)
        rows += [{"nid": nid, "sent_idx": i, "text": s} for i, s in enumerate(cleaned)]
    out = pd.DataFrame(rows, columns=["nid", "sent_idx", "text"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    return out, lines_stat, sent_stat, len(bp)


def check(nid, cfg):
    """한 리포트의 줄마다 (a) 결과와, 문장마다 (b) 결과를 적은 확인용 파일을 만든다. 원문이 들어 있다."""
    pc = cfg["pdf"]
    r, split, _ = split_all(pc)
    bp = boilerplate({n: body for n, (_, body) in split.items()}, pc)
    title = r.loc[r["nid"] == nid, "title"].iloc[0]
    lines = (TEXT / f"{nid}.txt").read_text(encoding="utf-8").splitlines()
    head, body, reasons = split_pdf(title, lines, pc)
    out = ["# 줄 (a)"] + [f"[{r or 'keep'}] {line}" for line, r in zip(lines, reasons)]
    out += ["", "# 문장 (b)"] + [f"[{'boilerplate' if key(s) in bp else 'keep'}] {s}" for s in head + body]
    path = OUT.parent / f"pdf_check_{nid}.txt"
    path.write_text("\n".join(out), encoding="utf-8")
    return path


if __name__ == "__main__":
    cfg = load_config()
    if len(sys.argv) > 2 and sys.argv[1] == "--check":
        print("확인 파일:", check(int(sys.argv[2]), cfg))
        sys.exit()
    out, lines_stat, sent_stat, n_bp = build(cfg)
    total = sum(lines_stat.values())
    print(f"줄 {total:,}개: " + ", ".join(f"{k} {v:,} ({v / total:.1%})" for k, v in lines_stat.most_common()))
    print(f"본문 문장 {sent_stat['body']:,}개 중 정형 문구 {sent_stat['boilerplate']:,}개 뺌 (정형 문구 종류 {n_bp:,}개)")
    per = out.groupby("nid").size()
    print(f"v1 전처리 뒤 문장 {len(out):,}개, 리포트 {per.size:,}개")
    print("리포트당 문장 수:", per.describe()[["min", "25%", "50%", "75%", "max"]].round(1).to_dict())
    p = pd.read_parquet(PAIRS, columns=["r_nid", "p_nid", "period"])
    has = set(per.index)
    keep = p["r_nid"].isin(has) & p["p_nid"].isin(has)
    t = pd.DataFrame({"쌍": p.groupby("period").size(), "남는 쌍": keep.groupby(p["period"]).sum()})
    t["빠지는 비율"] = (1 - t["남는 쌍"] / t["쌍"]).round(4)
    print(t.to_string())
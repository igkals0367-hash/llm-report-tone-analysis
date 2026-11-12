"""Step 4: 텍스트 전처리 (7장). 문장 나누기 → 의견·목표가 문장 빼기 → 회사명 가리기 → 숫자 지우기 → 빈 문장 버리기."""
import pathlib
import re

import pandas as pd

from src.config import load_config

OPINION = re.compile(r"목표주가|목표가|투자의견|(?<![A-Za-z])TP(?![A-Za-z])")   # 7장 Step 4-2, 16장
JOSA = ["으로부터", "에서는", "으로는", "에게서", "에서", "으로", "부터", "까지", "보다", "처럼", "에게",
        "과의", "와의", "의", "은", "는", "이", "가", "을", "를", "에", "와", "과", "도", "로", "만"]  # 16장
MASK = "OO"
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")                                     # 3, 11.4, 300,000
WORD = "0-9A-Za-z가-힣"                                                     # 경계 판정에 쓰는 '글자'
BULLET = re.compile(r"^([-•·▪ㆍ*▶▷■□○●◆◇※\[]|[①-⑳]|\d+[).](?!\d)|\(\d+\))")      # 글머리표로 시작하는 줄
LINE_END = re.compile(r"[.?!:]$")                                           # 문장이 끝난 줄
LETTER = re.compile(r"[A-Za-z가-힣]")                                        # 빈 문장 판정: 글자가 하나라도 있나

def split_sentences(title, summary):
    """제목을 0번 문장으로 두고 요약을 문장으로 자른다 (7장 Step 4-1, 16장).
    줄바꿈은 앞 줄이 . ? ! :로 끝나거나 다음 줄이 글머리표로 시작할 때만 경계로 보고, 아니면 공백으로 잇는다."""
    out = [title.strip()] if title and title.strip() else []
    blocks, cur = [], ""
    for line in (summary or "").split("\n"):
        line = line.strip()
        if not line:
            continue
        if cur and (LINE_END.search(cur) or BULLET.match(line)):
            blocks.append(cur)
            cur = line
        else:
            cur = f"{cur} {line}" if cur else line
    if cur:
        blocks.append(cur)
    for b in blocks:
        out += [s.strip() for s in re.split(r"(?<=[.?!])\s+", b) if s.strip()]
    return out


def alias_pattern(aliases):
    """회사명 가리기 패턴 (7장 Step 4-3, 16장). 긴 이름부터, 앞뒤 경계를 지킬 때만."""
    names = "|".join(re.escape(n) for n in sorted(set(aliases), key=len, reverse=True))
    josa = "|".join(sorted(JOSA, key=len, reverse=True))
    return re.compile(rf"(?<![{WORD}])(?:{names})(?=$|[^{WORD}]|{josa})")


def clean_sentences(sentences, aliases):
    """7장 Step 4-2~5. aliases: 그 종목의 이름과 별명 목록. 남은 문장 목록을 돌려준다."""
    pattern = alias_pattern(aliases) if aliases else None
    out = []
    for s in sentences:
        if OPINION.search(s):                         # 2. 의견·목표가 문장은 통째로 뺀다
            continue
        if pattern:
            s = pattern.sub(MASK, s)                  # 3. 회사명 가리기
        s = NUMBER.sub("", s)                         # 4. 숫자 지우기
        s = re.sub(r"\s+", " ", s).strip()            #    숫자가 빠진 자리의 겹친 공백 정리
        if LETTER.search(s):                                          # 5. 빈 문장은 버린다
            out.append(s)
    return out


REPORTS = pathlib.Path("data/processed/reports.parquet")
PAIRS = pathlib.Path("data/processed/pairs.parquet")
ALIASES = pathlib.Path("configs/aliases.csv")
OUT = pathlib.Path("data/interim/sentences.parquet")


def build_sentences(cfg):
    """표본 쌍의 R·P 리포트를 전처리해 sentences.parquet로 저장한다 (7장 Step 4)."""
    pairs = pd.read_parquet(PAIRS)
    nids = set(pairs["r_nid"]) | set(pairs["p_nid"])
    r = pd.read_parquet(REPORTS, columns=["nid", "item_code", "item_name", "title", "summary_text"])
    aliases = (pd.read_csv(ALIASES, dtype={"item_code": str})
               .groupby("item_code")["alias"].apply(set).to_dict())
    for code, name in r[["item_code", "item_name"]].drop_duplicates().itertuples(index=False):
        aliases.setdefault(code, set()).add(name)      # 네이버가 그 종목코드에 붙인 이름 전부 (16장)
    r = r[r["nid"].isin(nids)]

    rows = []
    for nid, code, _, title, summary in r.itertuples(index=False):
        sents = clean_sentences(split_sentences(title, summary), aliases[code])
        rows += [{"nid": nid, "sent_idx": i, "text": s} for i, s in enumerate(sents)]
    out = pd.DataFrame(rows, columns=["nid", "sent_idx", "text"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    empty = nids - set(out["nid"])                    # 남은 문장이 하나도 없는 리포트 (7장 Step 4-5)
    lost = pairs["r_nid"].isin(empty) | pairs["p_nid"].isin(empty)
    return out, len(nids), empty, int(lost.sum())


if __name__ == "__main__":
    out, n, empty, lost = build_sentences(load_config())
    per = out.groupby("nid").size()
    print(f"리포트 {n:,}건 → 문장 {len(out):,}개")
    print("리포트당 문장 수:", per.describe()[["min", "25%", "50%", "75%", "max"]].to_dict())
    print(f"남은 문장이 없는 리포트 {len(empty)}건 → 빠질 쌍 {lost}개")
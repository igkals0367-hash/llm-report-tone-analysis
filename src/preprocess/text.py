"""Step 4: 텍스트 전처리 (7장). 문장 나누기 → 의견·목표가 문장 빼기 → 회사명 가리기 → 숫자 지우기 → 빈 문장 버리기."""
import re

OPINION_WORDS = ["목표주가", "목표가", "투자의견", "TP"]                    # 7장 Step 4-2
JOSA = ["으로부터", "에서는", "으로는", "에게서", "에서", "으로", "부터", "까지", "보다", "처럼", "에게",
        "과의", "와의", "의", "은", "는", "이", "가", "을", "를", "에", "와", "과", "도", "로", "만"]  # 16장
MASK = "OO"
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")                                     # 3, 11.4, 300,000
WORD = "0-9A-Za-z가-힣"                                                     # 경계 판정에 쓰는 '글자'


def split_sentences(title, summary):
    """제목을 0번 문장으로 두고, 요약을 줄바꿈과 '. ? ! 뒤 공백'에서 자른다 (7장 Step 4-1)."""
    out = [title.strip()] if title and title.strip() else []
    for line in (summary or "").split("\n"):
        out += [s.strip() for s in re.split(r"(?<=[.?!])\s+", line) if s.strip()]
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
        if any(w in s for w in OPINION_WORDS):        # 2. 의견·목표가 문장은 통째로 뺀다
            continue
        if pattern:
            s = pattern.sub(MASK, s)                  # 3. 회사명 가리기
        s = NUMBER.sub("", s)                         # 4. 숫자 지우기
        s = re.sub(r"\s+", " ", s).strip()            #    숫자가 빠진 자리의 겹친 공백 정리
        if s:                                         # 5. 빈 문장은 버린다
            out.append(s)
    return out
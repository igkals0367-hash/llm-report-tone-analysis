"""Step 2: 네이버 원본 리포트를 정리해 data/processed/reports.parquet 하나로 만든다.
사용: python -m src.preprocess.reports
"""
import html as html_lib
import json
import pathlib
import re

import pandas as pd

from src.config import load_config

RAW = pathlib.Path("data/raw/naver")
OUT = pathlib.Path("data/processed/reports.parquet")
OPINIONS = pathlib.Path("configs/opinions.csv")
HTML_TAGS = "p|br|div|span|strong|font|b"      # 실제로 쓰인 HTML 태그 7종 (16장)


def clean_html(text):
    """요약 HTML → 본문 텍스트 (5장 Step 2 ⑤)."""
    s = re.sub(r"</p\s*>|<br\s*/?>|</div\s*>", "\n", text, flags=re.I)  # 줄을 나누는 태그 → 줄바꿈
    s = re.sub(rf"</?(?:{HTML_TAGS})\b[^>]*>", "", s, flags=re.I)       # 7종 태그만 삭제
    s = html_lib.unescape(s).replace("\xa0", " ")                        # &amp; → &, &nbsp; → 공백
    lines = (line.strip() for line in s.split("\n"))
    return "\n".join(line for line in lines if line)                    # 빈 줄 버림


def load_raw(raw=RAW):
    rows = [json.loads(line) for f in sorted(raw.glob("*.jsonl")) for line in open(f, encoding="utf-8")]
    return pd.DataFrame(rows)


def build_reports(cfg):
    """raw jsonl → reports.parquet (18장). 정리된 표와 사유별 건수를 돌려준다."""
    raw = load_raw()

    # ② [AI] 리포트는 처음부터 뺀다
    is_ai = raw["title"].fillna("").str.startswith("[AI]")
    df = raw[~is_ai].copy()

    # ④ 의견: 순위표에 없는 표기가 나오면 멈춘다
    ranks = pd.read_csv(OPINIONS, encoding="utf-8", dtype={"opinion_text": str})
    unknown = set(df["opinionText"]) - set(ranks["opinion_text"])
    if unknown:
        raise ValueError(f"configs/opinions.csv에 없는 의견 표기: {unknown}")
    opinion_rank = df["opinionText"].map(dict(zip(ranks["opinion_text"], ranks["opinion_rank"]))).astype("Int64")
    opinion_group = df["opinionType"].where(opinion_rank.notna(), "none")   # 순위가 빈칸 → 의견 없음

    # ③ 목표가: 정수로, 없거나 0이면 빈값
    goal = pd.to_numeric(df["goalPrice"]).astype("Int64").replace(0, pd.NA)

    out = pd.DataFrame({
        "nid": df["nid"].astype("int64"),                                    # 글자 → 숫자 (정렬 때문에)
        "write_date": pd.to_datetime(df["writeDate"], format="%Y-%m-%d"),
        "broker_code": df["brokerCode"],
        "broker_name": df["brokerName"],
        "item_code": df["itemCode"],
        "item_name": df["itemName"],
        "opinion_group": opinion_group,
        "opinion_text": df["opinionText"],
        "opinion_rank": opinion_rank,
        "goal_price": goal,
        "title": df["title"],
        "summary_text": df["content"].fillna("").map(clean_html),            # ⑤
    })
    out["has_opinion"] = out["opinion_group"] != "none"
    out["has_goal_price"] = out["goal_price"].notna()
    out["has_summary"] = out["summary_text"] != ""
    out["is_neutral_ref"] = out["opinion_group"] == "neutral"                # 유니버스 조건은 Step 10에서

    # ① 게시일 순, 같은 날이면 nid가 작은 것 먼저
    out = out.sort_values(["write_date", "nid"], ignore_index=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    counts = {
        "원본": len(raw),
        "[AI] 제외": int(is_ai.sum()),
        "남은 리포트": len(out),
        "의견 없음 (표시)": int((~out["has_opinion"]).sum()),
        "목표가 없음 (표시)": int((~out["has_goal_price"]).sum()),
        "요약 없음 (표시)": int((~out["has_summary"]).sum()),
        "중립 참고 (표시)": int(out["is_neutral_ref"].sum()),
    }
    return out, counts


if __name__ == "__main__":
    _, counts = build_reports(load_config())
    for k, v in counts.items():
        print(f"{k:>16}: {v:,}")
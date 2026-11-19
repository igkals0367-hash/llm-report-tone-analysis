"""분류 고정 장치 (18장). 결과 데이터(H1·H2)와 실제 등급을 읽는 코드는 맨 처음 require_lock()을 부른다.
analysis_mode는 H1·H2가 어떤 글 출처의 등급으로 돌고 무엇으로 판정하는지 정한다 (19장)."""
import json
import pathlib

import pandas as pd

from src.config import source_path

CUTOFFS = pathlib.Path("models/cutoffs.json")
PAIRS = pathlib.Path("data/processed/pairs.parquet")
BONFERRONI = {"summary": 1, "pdf": 2}      # v2는 같은 결과 데이터에 두 번째로 도전하므로 alpha를 둘로 나눈다 (19장)


def require_lock(path=CUTOFFS):
    """cutoffs.json이 없으면 오류를 내고 멈춘다(로그인처럼: 있어야 통과). 있으면 그 내용을 돌려준다."""
    path = pathlib.Path(path)
    if not path.exists():
        raise RuntimeError(f"{path}이 없습니다. 분류를 고정(Step 8)하기 전에는 결과를 계산하지 않습니다 (18장).")
    return json.loads(path.read_text(encoding="utf-8"))


def analysis_mode(cfg, v1same=False):
    """H1·H2 실행 방식 (19장 검정). → {"cfg", "keep", "tag", "alpha", "label"}
    - 보통: 설정의 text_source 그대로. v1은 alpha 그대로, v2(pdf)는 alpha ÷ 2 = 0.025, 파일·기록 이름에 _pdf.
    - v1same: v1 등급·컷오프로, v2에 남은 쌍(pairs_pdf.parquet의 r_nid)만 계산한다. 참고 비교라 판정하지 않는다(alpha None).
      v2를 하는 중(text_source: pdf)에만 돈다."""
    source = cfg.get("text_source", "summary")
    if v1same:
        if source != "pdf":
            raise ValueError("같은 쌍 v1 비교는 text_source: pdf일 때만 돈다 (19장).")
        keep = set(pd.read_parquet(source_path(PAIRS, cfg), columns=["r_nid"])["r_nid"])
        return {"cfg": {**cfg, "text_source": "summary"}, "keep": keep, "tag": "_v1same", "alpha": None,
                "label": "같은 쌍 v1 비교: v1 등급, v2에 남은 쌍만 (참고, 판정에 쓰지 않음)"}
    source_path(PAIRS, cfg)                                      # text_source 값 확인
    return {"cfg": cfg, "keep": None, "tag": "_pdf" if source == "pdf" else "",
            "alpha": cfg["permutation"]["alpha"] / BONFERRONI[source],
            "label": "v2: 제목 + PDF 본문" if source == "pdf" else "v1: 제목 + 요약"}
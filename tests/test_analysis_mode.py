"""19장 검정: 글 출처에 따른 H1·H2 실행 방식 (파일 꼬리표, 판정 기준, 같은 쌍 v1 비교)."""
import pandas as pd
import pytest

from src.analysis.guard import analysis_mode

PM = {"permutation": {"alpha": 0.05}}


def test_v1_keeps_alpha_and_names():
    m = analysis_mode({**PM, "text_source": "summary"})
    assert (m["tag"], m["alpha"], m["keep"]) == ("", 0.05, None)


def test_v2_halves_alpha():                       # 본페로니: 0.05를 두 번으로 나눈다
    m = analysis_mode({**PM, "text_source": "pdf"})
    assert m["tag"] == "_pdf" and m["alpha"] == pytest.approx(0.025)


def test_v1same_needs_v2():
    with pytest.raises(ValueError):
        analysis_mode({**PM, "text_source": "summary"}, v1same=True)


def test_v1same_uses_v1_grades_on_v2_pairs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/processed").mkdir(parents=True)
    pd.DataFrame({"r_nid": [1, 3]}).to_parquet(tmp_path / "data/processed/pairs_pdf.parquet")
    m = analysis_mode({**PM, "text_source": "pdf"}, v1same=True)
    assert m["cfg"]["text_source"] == "summary"   # 등급·컷오프는 v1 파일
    assert m["keep"] == {1, 3} and m["alpha"] is None and m["tag"] == "_v1same"
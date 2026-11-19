"""19장 코드: 글 출처(text_source)에 따라 파일 이름만 바뀐다."""
from pathlib import Path

import pytest

from src.config import source_path


def test_summary_keeps_v1_name():                # v1 파일 이름은 그대로
    assert source_path("data/processed/tone.parquet", {"text_source": "summary"}) == Path("data/processed/tone.parquet")
    assert source_path("data/processed/tone.parquet", {}) == Path("data/processed/tone.parquet")   # 설정에 없으면 v1


def test_pdf_adds_suffix():
    assert source_path("data/interim/sentences.parquet", {"text_source": "pdf"}) == Path("data/interim/sentences_pdf.parquet")
    assert source_path("models/cutoffs.json", {"text_source": "pdf"}) == Path("models/cutoffs_pdf.json")


def test_unknown_source_stops():                 # 'PDF'처럼 잘못 적으면 v1 파일을 덮어쓰지 않게 멈춘다
    with pytest.raises(ValueError):
        source_path("data/processed/tone.parquet", {"text_source": "PDF"})
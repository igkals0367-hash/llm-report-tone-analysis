"""v2 Step 4b (19장): PDF 글 정리 규칙. 예시 문장은 지어낸 것이다."""
from src.preprocess.pdf_text import boilerplate, key, line_reason, split_pdf

CFG = {"max_symbol_ratio": 0.5, "min_line_hangul": 10, "boilerplate_min_reports": 2}


def test_line_reason():
    assert line_reason("매출액 1,234 1,456 1,678 12.3%", CFG) == "symbol"          # 표 줄 (숫자·기호 88%)
    assert line_reason("2024년 매출액은 1조 2,345억원(+12.3% YoY)으로 컨센서스를 상회했다.", CFG) is None  # 46%
    assert line_reason("Compliance Notice", CFG) == "short"                      # 한글 없음, 문장 끝 아님
    assert line_reason("우려된다.", CFG) is None                                   # 짧아도 문장 끝 조각은 남긴다
    assert line_reason("   ", CFG) == "empty"


def test_dropped_line_is_a_boundary():                                          # 빠진 줄 앞뒤를 잇지 않는다
    lines = ["하반기에는 원가 부담이 커질 수 있어 수익성 둔화가", "우려된다.", "영업이익 (%) 23 45 67",
             "신규 수주가 늘어 매출 성장이 이어질 전망이다"]
    head, body, reasons = split_pdf("제목입니다", lines, CFG)
    assert head == ["제목입니다"]
    assert body == ["하반기에는 원가 부담이 커질 수 있어 수익성 둔화가 우려된다.", "신규 수주가 늘어 매출 성장이 이어질 전망이다"]
    assert reasons == [None, None, "symbol", None]


def test_boilerplate_counts_reports_not_sentences():
    bodies = {1: ["본 자료는 2024년 작성되었다.", "고유 문장 하나"], 2: ["본 자료는 2025년 작성되었다."], 3: ["고유 문장 둘"]}
    bp = boilerplate(bodies, CFG)
    assert key("본 자료는 2024년 작성되었다.") in bp                               # 숫자만 다른 같은 문장이 리포트 2개에
    assert key("고유 문장 하나") not in bp
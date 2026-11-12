"""18장 전처리 예시: 7장 Step 4 규칙대로 구현됐는지 확인한다."""
from src.preprocess.text import clean_sentences, split_sentences


def test_example_hynix():                        # 18장 예시 그대로
    title = "SK하이닉스 HBM 수요는 견조"
    summary = ("3분기 영업이익은 11.4조원으로 컨센서스를 상회했다. "
               "투자의견 매수와 목표주가 300,000원을 유지한다. "
               "다만 SK하이닉스의 범용 DRAM 가격은 4분기부터 둔화될 전망이다.")
    got = clean_sentences(split_sentences(title, summary), ["SK하이닉스", "하이닉스"])
    assert got == [
        "OO HBM 수요는 견조",
        "분기 영업이익은 조원으로 컨센서스를 상회했다.",
        "다만 OO의 범용 DRAM 가격은 분기부터 둔화될 전망이다.",
    ]


def test_name_inside_longer_name():              # 18장: 뒤쪽 경계
    assert clean_sentences(["LG전자와 LG 모두 부진했다."], ["LG"]) == ["LG전자와 OO 모두 부진했다."]


def test_front_boundary():                       # 16장: 앞쪽 경계
    got = clean_sentences(["SKT의 가입자가 늘었다.", "KT의 가입자가 늘었다."], ["KT"])
    assert got == ["SKT의 가입자가 늘었다.", "OO의 가입자가 늘었다."]


def test_wrapped_lines_are_joined():             # 16장: 문장 중간 줄바꿈은 공백으로 잇는다
    summary = "영업손실은 150억원으로\n추정된다. 매출은\n3.5조원으로 늘었다."
    assert split_sentences("제목", summary) == [
        "제목", "영업손실은 150억원으로 추정된다.", "매출은 3.5조원으로 늘었다."]


def test_line_breaks_that_stay():                # 16장: 문장 끝·글머리표 앞의 줄바꿈은 경계
    summary = "WHAT'S THE STORY?\n실적이 좋다\n- 첫째 항목\n1) 둘째 항목"
    assert split_sentences("제목", summary) == [
        "제목", "WHAT'S THE STORY?", "실적이 좋다", "- 첫째 항목", "1) 둘째 항목"]


def test_tp_only_as_its_own_word():              # 16장: TPV, SOTP 속 'TP'는 의견 문장이 아니다
    got = clean_sentences(["TP 16,000원 유지", "TPV는 늘었다.", "SOTP 방식으로 평가", "TP를 상향"], [])
    assert got == ["TPV는 늘었다.", "SOTP 방식으로 평가"]


def test_nothing_left():                         # 7장 Step 4-5: 남는 문장이 없을 수 있다
    assert clean_sentences(["목표가 300,000원 유지", "2024"], ["삼성전자"]) == []


def test_no_letters_is_empty():                  # 16장: 숫자를 지운 뒤 글자가 없는 문장도 빈 문장
    got = clean_sentences(["1.", "첫째 항목이다.", "- 2024 = 3,000"], [])
    assert got == ["첫째 항목이다."]
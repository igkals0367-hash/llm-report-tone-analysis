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


def test_split_on_newline():                     # 7장 Step 4-1: 줄바꿈에서도 자른다
    assert split_sentences("제목", "첫 줄\n둘째 줄입니다. 셋째 문장") == ["제목", "첫 줄", "둘째 줄입니다.", "셋째 문장"]


def test_nothing_left():                         # 7장 Step 4-5: 남는 문장이 없을 수 있다
    assert clean_sentences(["목표가 300,000원 유지", "2024"], ["삼성전자"]) == []
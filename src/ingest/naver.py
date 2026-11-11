"""Step 1: 네이버페이 증권 리서치 종목 리포트 목록을 한 달씩 받아 원본 그대로 저장한다.
사용: python -m src.ingest.naver 2023-01 2026-11
"""
import calendar, json, pathlib, sys, time, urllib.parse, urllib.request

from src.config import load_config

API = "https://stock.naver.com/api/stockSecurity/researches/v2/company"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
SLEEP = load_config()["ingest"]["sleep_sec"]  # 요청 사이 쉬는 시간(초)
OUT = pathlib.Path("data/raw/naver")


def get(url):
    last = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:          # 잠깐 쉬었다가 다시 시도
            last = e
            time.sleep(2 * (attempt + 1))
    raise last


def fetch_month(year, month, out=OUT):
    """한 달치를 YYYY-MM.jsonl에 저장하고 (저장 건수, totalCount)를 돌려준다."""
    start = f"{year}-{month:02d}-01"
    end = f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{year}-{month:02d}.jsonl"
    saved, index, total = 0, 0, None
    with open(path, "w", encoding="utf-8") as f:
        while True:
            q = urllib.parse.urlencode({"index": index, "size": 50, "startDate": start, "endDate": end})
            d = get(f"{API}?{q}")
            total = (int(d["totalCount"]))
            for item in d.get("items") or []:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                saved += 1
            if not d.get("hasNext"):
                break
            index += 1
            time.sleep(SLEEP)
    return saved, total


def months(first, last):
    y, m = map(int, first.split("-"))
    y2, m2 = map(int, last.split("-"))
    while (y, m) <= (y2, m2):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


if __name__ == "__main__":
    for y, m in months(sys.argv[1], sys.argv[2]):
        saved, total = fetch_month(y, m)
        flag = "OK" if saved == total else "건수 불일치"
        print(f"{y}-{m:02d}: {saved}건 저장 (totalCount {total}) {flag}")
        time.sleep(SLEEP)
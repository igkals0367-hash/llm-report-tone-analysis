"""v2 Step 1b (19장): 네트워크 없이 확인할 수 있는 부분. 한글 세기, 이어받기, PDF가 없거나 받기에 실패할 때."""
import src.ingest.pdf as m

CFG = {"pdf": {"sleep_sec": 0, "retries": 0, "timeout_sec": 1, "min_hangul": 100}}


def test_hangul_count():                         # 깨진 글자 (cid:12)나 영어, 숫자는 세지 않는다
    assert m.hangul_count("매출 12% 증가 (cid:12) abc") == 4


def test_todo_skips_finished_and_retries_failed():
    manifest = [{"nid": "1", "status": "ok"}, {"nid": "2", "status": "download_failed"},
                {"nid": "3", "status": "no_url"}, {"nid": "4", "status": "download_failed"},
                {"nid": "4", "status": "ok"}]      # 4는 실패 뒤 다시 받아 성공: 마지막 줄을 따른다
    assert m.todo([1, 2, 3, 4, 5], manifest) == [2, 5]


def test_no_url_and_download_failed(monkeypatch):
    monkeypatch.setattr(m, "fetch", lambda url, cfg, as_json=False: {"attachUrl": None})
    assert m.process(1, CFG)["status"] == "no_url"

    def broken(url, cfg, as_json=False):
        raise OSError("network")
    monkeypatch.setattr(m, "fetch", broken)
    assert m.process(2, CFG)["status"] == "download_failed"


def test_latest_keeps_last_row_per_nid():        # --reextract가 manifest를 새로 쓸 때 쓴다
    rows = [{"nid": "1", "status": "download_failed"}, {"nid": "2", "status": "ok"}, {"nid": "1", "status": "ok"}]
    assert [(r["nid"], r["status"]) for r in m.latest(rows)] == [("1", "ok"), ("2", "ok")]
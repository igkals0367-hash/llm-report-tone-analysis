"""분류 고정 장치 (18장): 잠그기 전에는 require_lock이 멈추고, 잠근 뒤에는 lock이 덮어쓰지 않는다."""
import json

import pytest

from src.analysis.guard import require_lock
from src.classify.grade import lock


def test_require_lock_stops_without_file(tmp_path):
    with pytest.raises(RuntimeError):
        require_lock(tmp_path / "cutoffs.json")


def test_require_lock_returns_record(tmp_path):
    p = tmp_path / "cutoffs.json"
    p.write_text(json.dumps({"lower": -0.1, "upper": 0.1}), encoding="utf-8")
    assert require_lock(p)["upper"] == 0.1


def test_lock_refuses_to_overwrite(tmp_path):
    p = tmp_path / "cutoffs.json"
    p.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        lock({}, path=p)
    assert p.read_text(encoding="utf-8") == "{}"     # 원래 봉투는 그대로
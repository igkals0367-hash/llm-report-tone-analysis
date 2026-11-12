"""분류 고정 장치 (18장). 결과 데이터(H1·H2)와 실제 등급을 읽는 코드는 맨 처음 require_lock()을 부른다."""
import json
import pathlib

CUTOFFS = pathlib.Path("models/cutoffs.json")


def require_lock(path=CUTOFFS):
    """cutoffs.json이 없으면 오류를 내고 멈춘다(로그인처럼: 있어야 통과). 있으면 그 내용을 돌려준다."""
    path = pathlib.Path(path)
    if not path.exists():
        raise RuntimeError(f"{path}이 없습니다. 분류를 고정(Step 8)하기 전에는 결과를 계산하지 않습니다 (18장).")
    return json.loads(path.read_text(encoding="utf-8"))
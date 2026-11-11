"""설정 파일을 읽는다. 0장 규칙 7: 숫자는 코드에 쓰지 않고 configs/default.yaml에서 읽는다."""
from pathlib import Path

import yaml


def load_config(path="configs/default.yaml"):
    """YAML 설정 파일을 읽어 dict로 돌려준다."""
    with open(Path(path), encoding="utf-8") as f:
        return yaml.safe_load(f)
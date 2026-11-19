from pathlib import Path

import yaml

SOURCES = ("summary", "pdf")


def load_config(path="configs/default.yaml"):
    with open(Path(path), encoding="utf-8") as f:
        return yaml.safe_load(f)


def source_path(path, cfg):
    """글 출처에 따른 파일 이름 (19장 코드). text_source가 pdf이면 이름 끝에 _pdf를 붙이고, summary(v1)면 그대로 둔다.
    값을 잘못 적으면 v1 파일을 덮어쓸 수 있으므로 두 값 말고는 멈춘다."""
    source = cfg.get("text_source", "summary")
    if source not in SOURCES:
        raise ValueError(f"text_source는 {' 또는 '.join(SOURCES)}여야 합니다: {source!r}")
    path = Path(path)
    return path.with_name(f"{path.stem}_pdf{path.suffix}") if source == "pdf" else path
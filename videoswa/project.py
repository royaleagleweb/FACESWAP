"""Save and load a Videoswa project file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

PROJECT_SUFFIX = ".videoswaproj"
PROJECT_VERSION = 1


def gender_mark(gender: Optional[int]) -> str:
    if gender == 1:
        return "♂"
    if gender == 0:
        return "♀"
    return ""


def save_project(path: Path, payload: dict) -> Path:
    path = Path(path)
    if path.suffix.lower() != PROJECT_SUFFIX:
        path = path.with_suffix(PROJECT_SUFFIX)
    body = {"version": PROJECT_VERSION, **payload}
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return path


def load_project(path: Path) -> dict:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("That project file is not a Videoswa project.")
    if int(data.get("version", 0)) != PROJECT_VERSION:
        raise ValueError("This project was written by a different Videoswa version.")
    return data

"""The log file and the headless self-test entry point."""

from pathlib import Path

from faceswap.diag import configure_logging, log_file_path
from videoswa.selftest import run_selftest


def test_log_file_follows_videoswa_log_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VIDEOSWA_LOG_DIR", str(tmp_path))
    path = configure_logging()
    assert path == tmp_path / "videoswa.log"
    assert path.is_file()
    assert log_file_path() == path
    text = path.read_text(encoding="utf-8")
    assert "Log file" in text


def test_selftest_fails_when_the_files_are_missing(tmp_path: Path) -> None:
    code = run_selftest([str(tmp_path / "missing.jpg"), str(tmp_path / "missing.mp4")])
    assert code == 1

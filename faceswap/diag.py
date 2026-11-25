"""Log file, crash hooks, and the path Roy can open after a freeze.

On Windows the file is ``%LOCALAPPDATA%\\Videoswa\\videoswa.log``.
Set ``VIDEOSWA_LOG_DIR`` to point the file somewhere else (tests do this).
"""

from __future__ import annotations

import faulthandler
import logging
import os
import sys
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_CONFIGURED = False
_LOG_PATH: Optional[Path] = None
_FAULT_FILE = None


def log_file_path() -> Path:
    override = os.environ.get("VIDEOSWA_LOG_DIR")
    if override:
        return Path(override) / "videoswa.log"
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "Videoswa" / "videoswa.log"
    return Path.home() / ".local" / "share" / "Videoswa" / "videoswa.log"


def configure_logging() -> Path:
    """Attach a rotating DEBUG log. Safe to call more than once."""
    global _CONFIGURED, _LOG_PATH, _FAULT_FILE
    path = log_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _LOG_PATH = path
    if not _CONFIGURED:
        handler = RotatingFileHandler(
            path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        for name in ("faceswap", "videoswa", "qt"):
            log = logging.getLogger(name)
            log.setLevel(logging.DEBUG)
            log.addHandler(handler)
        _FAULT_FILE = open(path, "a", encoding="utf-8")
        if "pytest" not in sys.modules:
            faulthandler.enable(_FAULT_FILE, all_threads=True)
        logging.getLogger("faceswap").info("Log file: %s", path)
        _CONFIGURED = True
    return path


def install_exception_hooks() -> None:
    """Write uncaught errors and Qt messages to the log, and show a dialog."""
    configure_logging()
    sys.excepthook = _excepthook
    _install_qt_message_handler()


def _excepthook(exc_type, exc, tb) -> None:
    if exc_type is KeyboardInterrupt:
        sys.__excepthook__(exc_type, exc, tb)
        return
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    logging.getLogger("faceswap").error("Uncaught exception\n%s", text)
    _show_error_dialog(
        "Videoswa hit an error",
        f"{exc_type.__name__}: {exc}\n\nThe details are in:\n{log_file_path()}",
    )
    sys.__excepthook__(exc_type, exc, tb)


def _install_qt_message_handler() -> None:
    try:
        from PySide6.QtCore import QtMsgType, qInstallMessageHandler
    except Exception:
        return

    def _handler(mode, _context, message) -> None:
        level = logging.INFO
        if mode in (QtMsgType.QtWarningMsg,):
            level = logging.WARNING
        elif mode in (QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
            level = logging.ERROR
        elif mode == QtMsgType.QtDebugMsg:
            level = logging.DEBUG
        logging.getLogger("qt").log(level, "%s", message)
        if mode == QtMsgType.QtFatalMsg:
            _show_error_dialog(
                "Videoswa stopped",
                f"{message}\n\nThe details are in:\n{log_file_path()}",
            )

    qInstallMessageHandler(_handler)


def _show_error_dialog(title: str, message: str) -> None:
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        return
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
    except Exception:
        return
    if QApplication.instance() is None:
        return
    QMessageBox.critical(None, title, message)

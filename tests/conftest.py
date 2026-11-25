"""Shut Qt down before the interpreter tears the bindings down."""


def pytest_sessionfinish(session, exitstatus):
    del session, exitstatus
    try:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            return
        app.closeAllWindows()
        app.processEvents()
        app.quit()
    except Exception:
        return

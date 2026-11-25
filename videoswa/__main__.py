"""``python -m videoswa`` opens the window. ``--selftest`` runs headless."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from faceswap.diag import configure_logging, install_exception_hooks

    args = list(sys.argv[1:] if argv is None else argv)
    configure_logging()
    if args and args[0] == "--selftest":
        from videoswa.selftest import run_selftest

        return run_selftest(args[1:])
    if args and args[0] == "--bench":
        from videoswa.bench import run_bench

        return run_bench(args[1:])
    install_exception_hooks()
    from videoswa.desktop import main as desktop_main

    return desktop_main()


if __name__ == "__main__":
    sys.exit(main())

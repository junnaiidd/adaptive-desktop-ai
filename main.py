"""Application entry point for Adaptive Desktop AI."""

from __future__ import annotations

import platform
import sys


def main() -> int:
    """Start the minimal Windows application shell."""
    if platform.system() != "Windows":
        print("Adaptive Desktop AI currently targets Windows.")
        return 1

    try:
        from app.ui.application import run_application
    except ModuleNotFoundError as error:
        if error.name == "PySide6":
            print("PySide6 is required to launch the Adaptive Desktop AI interface.")
            return 1
        raise

    return run_application(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())

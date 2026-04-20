"""
Portable desktop launcher for Scientific Parser.

Double-click the built .exe: starts a local HTTP server and opens the UI in a browser.
Works without installing Python; requires internet for external APIs (Crossref, etc.).
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser


def _resource_root() -> str:
    """Directory where app files live (project folder or PyInstaller extract dir)."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return str(sys._MEIPASS)
    return os.path.dirname(os.path.abspath(__file__))


def _adjust_paths(root: str) -> None:
    """Ensure imports and relative paths (models/, data/) resolve when frozen."""
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)


def _serve(host: str, port: int) -> None:
    try:
        from waitress import serve as waitress_serve
    except ImportError as e:
        raise RuntimeError(
            "Для portable-сборки нужен пакет waitress. Установите: pip install waitress"
        ) from e

    # Import after path adjustment
    from app import app  # noqa: WPS433

    waitress_serve(app, host=host, port=port, threads=6)


def main() -> int:
    parser = argparse.ArgumentParser(description="Scientific Parser — локальный запуск")
    parser.add_argument("--host", default="127.0.0.1", help="Адрес привязки")
    parser.add_argument("--port", type=int, default=8765, help="Порт (по умолчанию 8765)")
    parser.add_argument("--no-browser", action="store_true", help="Не открывать браузер")
    args = parser.parse_args()

    root = _resource_root()
    _adjust_paths(root)

    # Writable DB next to .exe when frozen (not inside _MEIPASS)
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        db_dir = os.path.join(exe_dir, "ScientificParser_data")
        os.makedirs(db_dir, exist_ok=True)
        os.environ.setdefault("SP_APP_HOME", db_dir)

    url = f"http://{args.host}:{args.port}/"

    if not args.no_browser:

        def _open_browser() -> None:
            webbrowser.open(url)

        threading.Timer(1.2, _open_browser).start()

    print(f"Scientific Parser: {url}")
    print("Закройте это окно, чтобы остановить сервер.")
    try:
        _serve(args.host, args.port)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

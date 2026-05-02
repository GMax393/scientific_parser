"""
Локальный запуск Scientific Parser без установки Python.

Собранный .exe:
  • Интерфейс — отдельное окно на вашем ПК (не страница в интернете).
  • Обработка страниц и ML выполняется локально на компьютере.
  • При наличии интернета выполняются запросы к внешним каталогам статей (Crossref, OpenAlex и т.д.).

По умолчанию на Windows используется встроенное окно (Microsoft WebView2).
Опции: --browser (внешний браузер), --no-browser (только консоль).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser

LOGGER = logging.getLogger(__name__)


def _resource_root() -> str:
    """Каталог с кодом и моделями (папка проекта или временная распаковка PyInstaller)."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return str(sys._MEIPASS)
    return os.path.dirname(os.path.abspath(__file__))


def _adjust_paths(root: str) -> None:
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

    from app import app  # noqa: WPS433

    waitress_serve(app, host=host, port=port, threads=6)


def _wait_server_ready(base_url: str, timeout_sec: float = 30.0) -> bool:
    health = base_url.rstrip("/") + "/health"
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(health, timeout=0.8)
            return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.12)
    return False


def _embedded_window_possible() -> bool:
    try:
        import webview  # noqa: F401

        return True
    except ImportError:
        return False


def _run_embedded_window(url: str, title: str) -> None:
    import webview

    webview.create_window(title, url, width=1180, height=820, min_size=(720, 520))
    webview.start()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scientific Parser — локальное приложение (обработка на этом компьютере)"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Адрес привязки сервера")
    parser.add_argument("--port", type=int, default=8765, help="Порт (по умолчанию 8765)")
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Открыть интерфейс во внешнем браузере вместо встроенного окна",
    )
    parser.add_argument("--no-browser", action="store_true", help="Не открывать ни окно, ни браузер")

    args = parser.parse_args()

    root = _resource_root()
    _adjust_paths(root)

    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        db_dir = os.path.join(exe_dir, "ScientificParser_data")
        os.makedirs(db_dir, exist_ok=True)
        os.environ.setdefault("SP_APP_HOME", db_dir)

    url = f"http://{args.host}:{args.port}/"
    host = args.host
    port = args.port

    use_embedded = (
        not args.no_browser
        and not args.browser
        and sys.platform.startswith("win")
        and _embedded_window_possible()
    )

    if use_embedded:

        def _worker() -> None:
            try:
                _serve(host, port)
            except Exception:
                LOGGER.exception("Ошибка сервера Waitress")

        threading.Thread(target=_worker, daemon=True).start()

        win_title = "Scientific Parser — локальная версия"

        if not _wait_server_ready(url):
            print("Не удалось запустить локальный сервер. Проверьте, свободен ли порт", port)
            return 1

        print(win_title)
        print("Открыто отдельное окно на этом компьютере (данные не отправляются «на сайт сервиса»).")
        print(f"Локальный адрес процесса: {url}")
        print("Закройте окно приложения или нажмите Ctrl+C в консоли.")
        try:
            _run_embedded_window(url, win_title)
        except KeyboardInterrupt:
            return 0
        return 0

    if args.no_browser:
        print(f"Scientific Parser: {url}")
        print("Откройте адрес в браузере вручную. Это окно не закрывайте — иначе сервер остановится.")
        try:
            _serve(host, port)
        except KeyboardInterrupt:
            return 0
        return 0

    def _open_browser() -> None:
        webbrowser.open(url)

    threading.Timer(1.0, _open_browser).start()

    print(f"Scientific Parser (внешний браузер): {url}")
    print("Закройте это окно консоли, чтобы остановить сервер.")
    try:
        _serve(host, port)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

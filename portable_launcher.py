"""
Локальный запуск BiblioParser без установки Python.

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
import traceback
import urllib.error
import urllib.request
import webbrowser

LOGGER = logging.getLogger(__name__)


def _say(msg: str) -> None:
    """Сообщение в консоль с flush — иначе при exe чёрный экран «молчит» десятки секунд."""
    print(msg, flush=True)


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
        description="BiblioParser — локальное приложение (обработка на этом компьютере)"
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
    _frozen = getattr(sys, "frozen", False)

    if _frozen:
        _say("")
        _say("=" * 60)
        _say("  BiblioParser — локальная версия")
        _say("=" * 60)
        _say("Первый запуск может занять 1–2 минуты: загружаются библиотеки (pandas, ML…).")
        _say("Это НОРМАЛЬНО, если сейчас «ничего не происходит» — дождитесь текста ниже.")
        _say("")

    if _frozen:
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        db_dir = os.path.join(exe_dir, "ScientificParser_data")
        os.makedirs(db_dir, exist_ok=True)
        os.environ.setdefault("SP_APP_HOME", db_dir)

    # Сразу проверить импорт приложения — иначе ошибка теряется в фоновом потоке Waitress.
    if _frozen:
        _say("Загрузка модулей приложения, подождите…")
    try:
        import app  # noqa: F401, WPS433
    except Exception:
        print("=== Ошибка при загрузке приложения (import app) ===", file=sys.stderr)
        traceback.print_exc()
        print(
            "\nЕсли путь к .exe содержит необычные символы — скопируйте exe в папку "
            "вроде C:\\Apps\\ScientificParser и запустите оттуда.",
            file=sys.stderr,
        )
        if _frozen:
            _say("")
            _say("-" * 60)
            _say("ОШИБКА: прочтите красный/чёрный текст выше. Затем нажмите Enter, чтобы закрыть.")
            _say("-" * 60)
            input()
        return 1
    if _frozen:
        _say("Модули загружены OK.")
        _say("")

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
                print("=== Ошибка сервера Waitress ===", file=sys.stderr)
                traceback.print_exc()

        threading.Thread(target=_worker, daemon=True).start()

        win_title = "BiblioParser — локальная версия"

        if _frozen:
            _say(f"Запуск встроенного сервера на порту {port}, жду ответ /health (до ~90 с)…")
            _say("(Если антивирус проверяет exe — это может занять время.)")

        if not _wait_server_ready(url, timeout_sec=90.0):
            _say(
                f"Не удалось дождаться ответа /health (порт {port}). "
                "Порт занят или ошибка сервера — см. текст выше."
            )
            if _frozen:
                _say("Нажмите Enter, чтобы закрыть это окно.")
                input()
            return 1

        _say("")
        _say(win_title)
        _say("Открываю окно приложения. Если его не видно — проверьте панель задач (может быть позади других окон).")
        _say(f"Локальный адрес: {url}")
        _say("Закройте окно приложения или это консольное окно (Ctrl+C), чтобы завершить работу.")
        _say("")
        try:
            _run_embedded_window(url, win_title)
        except KeyboardInterrupt:
            return 0
        except Exception:
            print("=== Ошибка встроенного окна (WebView2 / pywebview) ===", file=sys.stderr)
            traceback.print_exc()
            print(
                "\nЗапустите с открытием в обычном браузере:\n"
                "  ScientificParser.exe --browser\n"
                "или установите WebView2 Runtime с сайта Microsoft.",
                file=sys.stderr,
            )
            if _frozen:
                _say("Нажмите Enter, чтобы закрыть окно после прочтения сообщения об ошибке.")
                input()
            return 1
        return 0

    if args.no_browser:
        print(f"BiblioParser: {url}")
        print("Откройте адрес в браузере вручную. Это окно не закрывайте — иначе сервер остановится.")
        try:
            _serve(host, port)
        except KeyboardInterrupt:
            return 0
        return 0

    def _open_browser() -> None:
        webbrowser.open(url)

    threading.Timer(1.0, _open_browser).start()

    print(f"BiblioParser (внешний браузер): {url}")
    print("Закройте это окно консоли, чтобы остановить сервер.")
    try:
        _serve(host, port)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Тонкий клиент Scientific Parser: только окно + HTTP к вашему серверу.
Зависимости: Python 3.10+ (tkinter), без requests — urllib из стандартной библиотеки.

Сборка exe (Windows), из каталога scientific_parser:
  pip install pyinstaller
  pyinstaller --noconfirm --onefile --windowed --name ScientificParserThin desktop_client/sp_desktop.py
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from tkinter import END, Button, Entry, Frame, Label, Radiobutton, StringVar, Text, Tk, messagebox, scrolledtext


def _config_file() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "sp_thin_config.json"
    return Path(__file__).resolve().parent / "sp_thin_config.json"


def load_base_url() -> str:
    p = _config_file()
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return (data.get("api_base") or "").strip()
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return ""


def save_base_url(url: str) -> None:
    p = _config_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"api_base": url.strip()}, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_metadata(base: str, mode: str, value: str, timeout: int = 120) -> dict:
    base = base.rstrip("/")
    params: dict[str, str] = {}
    if mode == "doi":
        params["doi"] = value
    elif mode == "url":
        params["url"] = value
    else:
        params["title"] = value
    q = urllib.parse.urlencode(params)
    url = f"{base}/api/work?{q}"
    req = urllib.request.Request(url, headers={"User-Agent": "ScientificParserThin/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body)


def format_metadata_human(d: dict) -> str:
    lines: list[str] = []
    if d.get("title"):
        lines.append(d["title"])
    authors = d.get("authors") or []
    if authors:
        lines.append("Авторы: " + ", ".join(str(a) for a in authors[:20]))
    bits = [d.get("journal"), d.get("year"), d.get("volume"), d.get("issue"), d.get("pages")]
    bits = [str(b) for b in bits if b]
    if bits:
        lines.append("Издание: " + ", ".join(bits))
    if d.get("doi"):
        lines.append("DOI: " + str(d["doi"]))
    if d.get("source_url"):
        lines.append("URL: " + str(d["source_url"]))
    if d.get("pdf_url"):
        lines.append("PDF: " + str(d["pdf_url"]))
    return "\n".join(lines) if lines else "(нет полей)"


class App:
    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("Scientific Parser (тонкий клиент)")
        self.root.minsize(520, 420)

        top = Frame(self.root)
        top.pack(fill="x", padx=8, pady=6)

        Label(top, text="URL API (ваш сервер):").pack(anchor="w")
        self.entry_base = Entry(top)
        self.entry_base.pack(fill="x", pady=2)
        self.entry_base.insert(0, load_base_url() or "https://ваш-домен")

        mode_row = Frame(self.root)
        mode_row.pack(fill="x", padx=8, pady=4)
        self.mode = StringVar(value="doi")
        for text, val in (("DOI", "doi"), ("URL", "url"), ("Название", "title")):
            Radiobutton(mode_row, text=text, variable=self.mode, value=val).pack(side="left", padx=4)

        Label(self.root, text="Запрос:").pack(anchor="w", padx=8)
        self.entry_query = Entry(self.root)
        self.entry_query.pack(fill="x", padx=8, pady=2)

        btn_row = Frame(self.root)
        btn_row.pack(fill="x", padx=8, pady=6)
        Button(btn_row, text="Запросить", command=self.on_run).pack(side="left")
        Button(btn_row, text="Сохранить URL сервера", command=self.on_save_base).pack(side="left", padx=8)

        Label(self.root, text="Ответ:").pack(anchor="w", padx=8)
        self.out = scrolledtext.ScrolledText(self.root, height=16, wrap="word", font=("Consolas", 10))
        self.out.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def on_save_base(self) -> None:
        save_base_url(self.entry_base.get().strip())
        messagebox.showinfo("Сохранено", str(_config_file()))

    def on_run(self) -> None:
        base = self.entry_base.get().strip()
        q = self.entry_query.get().strip()
        mode = self.mode.get()
        if not base or "ваш-домен" in base:
            messagebox.showwarning("Нужен URL", "Укажите реальный URL сервера (https://…).")
            return
        if not q:
            messagebox.showwarning("Пусто", "Введите DOI, URL или название.")
            return
        self.out.delete("1.0", END)
        self.out.insert(END, "Загрузка…\n")
        self.root.update_idletasks()
        try:
            data = fetch_metadata(base, mode, q)
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = str(e)
            self.out.delete("1.0", END)
            self.out.insert(END, f"HTTP {e.code}\n{err_body}")
            return
        except Exception as e:
            self.out.delete("1.0", END)
            self.out.insert(END, f"Ошибка: {e!r}")
            return

        self.out.delete("1.0", END)
        if not data.get("ok"):
            self.out.insert(END, json.dumps(data, ensure_ascii=False, indent=2))
            return
        meta = data.get("metadata") or {}
        self.out.insert(END, format_metadata_human(meta) + "\n\n--- JSON ---\n")
        self.out.insert(END, json.dumps(data, ensure_ascii=False, indent=2))

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    App().run()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Telegram-бот для BiblioParser.

  set TELEGRAM_BOT_TOKEN=...
  set SP_PUBLIC_URL=http://127.0.0.1:5000   (для справки в /help)
  python telegram_bot.py

Команды: /start, /help, /doi, /url, /title
Также: свободный текст (DOI / URL / длинное название), несколько строк — пакетный режим.
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
import time
from dataclasses import asdict
from html import escape as html_escape
from typing import Any, Dict, List, Optional

import requests

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
BASE = (os.getenv("SP_PUBLIC_URL") or "http://127.0.0.1:5000").rstrip("/")

BATCH_MAX_LINES = int(os.getenv("TELEGRAM_BATCH_MAX", "12"))
BATCH_PAUSE_SEC = float(os.getenv("TELEGRAM_BATCH_PAUSE", "1.8"))
SESSION_TTL_SEC = 3600
MAX_LINE_LEN_FOR_BATCH = 280

TG_API = "https://api.telegram.org"

try:
    from inference_pipeline import (
        DOI_RE,
        PaperMetadata,
        export_bibtex,
        export_ris,
        extract_metadata_from_doi,
        extract_metadata_from_title,
        extract_metadata_from_url,
        format_citation,
        search_metadata_candidates_from_title,
    )
    from net_security import is_public_http_url
except ImportError as _imp_err:
    raise SystemExit(
        "Запустите бот из каталога scientific_parser с установленными зависимостями "
        "(нужны inference_pipeline и net_security): "
        + str(_imp_err)
    ) from _imp_err

_sessions_lock = threading.Lock()
_sessions: Dict[str, Dict[str, Any]] = {}

WELCOME_HTML = """🎓 <b>BiblioParser</b>

Я помогаю собрать метаданные статьи: авторы, журнал, год, DOI, ссылка на PDF (если есть).

<b>Команды</b>
• <code>/doi</code> — DOI, пример: <code>/doi 10.1038/s41586-020-2649-2</code>
• <code>/url</code> — ссылка на страницу статьи
• <code>/title</code> — поиск по названию (несколько вариантов — кнопка «Следующий вариант»)

<b>Без команд</b>
• Одна строка: похоже на URL, DOI или длинное название — обработаю автоматически
• <b>Несколько строк</b> (до """ + str(BATCH_MAX_LINES) + """): каждая строка — отдельный DOI, URL или название; между запросами пауза ~""" + str(
    int(BATCH_PAUSE_SEC)
) + """ с

Под результатом — кнопки: <b>PDF</b>, <b>цитата</b>, <b>BibTeX</b>, <b>RIS</b>.

Сложные страницы удобнее открыть в веб-интерфейсе: """ + html_escape(
    BASE
)


def _post(token: str, method: str, payload: dict) -> dict:
    r = requests.post(f"{TG_API}/bot{token}/{method}", json=payload, timeout=120)
    try:
        return r.json()
    except Exception:
        return {"ok": False, "description": r.text[:500]}


def _answer_cb(token: str, cq_id: str, text: str = "") -> None:
    body: Dict[str, Any] = {"callback_query_id": cq_id}
    if text:
        body["text"] = text[:200]
        body["show_alert"] = False
    _post(token, "answerCallbackQuery", body)


def _reply_html(token: str, chat_id: int, text: str) -> None:
    _post(
        token,
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": text[:4096],
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        },
    )


def _session_create(papers_as_dicts: List[Dict[str, Any]]) -> str:
    sid = secrets.token_urlsafe(8)[:12].replace("-", "x")
    with _sessions_lock:
        _sessions[sid] = {
            "t": time.time(),
            "papers": papers_as_dicts,
            "idx": 0,
            "message_id": None,
            "chat_id": None,
        }
    return sid


def _session_touch(sid: str) -> Optional[Dict[str, Any]]:
    with _sessions_lock:
        now = time.time()
        dead = [k for k, v in _sessions.items() if now - v.get("t", 0) > SESSION_TTL_SEC]
        for k in dead:
            del _sessions[k]
        s = _sessions.get(sid)
        if not s:
            return None
        s["t"] = now
        return s


def _session_set_msg(sid: str, chat_id: int, message_id: int) -> None:
    with _sessions_lock:
        if sid in _sessions:
            _sessions[sid]["chat_id"] = chat_id
            _sessions[sid]["message_id"] = message_id


def _paper_from_dict(d: Dict[str, Any]) -> PaperMetadata:
    return PaperMetadata(
        title=d.get("title"),
        authors=d.get("authors"),
        year=d.get("year"),
        journal=d.get("journal"),
        volume=d.get("volume"),
        issue=d.get("issue"),
        pages=d.get("pages"),
        doi=d.get("doi"),
        pdf_url=d.get("pdf_url"),
        source_url=d.get("source_url"),
        enriched_by=d.get("enriched_by"),
        confidence=d.get("confidence"),
        search_score=d.get("search_score"),
        matched_query=d.get("matched_query"),
    )


def _format_one_paper_html(d: Dict[str, Any], *, header: str) -> str:
    lines: List[str] = [f"{header}"]
    conf = d.get("confidence") or {}
    if float(conf.get("doi_invalid") or 0) >= 1.0:
        lines.append("⚠️ <i>Строка DOI не похожа на стандартный формат — проверьте ввод.</i>")
    if float(conf.get("retracted_openalex") or 0) >= 1.0:
        lines.append("🛑 <i>OpenAlex: возможна рестракция — проверьте у издателя.</i>")

    if d.get("title"):
        lines.append(f"📌 <b>{html_escape(str(d['title']))}</b>")

    authors = d.get("authors") or []
    if authors:
        au = ", ".join(str(a) for a in authors[:15])
        if len(authors) > 15:
            au += "…"
        lines.append(f"👥 {html_escape(au)}")

    bib_parts = []
    if d.get("journal"):
        bib_parts.append(html_escape(str(d["journal"])))
    if d.get("year"):
        bib_parts.append(html_escape(str(d["year"])))
    if d.get("volume"):
        bib_parts.append("т. " + html_escape(str(d["volume"])))
    if d.get("issue"):
        bib_parts.append("№ " + html_escape(str(d["issue"])))
    if d.get("pages"):
        bib_parts.append("с. " + html_escape(str(d["pages"])))
    if bib_parts:
        lines.append("📰 " + ", ".join(bib_parts))

    if d.get("doi"):
        lines.append(f"🔗 DOI: <code>{html_escape(str(d['doi']))}</code>")

    if d.get("source_url"):
        lines.append(f"🌐 <a href=\"{html_escape(str(d['source_url']), quote=True)}\">страница / DOI</a>")

    if d.get("pdf_url"):
        lines.append(f"📄 <a href=\"{html_escape(str(d['pdf_url']), quote=True)}\">PDF (ссылка)</a>")

    try:
        cite = format_citation(_paper_from_dict(d), style="gost")
        lines.append("")
        lines.append("📚 <b>Цитата (ГОСТ-подобно)</b>")
        lines.append(f"<pre>{html_escape(cite)}</pre>")
    except Exception:
        pass

    return "\n".join(lines)


def _inline_kb(sid: str, paper: Dict[str, Any], n_variants: int, idx: int) -> Dict[str, Any]:
    rows: List[List[Dict[str, str]]] = []
    if paper.get("pdf_url"):
        rows.append([{"text": "📄 Открыть PDF", "url": str(paper["pdf_url"])}])
    rows.append(
        [
            {"text": "📋 Цитата", "callback_data": f"c:{sid}"},
            {"text": "BibTeX", "callback_data": f"b:{sid}"},
            {"text": "RIS", "callback_data": f"r:{sid}"},
        ]
    )
    if n_variants > 1:
        nxt = (idx + 1) % n_variants
        rows.append(
            [{"text": f"🔁 След. вариант ({nxt + 1}/{n_variants})", "callback_data": f"n:{sid}"}]
        )
    return {"inline_keyboard": rows}


def _deliver_papers(
    token: str,
    chat_id: int,
    papers: List[PaperMetadata],
    header: str,
) -> None:
    if not papers:
        return
    dicts = [asdict(p) for p in papers]
    sid = _session_create(dicts)
    idx = 0
    text = _format_one_paper_html(dicts[idx], header=header)
    kb = _inline_kb(sid, dicts[idx], len(dicts), idx)
    res = _post(
        token,
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": text[:4096],
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
            "reply_markup": kb,
        },
    )
    if res.get("ok") and res.get("result", {}).get("message_id"):
        _session_set_msg(sid, chat_id, int(res["result"]["message_id"]))
    elif not res.get("ok"):
        LOGGER.warning("sendMessage: %s", res)


def _extract_doi(line: str) -> Optional[str]:
    m = DOI_RE.search(line)
    return m.group(0) if m else None


def _classify_line(line: str) -> str:
    low = line.strip().lower()
    if low.startswith("http://") or low.startswith("https://"):
        return "url"
    if _extract_doi(line):
        return "doi"
    return "title"


def _word_count(s: str) -> int:
    return len([w for w in s.replace("\n", " ").split() if w.strip()])


def _has_intraword_caps(s: str) -> bool:
    """Слова вроде NumPy, LaTeX — типичны для названий статей, редки в бытовом чате."""
    for w in s.split():
        if len(w) < 4:
            continue
        core = w.strip(".,;:\"'()[]{}")
        if any(ch.isupper() for ch in core[1:]):
            return True
    return False


def _looks_like_article_title(s: str) -> bool:
    if len(s) < 22:
        return False
    wc = _word_count(s)
    if wc < 3:
        return False
    if len(s) >= 48 and wc >= 4:
        return True
    if wc >= 6 and len(s) >= 32:
        return True
    if wc >= 4 and len(s) >= 28 and (
        _has_intraword_caps(s)
        or any(sep in s for sep in (":", "—", "–", " - "))
        or any(len(w.strip(".,;:\"'()[]{}")) > 11 for w in s.split())
    ):
        return True
    return False


def _is_batch_candidate(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2 or len(lines) > BATCH_MAX_LINES:
        return False
    if any(len(ln) > MAX_LINE_LEN_FOR_BATCH for ln in lines):
        return False
    return True


def _resolve_title_candidates(title: str) -> List[PaperMetadata]:
    cands = search_metadata_candidates_from_title(title, max_results=4)
    if not cands:
        p = extract_metadata_from_title(title)
        if p.title or p.doi:
            return [p]
        return []
    return cands


def _handle_batch(token: str, chat_id: int, raw: str) -> None:
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    lines = lines[:BATCH_MAX_LINES]
    _reply_html(
        token,
        chat_id,
        f"📚 Пакет: <b>{len(lines)}</b> строк. Пауза ~{BATCH_PAUSE_SEC} с между запросами.",
    )
    for i, line in enumerate(lines):
        kind = _classify_line(line)
        label = f"📎 {i + 1}/{len(lines)}"
        try:
            if kind == "url":
                if not is_public_http_url(line):
                    _reply_html(token, chat_id, f"{label} — пропуск: URL не прошёл проверку безопасности.")
                    continue
                p = extract_metadata_from_url(line)
                _deliver_papers(token, chat_id, [p], header=f"{label} · URL")
            elif kind == "doi":
                doi = _extract_doi(line) or line.strip()
                p = extract_metadata_from_doi(doi)
                _deliver_papers(token, chat_id, [p], header=f"{label} · DOI")
            else:
                cands = _resolve_title_candidates(line)
                if not cands:
                    _reply_html(
                        token,
                        chat_id,
                        f"{label} — не найдено: <i>{html_escape(line[:120])}</i>",
                    )
                else:
                    _deliver_papers(token, chat_id, cands, header=f"{label} · название")
        except Exception as e:
            LOGGER.exception("batch line %s", i)
            _reply_html(token, chat_id, f"{label} — ошибка: <code>{html_escape(str(e)[:400])}</code>")
        if i < len(lines) - 1:
            time.sleep(BATCH_PAUSE_SEC)


def _smart_single(token: str, chat_id: int, text: str) -> None:
    s = text.strip()
    low = s.lower()
    if low.startswith("http://") or low.startswith("https://"):
        if not is_public_http_url(s):
            _reply_html(token, chat_id, "URL не прошёл проверку безопасности (SSRF).")
            return
        _reply_html(token, chat_id, "⏳ Загружаю страницу…")
        p = extract_metadata_from_url(s)
        _deliver_papers(token, chat_id, [p], header="📎 Результат по URL")
        return

    doi = _extract_doi(s)
    if doi and len(s) < 90 and _word_count(s) <= 6:
        _handle_doi(token, chat_id, doi)
        return

    if _looks_like_article_title(s):
        _reply_html(token, chat_id, "⏳ Ищу по названию…")
        cands = _resolve_title_candidates(s)
        if not cands:
            _reply_html(token, chat_id, "Не удалось найти запись. Уточните название или пришлите DOI/URL.")
            return
        _deliver_papers(token, chat_id, cands, header="📎 Результат по названию")
        return

    _reply_html(
        token,
        chat_id,
        "Не уверен, как обработать сообщение. Используйте команды из <code>/help</code> "
        "или пришлите отдельной строкой DOI, https://… или полное название статьи.",
    )


def _handle_doi(token: str, chat_id: int, doi: str) -> None:
    doi = (doi or "").strip()
    if not doi:
        _reply_html(token, chat_id, "После <code>/doi</code> укажите идентификатор, например: <code>/doi 10.1038/...</code>")
        return
    try:
        p = extract_metadata_from_doi(doi)
        _deliver_papers(token, chat_id, [p], header="📎 Результат по DOI")
    except Exception as e:
        LOGGER.exception("doi")
        _reply_html(token, chat_id, f"Ошибка: <code>{html_escape(str(e))}</code>")


def _handle_url(token: str, chat_id: int, url: str) -> None:
    url = (url or "").strip()
    if not url:
        _reply_html(token, chat_id, "Пример: <code>/url https://journal.ru/article/...</code>")
        return
    if not is_public_http_url(url):
        _reply_html(token, chat_id, "URL не прошёл проверку безопасности (SSRF).")
        return
    try:
        _reply_html(token, chat_id, "⏳ Загружаю страницу…")
        p = extract_metadata_from_url(url)
        _deliver_papers(token, chat_id, [p], header="📎 Результат по URL")
    except Exception as e:
        LOGGER.exception("url")
        _reply_html(token, chat_id, f"Ошибка: <code>{html_escape(str(e))}</code>")


def _handle_title(token: str, chat_id: int, title: str) -> None:
    title = (title or "").strip()
    if len(title) < 4:
        _reply_html(token, chat_id, "Укажите осмысленное название после <code>/title</code>.")
        return
    try:
        _reply_html(token, chat_id, "⏳ Ищу варианты…")
        cands = _resolve_title_candidates(title)
        if not cands:
            _reply_html(token, chat_id, "Ничего не найдено. Уточните или попробуйте <code>/doi</code>.")
            return
        _deliver_papers(token, chat_id, cands, header="📎 Результат по названию")
    except Exception as e:
        LOGGER.exception("title")
        _reply_html(token, chat_id, f"Ошибка: <code>{html_escape(str(e))}</code>")


def _handle_callback(token: str, upd: Dict[str, Any]) -> None:
    cq = upd.get("callback_query") or {}
    cq_id = cq.get("id")
    data = (cq.get("data") or "").strip()
    msg = cq.get("message") or {}
    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    message_id = msg.get("message_id")
    if not cq_id or not data or not chat_id:
        return

    parts = data.split(":", 1)
    if len(parts) != 2:
        _answer_cb(token, cq_id, "Некорректные данные")
        return
    kind, sid = parts[0], parts[1]

    sess = _session_touch(sid)
    if not sess:
        _answer_cb(token, cq_id, "Сессия устарела — запросите снова")
        return

    papers: List[Dict[str, Any]] = sess.get("papers") or []
    if not papers:
        _answer_cb(token, cq_id, "Нет данных")
        return

    if kind == "n":
        idx = int(sess.get("idx") or 0)
        idx = (idx + 1) % len(papers)
        sess["idx"] = idx
        d = papers[idx]
        text = _format_one_paper_html(d, header=f"📎 Вариант {idx + 1} из {len(papers)}")
        kb = _inline_kb(sid, d, len(papers), idx)
        res = _post(
            token,
            "editMessageText",
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": text[:4096],
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
                "reply_markup": kb,
            },
        )
        if not res.get("ok"):
            LOGGER.warning("editMessageText: %s", res)
        _answer_cb(token, cq_id, "Вариант обновлён")
        return

    idx = int(sess.get("idx") or 0)
    d = papers[idx]
    meta = _paper_from_dict(d)

    if kind == "c":
        try:
            cite = format_citation(meta, style="gost")
        except Exception as e:
            _answer_cb(token, cq_id, str(e)[:180])
            return
        _reply_html(token, chat_id, "📋 <b>Цитата</b> — скопируйте текст ниже:\n<pre>" + html_escape(cite) + "</pre>")
        _answer_cb(token, cq_id, "Сообщение с цитатой отправлено")
        return

    if kind == "b":
        try:
            bib = export_bibtex([meta])
        except Exception as e:
            _answer_cb(token, cq_id, str(e)[:180])
            return
        chunk = bib[:4000]
        if len(bib) > 4000:
            chunk += "\n…"
        _reply_html(token, chat_id, "<b>BibTeX</b>\n<pre>" + html_escape(chunk) + "</pre>")
        _answer_cb(token, cq_id, "BibTeX в чате")
        return

    if kind == "r":
        try:
            ris = export_ris([meta])
        except Exception as e:
            _answer_cb(token, cq_id, str(e)[:180])
            return
        chunk = ris[:4000]
        if len(ris) > 4000:
            chunk += "\n…"
        _reply_html(token, chat_id, "<b>RIS</b>\n<pre>" + html_escape(chunk) + "</pre>")
        _answer_cb(token, cq_id, "RIS в чате")
        return

    _answer_cb(token, cq_id, "Неизвестное действие")


def _telegram_unreachable_message(exc: BaseException) -> str:
    return (
        "Не удаётся подключиться к api.telegram.org.\n\n"
        "Проверьте интернет, фаервол, VPN.\n\n"
        f"{exc!r}"
    )


def main() -> None:
    token = TOKEN
    if not token:
        raise SystemExit("Задайте TELEGRAM_BOT_TOKEN")

    LOGGER.info("Проверка связи с Telegram API…")
    try:
        ping = requests.get(f"{TG_API}/bot{token}/getMe", timeout=15)
        ping.raise_for_status()
        me = ping.json().get("result") or {}
        LOGGER.info("Бот: @%s", me.get("username") or me.get("first_name") or "?")
    except requests.exceptions.RequestException as e:
        LOGGER.error(_telegram_unreachable_message(e))
        raise SystemExit(1) from e

    offset = 0
    LOGGER.info(
        "Long polling. Пакет: до %s строк, пауза %s с. SP_PUBLIC_URL=%s",
        BATCH_MAX_LINES,
        BATCH_PAUSE_SEC,
        BASE,
    )

    while True:
        try:
            r = requests.get(
                f"{TG_API}/bot{token}/getUpdates",
                params={"timeout": 50, "offset": offset},
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
        except requests.exceptions.RequestException as e:
            LOGGER.warning("Связь с Telegram, пауза 15 с: %s", e)
            time.sleep(15)
            continue

        for upd in data.get("result") or []:
            offset = upd["update_id"] + 1

            if upd.get("callback_query"):
                try:
                    _handle_callback(token, upd)
                except Exception:
                    LOGGER.exception("callback")
                continue

            msg = upd.get("message") or {}
            chat = msg.get("chat") or {}
            chat_id = chat.get("id")
            text = (msg.get("text") or "").strip()
            if not chat_id or not text:
                continue

            lower = text.lower()
            if lower in ("/start", "/help"):
                _reply_html(token, chat_id, WELCOME_HTML)
            elif lower.startswith("/doi"):
                _handle_doi(token, chat_id, text[4:].strip())
            elif text.startswith("/url "):
                _handle_url(token, chat_id, text[5:].strip())
            elif text.startswith("/title "):
                _handle_title(token, chat_id, text[7:].strip())
            elif _is_batch_candidate(text):
                try:
                    _handle_batch(token, chat_id, text)
                except Exception:
                    LOGGER.exception("batch")
                    _reply_html(token, chat_id, "Ошибка пакетной обработки.")
            else:
                _smart_single(token, chat_id, text)


if __name__ == "__main__":
    main()

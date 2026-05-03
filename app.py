import json
import logging
import os
import re
import sqlite3
import uuid
from dataclasses import asdict
from html import escape
from typing import List, Optional, Tuple
from urllib.parse import quote_plus
from urllib.parse import urlparse

import requests
from flask import Flask, Response, g, jsonify, request, session, stream_with_context
from flask_caching import Cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from inference_pipeline import (
    PaperMetadata,
    _cyberleninka_pdf_url_from_article_page,
    crossref_enrich,
    explain_candidate_ranking,
    explain_title_interpretation,
    export_bibtex,
    export_ris,
    extract_metadata_from_doi,
    extract_metadata_from_title,
    extract_metadata_from_url,
    extract_pdf_url_from_html,
    extract_title_from_html_basic,
    fetch_html,
    format_bibliography_list,
    format_citation,
    guess_title_from_url,
    is_cyberleninka_non_article_page,
    is_journal_rubric_list_url,
    query_has_cyrillic,
    resolve_article_from_journal_listing,
    resolve_article_from_listing_by_title,
    search_cyberleninka_candidates_by_title,
    search_metadata_candidates_from_title,
    similarity_normalized_titles,
)
from net_security import is_public_http_url

app = Flask(__name__)
LOGGER = logging.getLogger(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me-in-production")

# Portable .exe writes state next to the executable via SP_APP_HOME (see portable_launcher.py).
_app_data_dir = os.getenv(
    "SP_APP_HOME",
    os.path.join(os.path.dirname(__file__), "data"),
)
DB_PATH = os.path.join(_app_data_dir, "app_state.db")
MAX_URL_LEN = 2048
MAX_TITLE_LEN = 350
MAX_DOI_LEN = 255
MAX_PAYLOAD_LEN = 50000
MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_BATCH_LINES = 10
MAX_BATCH_CHARS = 12000

CITATION_STYLES = ("gost", "apa", "ieee", "journal_auto", "springer", "nature")


def _citation_style(value: str) -> str:
    v = (value or "gost").strip().lower()
    return v if v in CITATION_STYLES else "gost"


# Лимиты по IP: ключ — get_remote_address. Для нескольких воркеров задайте RATELIMIT_STORAGE_URI (например redis://...).
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["120 per hour", "30 per minute"],
    storage_uri=os.getenv("RATELIMIT_STORAGE_URI", "memory://"),
)

_cache_config: dict = {"CACHE_TYPE": "SimpleCache"}
if (os.getenv("REDIS_URL") or "").strip():
    _cache_config = {"CACHE_TYPE": "RedisCache", "CACHE_REDIS_URL": os.getenv("REDIS_URL")}
cache = Cache(app, config=_cache_config)

APP_THEME_CSS = """
:root {
  --bg: #f4f7fb;
  --card: #ffffff;
  --text: #1f2937;
  --muted: #6b7280;
  --border: #e5e7eb;
  --accent: #2563eb;
  --accent-hover: #1d4ed8;
  --success: #0f766e;
  --warning: #b45309;
  --danger: #b91c1c;
  --topbar-bg: rgba(255, 255, 255, 0.88);
  --shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Segoe UI", Arial, sans-serif;
  background: radial-gradient(circle at top right, #e9f2ff, var(--bg) 45%);
  color: var(--text);
  scroll-behavior: smooth;
}
[data-theme="dark"] {
  --bg: #0b1220;
  --card: #0f172a;
  --text: #e5e7eb;
  --muted: #94a3b8;
  --border: #1e293b;
  --accent: #3b82f6;
  --accent-hover: #60a5fa;
  --topbar-bg: rgba(8, 15, 29, 0.82);
  --shadow: 0 10px 24px rgba(0, 0, 0, 0.35);
}
.container { max-width: 1050px; margin: 18px auto; padding: 0 16px 36px; }
.topbar {
  position: sticky;
  top: 10px;
  z-index: 40;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  background: var(--topbar-bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 8px 10px;
  backdrop-filter: blur(8px);
}
.topbar-left, .topbar-right { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.topbar-left {
  flex: 1 1 auto;
  min-width: 0;
}
.topbar-right {
  flex: 0 0 auto;
}
.topbar-tabs {
  position: relative;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px;
  border-radius: 14px;
  background: color-mix(in srgb, var(--card) 80%, #dbeafe 20%);
  border: 1px solid color-mix(in srgb, var(--border) 78%, #bfdbfe 22%);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.5);
}
.tab-indicator {
  position: absolute;
  top: 4px;
  left: 4px;
  height: calc(100% - 8px);
  border-radius: 10px;
  background: linear-gradient(135deg, var(--accent), #60a5fa);
  box-shadow: 0 10px 24px rgba(37, 99, 235, 0.22);
  transition: transform 0.28s ease, width 0.28s ease, opacity 0.2s ease;
  z-index: 0;
  opacity: 0;
}
.brand {
  font-weight: 700;
  font-size: 14px;
  color: var(--text);
  padding: 6px 8px;
}
.result-nav {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}
.hero {
  background: linear-gradient(120deg, #0f172a, #1e3a8a);
  color: #fff;
  border-radius: 16px;
  padding: 22px 24px;
  box-shadow: var(--shadow);
}
.hero h1, .hero h2 { margin: 0 0 6px 0; font-size: 28px; font-weight: 700; }
.hero p { margin: 0; color: #cbd5e1; font-size: 14px; }
.card {
  margin-top: 16px;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px;
  box-shadow: var(--shadow);
}
.grid-2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}
.field-wrap { position: relative; }
.field-icon {
  position: absolute;
  left: 10px;
  top: 50%;
  transform: translateY(-50%);
  font-size: 14px;
  opacity: 0.75;
  pointer-events: none;
}
.with-icon {
  padding-left: 34px;
}
.label { font-size: 13px; color: var(--muted); margin-bottom: 5px; display: block; }
input, select, button {
  font-family: inherit;
  font-size: 14px;
  -webkit-tap-highlight-color: transparent;
  touch-action: manipulation;
}
input, select {
  width: 100%;
  padding: 10px 11px;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--card);
  color: var(--text);
}
input:focus, select:focus {
  border-color: #93c5fd;
  box-shadow: 0 0 0 3px #dbeafe;
  outline: none;
}
.btn {
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 9px 12px;
  background: var(--card);
  color: var(--text);
  cursor: pointer;
  text-decoration: none;
  display: inline-block;
  -webkit-tap-highlight-color: transparent;
  touch-action: manipulation;
}
.btn:hover { filter: brightness(0.98); }
.tab-button {
  position: relative;
  z-index: 1;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  white-space: nowrap;
  transition: background 0.2s ease, color 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
}
.tab-button:hover {
  background: color-mix(in srgb, var(--card) 65%, #dbeafe 35%);
  border-color: color-mix(in srgb, var(--border) 70%, #93c5fd 30%);
  box-shadow: 0 6px 18px rgba(59, 130, 246, 0.12);
  transform: translateY(-1px);
}
.tab-button .tab-icon {
  font-size: 14px;
  line-height: 1;
}
.tab-button .tab-label {
  line-height: 1;
}
.nav-link.active {
  color: #fff;
  border-color: transparent;
  background: transparent;
  box-shadow: none;
}
.btn-primary {
  background: var(--accent);
  color: #fff;
}
.btn-primary:hover { background: var(--accent-hover); }
.btn-soft {
  background: #eef2ff;
  color: #1e3a8a;
  border: 1px solid #c7d2fe;
}
.btn-warn {
  background: #fef2f2;
  color: #991b1b;
  border: 1px solid #fecaca;
}
.btn-warn:hover { filter: brightness(0.97); }
.theme-toggle {
  min-width: 116px;
  text-align: center;
  border-radius: 12px;
  border-color: color-mix(in srgb, var(--border) 78%, #bfdbfe 22%);
  background: color-mix(in srgb, var(--card) 82%, #dbeafe 18%);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.45);
  transition: background 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
}
.theme-toggle:hover {
  background: color-mix(in srgb, var(--card) 70%, #dbeafe 30%);
  border-color: color-mix(in srgb, var(--border) 65%, #93c5fd 35%);
  box-shadow: 0 8px 20px rgba(59, 130, 246, 0.14);
  transform: translateY(-1px);
}
.section-title {
  margin: 0 0 10px;
  font-size: 18px;
  font-weight: 700;
}
.tab-shell {
  margin-top: 16px;
  position: relative;
  min-height: 320px;
}
.tab-panel {
  display: none;
  opacity: 0;
  transform: translateY(18px) scale(0.985);
  filter: blur(4px);
  transform-origin: top center;
  transition: opacity 0.32s ease, transform 0.32s ease, filter 0.32s ease;
}
.tab-panel.active {
  display: block;
  opacity: 1;
  transform: translateY(0) scale(1);
  filter: blur(0);
}
.info-card p {
  margin: 0 0 10px 0;
  line-height: 1.55;
  color: var(--text);
}
.info-card p:last-child { margin-bottom: 0; }
.mono-box {
  white-space: pre-wrap;
  background: color-mix(in srgb, var(--card) 85%, #e2e8f0 15%);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px 12px;
  margin: 0;
  font-size: 13px;
}
.meta-row {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}
.meta-item {
  background: color-mix(in srgb, var(--card) 86%, #e2e8f0 14%);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 8px 10px;
}
.meta-item .k {
  color: var(--muted);
  font-size: 12px;
}
.meta-item .v {
  margin-top: 2px;
  font-size: 14px;
}
.candidate {
  border: 1px solid var(--border);
  background: color-mix(in srgb, var(--card) 92%, #dbeafe 8%);
  border-radius: 12px;
  padding: 8px 10px;
  margin-top: 12px;
}
.candidate h4 {
  margin: 0 0 8px 0;
  font-size: 15px;
}
.candidate .row { margin-top: 6px; }
.variant-collapse {
  border: 1px solid var(--border);
  border-radius: 10px;
  background: color-mix(in srgb, var(--card) 92%, #dbeafe 8%);
  margin-top: 10px;
}
.variant-collapse summary {
  list-style: none;
  cursor: pointer;
  padding: 10px 12px;
  font-weight: 600;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.variant-collapse summary::-webkit-details-marker { display: none; }
.variant-collapse[open] summary {
  border-bottom: 1px solid var(--border);
}
.variant-body { padding: 10px 12px 12px; }
.hint {
  margin-top: 6px;
  font-size: 12px;
  color: var(--muted);
}
.action-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.summary-score {
  color: #64748b;
  font-weight: 500;
}
.mt-10 { margin-top: 10px; }
.mt-12 { margin-top: 12px; }
.mb-8 { margin-bottom: 8px; }
.empty-text { margin: 0; color: #64748b; }
.status-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 12px;
  border: 1px solid var(--border);
  font-weight: 600;
}
.status-ok { color: #065f46; background: #ecfdf5; border-color: #a7f3d0; }
.status-warn { color: #92400e; background: #fffbeb; border-color: #fde68a; }
.status-bad { color: #991b1b; background: #fef2f2; border-color: #fecaca; }
[data-theme="dark"] .status-ok { color: #6ee7b7; background: #042f2e; border-color: #115e59; }
[data-theme="dark"] .status-warn { color: #fcd34d; background: #3f2d05; border-color: #78350f; }
[data-theme="dark"] .status-bad { color: #fda4af; background: #3f1018; border-color: #7f1d1d; }
[data-theme="dark"] .variant-collapse {
  background: color-mix(in srgb, var(--card) 94%, #1d4ed8 6%);
}
[data-theme="dark"] .topbar-tabs {
  background: color-mix(in srgb, var(--card) 88%, #1d4ed8 12%);
  border-color: color-mix(in srgb, var(--border) 76%, #2563eb 24%);
}
[data-theme="dark"] .tab-button:hover {
  background: color-mix(in srgb, var(--card) 78%, #2563eb 22%);
  border-color: color-mix(in srgb, var(--border) 62%, #60a5fa 38%);
  box-shadow: 0 8px 20px rgba(59, 130, 246, 0.22);
}
[data-theme="dark"] .tab-indicator {
  box-shadow: 0 10px 28px rgba(59, 130, 246, 0.28);
}
[data-theme="dark"] .theme-toggle {
  background: color-mix(in srgb, var(--card) 86%, #1d4ed8 14%);
  border-color: color-mix(in srgb, var(--border) 72%, #2563eb 28%);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
}
[data-theme="dark"] .theme-toggle:hover {
  background: color-mix(in srgb, var(--card) 74%, #2563eb 26%);
  border-color: color-mix(in srgb, var(--border) 58%, #60a5fa 42%);
  box-shadow: 0 10px 24px rgba(59, 130, 246, 0.24);
}
.loading-overlay {
  position: fixed;
  inset: 0;
  background: rgba(15, 23, 42, 0.55);
  display: none;
  align-items: center;
  justify-content: center;
  z-index: 90;
}
.loading-panel {
  width: min(760px, 92vw);
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px;
}
.skeleton {
  border-radius: 8px;
  background: linear-gradient(90deg, #e2e8f0 25%, #f1f5f9 38%, #e2e8f0 63%);
  background-size: 400% 100%;
  animation: skeleton-loading 1.2s ease-in-out infinite;
}
[data-theme="dark"] .skeleton {
  background: linear-gradient(90deg, #27344d 25%, #334155 38%, #27344d 63%);
  background-size: 400% 100%;
}
.sk-line { height: 12px; margin-bottom: 10px; }
.sk-title { height: 20px; width: 40%; margin-bottom: 14px; }
@keyframes skeleton-loading {
  0% { background-position: 100% 50%; }
  100% { background-position: 0 50%; }
}
a { color: #1d4ed8; text-decoration: none; }
a:hover { text-decoration: underline; }
.variant-body a {
  overflow-wrap: anywhere;
  word-break: break-word;
}
.toast {
  display: none;
  position: fixed;
  right: 24px;
  bottom: 24px;
  background: #222;
  color: #fff;
  padding: 10px 14px;
  border-radius: 8px;
  max-width: min(420px, calc(100vw - 28px));
  z-index: 9999;
  box-shadow: 0 12px 28px rgba(0, 0, 0, 0.22);
}
@media (max-width: 760px) {
  .grid-2, .meta-row { grid-template-columns: 1fr; }
  .topbar { position: static; }
  .container {
    margin-top: 10px;
    padding:
      0 calc(10px + env(safe-area-inset-right))
      calc(28px + env(safe-area-inset-bottom))
      calc(10px + env(safe-area-inset-left));
  }
  .topbar {
    align-items: stretch;
    flex-direction: column;
    gap: 10px;
    padding: 10px;
  }
  .topbar-left {
    gap: 6px;
    width: 100%;
  }
  .topbar-right {
    width: 100%;
  }
  .brand {
    width: 100%;
    padding: 2px 2px 6px;
  }
  .topbar-tabs {
    width: 100%;
    gap: 6px;
    padding: 4px;
    overflow-x: auto;
    scrollbar-width: none;
  }
  .topbar-tabs::-webkit-scrollbar {
    display: none;
  }
  .tab-button {
    flex: 1 0 auto;
    justify-content: center;
    min-width: max-content;
    padding: 8px 10px;
    font-size: 13px;
  }
  .tab-button .tab-icon {
    font-size: 13px;
  }
  .theme-toggle {
    width: 100%;
    min-width: 0;
    justify-content: center;
    padding: 10px 12px;
  }
  .hero {
    padding: 18px 16px;
    border-radius: 14px;
  }
  .hero h1, .hero h2 {
    font-size: 22px;
  }
  .hero p {
    font-size: 13px;
  }
  .card {
    border-radius: 12px;
    padding: 14px 12px;
  }
  .section-title {
    font-size: 16px;
    margin-bottom: 8px;
  }
  .mono-box {
    font-size: 12.5px;
    padding: 9px 10px;
  }
  .variant-collapse summary {
    flex-direction: column;
    align-items: flex-start;
    gap: 4px;
    padding: 10px;
  }
  .variant-body {
    padding: 10px;
  }
  .candidate, .meta-item {
    padding: 9px;
  }
  .action-row .btn {
    width: 100%;
    text-align: center;
  }
  .result-nav {
    width: 100%;
    gap: 6px;
  }
  .result-nav .btn {
    flex: 1 1 calc(50% - 6px);
    min-width: 120px;
    text-align: center;
    padding: 9px 10px;
    font-size: 13px;
  }
  .btn, input, select {
    min-height: 44px;
  }
  #parseSubmitBtn {
    width: 100%;
  }
  .toast {
    right: calc(10px + env(safe-area-inset-right));
    left: calc(10px + env(safe-area-inset-left));
    bottom: calc(10px + env(safe-area-inset-bottom));
    max-width: none;
    border-radius: 10px;
  }
}

@media (max-width: 480px) {
  .topbar {
    border-radius: 14px;
  }
  .tab-button {
    gap: 6px;
    padding: 8px 9px;
    border-radius: 9px;
    font-size: 12px;
  }
  .tab-button .tab-label {
    max-width: 96px;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .theme-toggle {
    border-radius: 10px;
    font-size: 13px;
  }
  .result-nav .btn {
    flex: 1 1 100%;
    min-width: 0;
  }
  .tab-shell {
    min-height: 0;
  }
  .loading-panel {
    width: calc(100vw - 20px - env(safe-area-inset-left) - env(safe-area-inset-right));
    padding: 12px;
  }
}
"""


def _get_session_id() -> str:
    sid = session.get("sid")
    if not sid:
        sid = str(uuid.uuid4())
        session["sid"] = sid
    return sid


def _get_db() -> sqlite3.Connection:
    if "db" not in g:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


def _init_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS selected_papers (
                session_id TEXT NOT NULL,
                paper_key TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (session_id, paper_key)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                mode TEXT,
                url TEXT,
                doi TEXT,
                title_query TEXT,
                result_title TEXT,
                ok INTEGER NOT NULL DEFAULT 1,
                error_message TEXT
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


@app.teardown_appcontext
def _close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def _db_load_selected(session_id: str) -> List[PaperMetadata]:
    rows = _get_db().execute(
        "SELECT payload_json FROM selected_papers WHERE session_id = ? ORDER BY created_at",
        (session_id,),
    ).fetchall()
    result: List[PaperMetadata] = []
    for row in rows:
        try:
            result.append(PaperMetadata(**json.loads(row["payload_json"])))
        except Exception:
            continue
    return result


def _db_save_selected(session_id: str, paper: PaperMetadata) -> bool:
    payload = json.dumps(asdict(paper), ensure_ascii=False)
    key = _paper_key(paper)
    cur = _get_db().execute(
        "INSERT OR IGNORE INTO selected_papers(session_id, paper_key, payload_json) VALUES (?, ?, ?)",
        (session_id, key, payload),
    )
    _get_db().commit()
    return bool(cur.rowcount)


def _db_clear_selected(session_id: str) -> None:
    _get_db().execute("DELETE FROM selected_papers WHERE session_id = ?", (session_id,))
    _get_db().commit()


def _db_pop_last_selected(session_id: str) -> bool:
    """Удаляет последнюю по времени добавления запись сессии. True, если строка была удалена."""
    db = _get_db()
    row = db.execute(
        """
        SELECT paper_key FROM selected_papers
        WHERE session_id = ?
        ORDER BY datetime(created_at) DESC, paper_key DESC
        LIMIT 1
        """,
        (session_id,),
    ).fetchone()
    if not row:
        return False
    db.execute(
        "DELETE FROM selected_papers WHERE session_id = ? AND paper_key = ?",
        (session_id, row["paper_key"]),
    )
    db.commit()
    return True


def _anonymous_mode() -> bool:
    """При SP_ANONYMOUS=1 не пишем историю поиска (локальные/чувствительные сценарии)."""
    return os.getenv("SP_ANONYMOUS", "").strip().lower() in ("1", "true", "yes", "on")


def _db_log_search(
    session_id: str,
    mode_used: str,
    url: str,
    doi: str,
    title_query: str,
    result_title: str,
    ok: bool = True,
    error_message: Optional[str] = None,
) -> None:
    if _anonymous_mode():
        return
    try:
        _get_db().execute(
            """
            INSERT INTO search_history(session_id, mode, url, doi, title_query, result_title, ok, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                mode_used,
                (url or "")[:2000],
                (doi or "")[:MAX_DOI_LEN],
                (title_query or "")[:MAX_TITLE_LEN],
                (result_title or "")[:500],
                1 if ok else 0,
                (error_message or "")[:2000] if error_message else None,
            ),
        )
        _get_db().commit()
    except Exception:
        LOGGER.exception("search_history insert failed")


def _validate_input_length(value: str, max_len: int, field_name: str) -> Optional[str]:
    if value and len(value) > max_len:
        return f"Поле '{field_name}' слишком длинное (max {max_len})."
    return None


@app.errorhandler(requests.exceptions.RequestException)
def handle_request_exception(err):
    LOGGER.warning("External API request error: %s", err)
    return (
        """
        <html><body style="font-family:Arial; margin:40px;">
        <h2>Внешний сервис временно недоступен</h2>
        <p>Попробуйте повторить запрос через 10-20 секунд.</p>
        <p><a href="/">Вернуться на главную</a></p>
        </body></html>
        """,
        503,
    )


@app.get("/health")
@limiter.exempt
def health():
    # Иначе хостинг (Railway, Render и т.д.) получает 429 на health check и роняет инстанс.
    return jsonify({"status": "ok"}), 200


def _safe_filename(value: str, default: str = "article") -> str:
    # HTTP headers in local dev server are latin-1 encoded.
    # Keep filenames ASCII-only to avoid UnicodeEncodeError in Content-Disposition.
    raw = (value or "").strip()
    ascii_only = raw.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", ascii_only).strip("_")
    return cleaned[:80] or default


def _paper_key(p: PaperMetadata) -> str:
    if p.doi:
        return f"doi:{p.doi.lower()}"
    base = f"{(p.title or '').strip().lower()}|{(p.year or '').strip()}"
    return f"title:{base}"


def _merge_unique_papers(primary: List[PaperMetadata], extra: List[PaperMetadata], limit: int) -> List[PaperMetadata]:
    merged: List[PaperMetadata] = []
    seen = set()
    for paper in (primary or []) + (extra or []):
        if not paper:
            continue
        key = _paper_key(paper)
        if key in seen:
            continue
        seen.add(key)
        merged.append(paper)
        if len(merged) >= limit:
            break
    return merged


def _looks_like_direct_pdf_url(url: str) -> bool:
    """
    Unpaywall иногда кладёт в поле «url» landing (doi.org/html), а не файл PDF.
    Без этой проверки в списке кандидатов оказывается HTML и загрузка всегда падает.
    """
    u = (url or "").strip()
    if not u:
        return False
    low = u.lower().split("?", 1)[0].rstrip("/")
    if low.endswith(".pdf"):
        return True
    if "/format/pdf" in low or "/pdf/" in low:
        return True
    return False


def _query_unpaywall_pdf_url(doi: str) -> Optional[str]:
    doi = (doi or "").strip()
    if not doi:
        return None
    try:
        r = requests.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": "scientific.parser.demo@gmail.com"},
            timeout=20,
            headers={"User-Agent": "biblio-parser/1.0"},
        )
        if r.status_code != 200:
            return None
        data = r.json() or {}
    except Exception:
        return None

    def from_loc(loc: dict) -> Optional[str]:
        if not isinstance(loc, dict):
            return None
        ufp = (loc.get("url_for_pdf") or "").strip()
        if ufp:
            return ufp
        pub = (loc.get("url") or "").strip()
        if pub and _looks_like_direct_pdf_url(pub):
            return pub
        return None

    best = data.get("best_oa_location") or {}
    hit = from_loc(best if isinstance(best, dict) else {})
    if hit:
        return hit

    for loc in data.get("oa_locations") or []:
        hit = from_loc(loc if isinstance(loc, dict) else {})
        if hit:
            return hit
    return None


def _discover_pdf_from_html_landing(page_url: str) -> Optional[str]:
    """
    На странице издателя часто есть прямая ссылка на PDF (meta citation_pdf_url или <a href=...pdf>),
    хотя ни CrossRef, ни Unpaywall её не передали.
    """
    page_url = (page_url or "").strip()
    if not page_url.startswith("http"):
        return None
    path_low = page_url.lower().split("?", 1)[0]
    if path_low.endswith(".pdf"):
        return None
    try:
        html = fetch_html(page_url)
    except Exception:
        return None
    return extract_pdf_url_from_html(html, source_url=page_url)


def _collect_landing_discovery_urls(source_url: str, doi: str) -> List[str]:
    """Уникальные страницы, где имеет смысл искать ссылку на PDF."""
    out: List[str] = []
    seen = set()

    def push(u: str) -> None:
        u = (u or "").strip()
        if not u.startswith("http"):
            return
        key = u.split("#", 1)[0].rstrip("/").lower()
        if key in seen:
            return
        seen.add(key)
        out.append(u)

    push(source_url)
    if doi:
        push(f"https://doi.org/{doi}")
    return out


def _extract_arxiv_pdf_candidate(source_url: str) -> Optional[str]:
    source_url = (source_url or "").strip()
    if "arxiv.org/abs/" in source_url:
        return source_url.replace("/abs/", "/pdf/") + ".pdf"
    return None


def _build_pdf_candidates(*, pdf_url: str, doi: str, source_url: str) -> List[str]:
    candidates: List[str] = []

    def add(url: str) -> None:
        u = (url or "").strip()
        if not u:
            return
        if not re.match(r"^https?://", u, flags=re.IGNORECASE):
            return
        if not is_public_http_url(u):
            return
        if u not in candidates:
            candidates.append(u)

    add(pdf_url)
    if doi:
        add(_query_unpaywall_pdf_url(doi))

    for landing in _collect_landing_discovery_urls(source_url, doi):
        discovered = _discover_pdf_from_html_landing(landing)
        if discovered:
            add(discovered)

    add(_extract_arxiv_pdf_candidate(source_url))
    cl_pdf = _cyberleninka_pdf_url_from_article_page(source_url)
    if cl_pdf:
        add(cl_pdf)
    return candidates


def _try_download_pdf_from_candidates(candidates: List[str]) -> Tuple[Optional[Response], Optional[str], Optional[str]]:
    for candidate in candidates:
        try:
            parsed = urlparse(candidate)
            origin = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else ""
            r = requests.get(
                candidate,
                timeout=45,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "application/pdf,*/*;q=0.8",
                    "Referer": origin or "https://www.google.com/",
                },
                allow_redirects=True,
                stream=True,
            )
            if r.status_code != 200:
                r.close()
                continue
            content_type = (r.headers.get("Content-Type") or "").lower()
            content_length = int(r.headers.get("Content-Length") or 0)
            if content_length and content_length > MAX_PDF_BYTES:
                r.close()
                continue

            first_chunk = b""
            for chunk in r.iter_content(chunk_size=2048):
                first_chunk = chunk or b""
                break
            starts_like_pdf = first_chunk.startswith(b"%PDF")
            if not ("pdf" in content_type or starts_like_pdf):
                r.close()
                continue

            def generate():
                sent = 0
                try:
                    nonlocal first_chunk
                    if first_chunk:
                        sent += len(first_chunk)
                        if sent > MAX_PDF_BYTES:
                            return
                        yield first_chunk
                    for chunk in r.iter_content(chunk_size=8192):
                        if not chunk:
                            continue
                        sent += len(chunk)
                        if sent > MAX_PDF_BYTES:
                            break
                        yield chunk
                finally:
                    r.close()

            filename = f"{_safe_filename(candidate.split('/')[-1] or 'article')}.pdf"
            resp = Response(
                stream_with_context(generate()),
                mimetype="application/pdf",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"',
                    "X-Source-PDF-URL": candidate,
                },
            )
            return resp, candidate, None
        except Exception:
            continue
    return None, None, "Не удалось получить полный PDF: ссылка отсутствует или доступ ограничен издателем."


def _pdf_request_headers(candidate: str) -> dict:
    parsed = urlparse(candidate)
    origin = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else ""
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/pdf,*/*;q=0.8",
        "Referer": origin or "https://www.google.com/",
    }


def _probe_pdf_candidates(candidates: List[str]) -> Tuple[bool, Optional[str], str]:
    """Сначала HEAD (Content-Type, размер), при сомнении — GET с чтением первых байт (%PDF)."""
    for candidate in candidates:
        headers = _pdf_request_headers(candidate)
        try:
            h = requests.head(candidate, timeout=(5, 20), headers=headers, allow_redirects=True)
            if h.status_code == 200:
                ct = (h.headers.get("Content-Type") or "").lower()
                cl_raw = h.headers.get("Content-Length")
                if cl_raw:
                    try:
                        if int(cl_raw) > MAX_PDF_BYTES:
                            continue
                    except ValueError:
                        pass
                if "pdf" in ct:
                    return True, candidate, "PDF доступен (HEAD: Content-Type, размер в пределах лимита)."
        except Exception:
            pass

        try:
            r = requests.get(
                candidate,
                timeout=30,
                headers=headers,
                allow_redirects=True,
                stream=True,
            )
            if r.status_code != 200:
                r.close()
                continue
            content_type = (r.headers.get("Content-Type") or "").lower()
            try:
                content_length = int(r.headers.get("Content-Length") or 0)
            except ValueError:
                content_length = 0
            if content_length and content_length > MAX_PDF_BYTES:
                r.close()
                continue
            first_chunk = b""
            for chunk in r.iter_content(chunk_size=2048):
                first_chunk = chunk or b""
                break
            starts_like_pdf = first_chunk.startswith(b"%PDF")
            r.close()
            if "pdf" in content_type or starts_like_pdf:
                detail = "по сигнатуре и заголовкам ответа" if starts_like_pdf else "по Content-Type"
                return True, candidate, f"PDF доступен для скачивания ({detail})."
        except Exception:
            continue
    return False, None, "PDF не найден или ограничен paywall."


def _fulltext_info(item: PaperMetadata) -> dict:
    urls = _build_pdf_candidates(
        pdf_url=item.pdf_url or "",
        doi=item.doi or "",
        source_url=item.source_url or "",
    )
    if item.pdf_url:
        return {
            "status": "direct_pdf_candidate",
            "label": "Есть потенциальная PDF-ссылка",
            "links": urls,
            "reason": "Ссылка на PDF обнаружена, но доступность файла проверяется при скачивании.",
        }
    if urls:
        return {
            "status": "open_access_candidate",
            "label": "Найдены Open Access кандидаты",
            "links": urls,
            "reason": "Есть альтернативные источники (например, репозиторий/препринт).",
        }
    return {
        "status": "not_found_or_paywalled",
        "label": "Полный текст не найден",
        "links": [],
        "reason": "Вероятно, статья закрыта издателем (paywall) или OA-версия отсутствует.",
    }


def _language_catalog_hint_block(title_input: str, candidates: List[PaperMetadata]) -> str:
    """
    Подсказка при рассогласовании языка запроса и заголовка в каталогах / низкой уверенности.
    """
    q = (title_input or "").strip()
    if len(q) < 4 or not candidates:
        return ""

    top = candidates[0]
    top_title = (top.title or "").strip()
    if not top_title:
        return ""

    user_cyr = query_has_cyrillic(q)
    cand_cyr = query_has_cyrillic(top_title)
    sc = top.search_score
    low_conf = sc is not None and float(sc) < 0.52

    if user_cyr == cand_cyr and not low_conf:
        return ""

    parts: List[str] = []
    if user_cyr != cand_cyr:
        parts.append(
            "Язык вашего запроса не совпадает с тем, как заголовок этой работы записан в каталогах "
            "(часто CrossRef хранит англоязычный вариант, даже если на сайте журнала показан перевод названия)."
        )
    if top.doi:
        cr = crossref_enrich(top.doi)
        if cr and (cr.title or "").strip():
            parts.append(f"Заголовок в CrossRef (по DOI {top.doi}): {cr.title.strip()}")
    if low_conf and user_cyr == cand_cyr:
        parts.append(
            "Релевантность первого варианта относительно низкая — в выдачу могли попасть другие работы "
            "по схожим словам (медицина, компьютерное зрение и т.д.)."
        )
    parts.append("Что попробовать: английский вариант названия с сайта / DOI / прямая ссылка на статью.")
    text = "\n\n".join(p for p in parts if p)
    return f"""
        <div class="card" style="border-left:4px solid #2563eb;">
        <h3 class="section-title">Подсказка по поиску по названию</h3>
        <pre class="mono-box" style="white-space:pre-wrap;">{escape(text)}</pre>
        </div>
        """


def _journal_rubric_usage_block(
    url: str, title_input: str, *, listing_match_failed: bool, page_is_listing: bool
) -> str:
    if not (url or "").strip():
        return ""

    tit = (title_input or "").strip()
    if listing_match_failed and tit:
        msg = (
            "Не удалось сопоставить введённое название со ссылкой на странице списка "
            "(или нужная статья на другой странице списка).\n\n"
            "Проверьте формулировку или вставьте прямую ссылку на статью …/articles/… "
            "или укажите DOI."
        )
    elif page_is_listing and not tit:
        msg = (
            "Вы указали адрес страницы списка статей (рубрики), а не конкретной статьи.\n\n"
            "Сделайте одно из двух:\n"
            "• вставьте в поле «URL статьи» прямую ссылку на статью, вида …/ru/articles/… ;\n"
            "• либо оставьте URL рубрики и **дополнительно** введите в поле «Название статьи» "
            "точное название, как в списке на сайте — тогда система попытается найти ссылку в HTML списка."
        )
    else:
        return ""

    return f"""
        <div class="card" style="border-left:4px solid #b45309;">
        <h3 class="section-title">Страница журнала</h3>
        <pre class="mono-box" style="white-space:pre-wrap;">{escape(msg)}</pre>
        </div>
        """


def _generic_listing_hint_block(*, generic_match_failed: bool, title_input: str) -> str:
    if not generic_match_failed or not (title_input or "").strip():
        return ""
    msg = (
        "На странице по указанному URL не найдена ссылка, текст которой достаточно близок "
        "к введённому названию (поиск по ссылкам на том же сайте).\n\n"
        "Скопируйте заголовок с сайта дословно, откройте страницу списка выпуска/рубрики, "
        "куда входит статья, либо вставьте прямую ссылку на материал или DOI. "
        "Для главной страницы портала иногда проще оставить только поле «Название» без URL."
    )
    return f"""
        <div class="card" style="border-left:4px solid #0369a1;">
        <h3 class="section-title">Подбор статьи по странице</h3>
        <pre class="mono-box" style="white-space:pre-wrap;">{escape(msg)}</pre>
        </div>
        """


@app.get("/")
def index():
    return f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>BiblioParser Demo</title>
      <style>{APP_THEME_CSS}</style>
    </head>
    <body>
      <div class="container">
        <div class="topbar">
          <div class="topbar-left">
            <div class="brand">📚 BiblioParser</div>
            <div class="topbar-tabs" id="topbarTabs">
              <div class="tab-indicator" id="tabIndicator"></div>
              <a class="btn nav-link tab-button active" href="#home" data-tab-target="home">
                <span class="tab-icon">🏠</span>
                <span class="tab-label">Главная</span>
              </a>
              <a class="btn nav-link tab-button" href="#about" data-tab-target="about">
                <span class="tab-icon">ℹ️</span>
                <span class="tab-label">О проекте</span>
              </a>
              <a class="btn nav-link tab-button" href="#howto" data-tab-target="howto">
                <span class="tab-icon">✨</span>
                <span class="tab-label">Как пользоваться</span>
              </a>
            </div>
          </div>
          <div class="topbar-right">
            <a class="btn btn-soft nav-link" href="/history">История</a>
            <button id="themeToggle" class="btn theme-toggle">🌙 Dark</button>
          </div>
        </div>

        <div class="hero">
          <h1>BiblioParser Demo</h1>
          <p>Извлечение метаданных научных статей, проверка полного текста и экспорт библиографии.</p>
        </div>
        <div id="tabContentRoot" class="tab-shell">
          <section id="panel-home" class="tab-panel active" data-tab-panel="home">
            <div class="card">
              <form id="parseForm" method="POST" action="/parse">
                <label class="label"><b>URL статьи</b> (опционально)</label>
                <div class="field-wrap">
                  <span class="field-icon">🔗</span>
                  <input class="with-icon" name="url" placeholder="https://..." />
        </div>

                <div style="height:10px;"></div>
                <label class="label"><b>DOI</b> (опционально)</label>
                <div class="field-wrap">
                  <span class="field-icon">🧬</span>
                  <input class="with-icon" name="doi" placeholder="10.1016/j.net.2025.103970" />
        </div>

                <div style="height:10px;"></div>
                <label class="label"><b>Название статьи</b> (опционально)</label>
                <div class="field-wrap">
                  <span class="field-icon">📝</span>
                  <input class="with-icon" name="title" placeholder="Toward Verified Artificial Intelligence" />
        </div>

                <div class="grid-2" style="margin-top:12px;">
                  <div>
                    <label class="label"><b>Количество вариантов</b></label>
                    <input name="max_variants" type="number" min="1" max="10" value="5" />
                  </div>
                  <div>
                    <label class="label"><b>Стиль цитирования</b></label>
                    <select name="citation_style">
                      <option value="gost">ГОСТ-like</option>
                      <option value="apa">APA</option>
                      <option value="ieee">IEEE</option>
                      <option value="journal_auto">Журнал (авто по названию)</option>
                      <option value="springer">Springer-подобный</option>
                      <option value="nature">Nature-подобный</option>
                    </select>
                  </div>
                </div>

                <div style="margin-top:14px;">
                  <button id="parseSubmitBtn" class="btn btn-primary">Parse</button>
        </div>
      </form>
            </div>
            <div class="card" style="margin-top:14px;">
              <h3 class="section-title">Массовый импорт</h3>
              <p style="color:var(--muted);font-size:14px;">До {MAX_BATCH_LINES} строк: DOI (<code>10....</code>), URL или свободное название. Каждая строка обрабатывается отдельно (без эвристик «страница списка + заголовок»). Для сложных случаев используйте форму выше.</p>
              <form method="POST" action="/parse_batch">
                <label class="label"><b>Строки</b></label>
                <textarea name="batch" rows="7" style="width:100%;padding:10px;border-radius:8px;border:1px solid var(--border);font-family:inherit;" placeholder="10.1000/182&#10;https://doi.org/10.1000/182"></textarea>
                <div style="margin-top:10px;">
                  <label class="label"><b>Стиль цитирования</b></label>
                  <select name="citation_style">
                    <option value="gost">ГОСТ-like</option>
                    <option value="apa">APA</option>
                    <option value="ieee">IEEE</option>
                    <option value="journal_auto">Журнал (авто по названию)</option>
                    <option value="springer">Springer-подобный</option>
                    <option value="nature">Nature-подобный</option>
                  </select>
                </div>
                <div style="margin-top:12px;">
                  <button type="submit" class="btn btn-soft">Разобрать пакет</button>
                </div>
              </form>
            </div>
          </section>
          <section id="panel-about" class="tab-panel" data-tab-panel="about">
            <div class="card info-card">
              <h3 class="section-title">О проекте</h3>
              <p><b>BiblioParser</b> извлекает метаданные по URL, DOI или названию и собирает библиографию.</p>
              <p><b>Где участвует ML.</b> Классификатор блоков страницы: TF‑IDF (слово и символьные n‑граммы) + признаки DOM (тег, классы, глубина, плотность ссылок и др.) предсказывает тип каждого текстового блока — заголовок, авторы, год, журнал, DOI и т.д., чтобы вытащить поля со страницы журнала (не класс «References» целиком). Дополнительно можно включить эмбеддинги Sentence‑BERT по полю текста блока: при обучении задайте <code>USE_SBERT=1</code> и зависимости из <code>requirements-ml.txt</code>. LayoutLM и аналоги требуют отдельной разметки/инфраструктуры и не входят в базовый пайплайн.</p>
              <p><b>Нормализация запроса по названию.</b> CAPS приводятся к нормальному виду, строятся варианты раскладки (в т.ч. «ghbdtn» → «привет» через подстановку раскладки), показывается блок «Я вас понял так»; опечатки частично снимаются нечётким сопоставлением с заголовками из CrossRef.</p>
              <p><b>Безопасность и кэш.</b> Исходящие URL проверяются на SSRF в <code>net_security.py</code> (схема, запрет приватных IP после DNS, опционально <code>ALLOW_ONION=1</code> для onion‑хостов и <code>TOR_SOCKS_PROXY</code> для запросов через Tor). В процессе используется <code>lru_cache</code> для повторяющихся API‑запросов; при переменной <code>REDIS_URL</code> подключаются Redis и Flask‑Caching (см. также <code>RATELIMIT_STORAGE_URI</code> для лимитов).</p>
              <p><b>Интеграции.</b> JSON для внешних инструментов: <code>/api/work?doi=…</code>. Расширение Chrome: каталог <code>extensions/chrome</code>. Telegram: скрипт <code>telegram_bot.py</code> (токен в <code>TELEGRAM_BOT_TOKEN</code>, URL сервиса в <code>SP_PUBLIC_URL</code>).</p>
              <p><b>Срок жизни демо.</b> Публичный адрес зависит от хостинга; для демонстрации без сети запишите короткое видео экрана или используйте локальный запуск / portable (см. <code>README_PORTABLE.txt</code> при сборке).</p>
              <p><b>Открытые данные разметки.</b> Файл размеченных страниц лежит в репозитории в <code>data/annotated_dataset.json</code> (при публикации на GitHub может служить открытым датасетом).</p>
            </div>
          </section>
            <section id="panel-howto" class="tab-panel" data-tab-panel="howto">
            <div class="card info-card">
              <h3 class="section-title">Как пользоваться</h3>
              <p>Укажи <b>DOI</b> (если есть) — так точнее всего. Либо прямую <b>ссылку на страницу статьи</b> (не на список рубрики), либо <b>название</b> для поиска в каталогах.</p>
              <p>Если в URL открыта <b>страница списка</b> (рубрика, выпуск, раздел) на любом сайте, дополнительно введи <b>название</b> статьи как в списке — сервис ищет на этой странице ссылку с похожим текстом (на том же домене) и переходит к карточке статьи. Надёжнее по-прежнему дать прямую ссылку на материал или DOI.</p>
              <p>Название в CrossRef часто <b>на английском</b>, даже при русской странице журнала: при поиске по кириллице смотри подсказки на экране результата.</p>
              <p>После Parse открой варианты: у каждого есть блок <b>«Почему этот вариант в списке»</b> — краткое пояснение ранжирования (сходство заголовка, год, DOI, search_score).</p>
              <p><b>Экспорт BibTeX / RIS.</b> Добавь нужные варианты кнопкой «Добавить в итоговый список». В шапке результата или в блоке итогового списка нажми <b>BibTeX</b> или <b>RIS</b> — браузер скачает файл <code>selected_references.bib</code> или <code>selected_references.ris</code> для Zotero, JabRef и т.п. Отдельно доступен JSON для скриптов: <code>/api/work?doi=...</code>.</p>
              <p>Список сохраняется между поисками; на экране результата можно <b>убрать последний</b> добавленный источник или <b>очистить весь список</b>. Недавние запросы (режим и краткий итог) смотри в разделе <a href="/history">История</a> (отключается переменной <code>SP_ANONYMOUS=1</code>).</p>
            </div>
          </section>
        </div>
      </div>
      <div id="loadingOverlay" class="loading-overlay">
        <div class="loading-panel">
          <div class="sk-title skeleton"></div>
          <div class="sk-line skeleton"></div>
          <div class="sk-line skeleton"></div>
          <div class="sk-line skeleton" style="width:80%;"></div>
        </div>
      </div>
      <script>
        (function() {{
          const root = document.documentElement;
          const key = "sp_theme";
          const btn = document.getElementById("themeToggle");
          function apply(theme) {{
            if (theme === "dark") {{
              root.setAttribute("data-theme", "dark");
              btn.textContent = "☀️ Light";
            }} else {{
              root.removeAttribute("data-theme");
              btn.textContent = "🌙 Dark";
            }}
          }}
          apply(localStorage.getItem(key) || "light");
          btn.addEventListener("click", function() {{
            const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
            localStorage.setItem(key, next);
            apply(next);
          }});
        }})();
        (function() {{
          const q = new URLSearchParams(window.location.search);
          const form = document.getElementById("parseForm");
          if (!form) return;
          const doi = q.get("doi");
          const url = q.get("url");
          const title = q.get("title");
          if (doi) {{
            const el = form.querySelector('[name="doi"]');
            if (el) el.value = doi;
          }}
          if (url) {{
            const el = form.querySelector('[name="url"]');
            if (el) el.value = url;
          }}
          if (title) {{
            const el = form.querySelector('[name="title"]');
            if (el) el.value = title;
          }}
        }})();
        (function() {{
          const tabs = Array.from(document.querySelectorAll("[data-tab-target]"));
          const panels = Array.from(document.querySelectorAll("[data-tab-panel]"));
          const contentRoot = document.getElementById("tabContentRoot");
          const tabsWrap = document.getElementById("topbarTabs");
          const indicator = document.getElementById("tabIndicator");
          if (!tabs.length || !panels.length || !contentRoot) return;

          function updateIndicator(targetTab) {{
            if (!tabsWrap || !indicator || !targetTab) return;
            const wrapRect = tabsWrap.getBoundingClientRect();
            const tabRect = targetTab.getBoundingClientRect();
            indicator.style.width = tabRect.width + "px";
            indicator.style.transform = "translateX(" + (tabRect.left - wrapRect.left) + "px)";
            indicator.style.opacity = "1";
          }}

          function activateTab(name, updateHash = true) {{
            const safeName = panels.some((panel) => panel.dataset.tabPanel === name) ? name : "home";
            let activeTab = null;
            tabs.forEach((tab) => {{
              const isActive = tab.dataset.tabTarget === safeName;
              tab.classList.toggle("active", isActive);
              if (isActive) activeTab = tab;
            }});
            panels.forEach((panel) => {{
              panel.classList.toggle("active", panel.dataset.tabPanel === safeName);
            }});
            updateIndicator(activeTab);
            if (updateHash) {{
              history.replaceState(null, "", "#" + (safeName === "home" ? "home" : safeName));
            }}
            contentRoot.scrollIntoView({{ behavior: "smooth", block: "start" }});
          }}

          tabs.forEach((tab) => {{
            tab.addEventListener("click", function(event) {{
              event.preventDefault();
              activateTab(tab.dataset.tabTarget);
            }});
          }});

          const initial = (window.location.hash || "#home").replace("#", "");
          activateTab(initial, false);

          window.addEventListener("hashchange", function() {{
            const next = (window.location.hash || "#home").replace("#", "");
            activateTab(next, false);
          }});
          window.addEventListener("resize", function() {{
            const activeTab = tabs.find((tab) => tab.classList.contains("active"));
            updateIndicator(activeTab);
          }});
        }})();
        (function() {{
          const form = document.getElementById("parseForm");
          const overlay = document.getElementById("loadingOverlay");
          const submitBtn = document.getElementById("parseSubmitBtn");
          if (!form || !overlay || !submitBtn) return;
          form.addEventListener("submit", function() {{
            submitBtn.disabled = true;
            submitBtn.textContent = "Идет анализ...";
            overlay.style.display = "flex";
          }});
        }})();
      </script>
    </body>
    </html>
    """


@app.get("/save_candidate")
@limiter.limit("30/minute")
def save_candidate():
    raw = (request.args.get("data") or "").strip()
    if not raw:
        return jsonify({"ok": False, "message": "Нет данных"}), 400
    if len(raw) > MAX_PAYLOAD_LEN:
        return jsonify({"ok": False, "message": "Слишком большой payload"}), 400
    try:
        payload = json.loads(raw)
        item = PaperMetadata(**payload)
    except Exception as e:
        return jsonify({"ok": False, "message": f"Ошибка чтения данных: {e}"}), 400

    inserted = _db_save_selected(_get_session_id(), item)
    if not inserted:
        return jsonify({"ok": True, "message": "Уже добавлено"})
    return jsonify({"ok": True, "message": "Добавлено в итоговый список"})


@app.get("/clear_selected")
@limiter.limit("10/minute")
def clear_selected():
    _db_clear_selected(_get_session_id())
    return jsonify({"ok": True, "message": "Итоговый список очищен"})


@app.get("/pop_last_selected")
@limiter.limit("30/minute")
def pop_last_selected():
    sid = _get_session_id()
    if not _db_pop_last_selected(sid):
        return jsonify({"ok": False, "message": "Итоговый список уже пуст"})
    return jsonify({"ok": True, "message": "Последняя добавленная запись удалена"})


@app.get("/download_pdf")
@limiter.limit("20/minute")
def download_pdf():
    pdf_url = (request.args.get("pdf_url") or "").strip()
    doi = (request.args.get("doi") or "").strip()
    source_url = (request.args.get("source_url") or "").strip()
    title = (request.args.get("title") or "article").strip()

    err = (
        _validate_input_length(pdf_url, MAX_URL_LEN, "pdf_url")
        or _validate_input_length(source_url, MAX_URL_LEN, "source_url")
        or _validate_input_length(doi, MAX_DOI_LEN, "doi")
    )
    if err:
        return err, 400

    candidates = _build_pdf_candidates(pdf_url=pdf_url, doi=doi, source_url=source_url)
    if not candidates:
        return (
            "Не удалось определить прямую ссылку на PDF. "
            "Вероятно, у издателя закрытый доступ. Попробуйте другую статью или Open Access версию.",
            404,
        )

    pdf_response, used_url, err = _try_download_pdf_from_candidates(candidates)
    if not pdf_response:
        attempts = "<br/>".join(escape(x) for x in candidates)
        return f"{err}<br/><br/>Проверенные ссылки:<br/>{attempts}", 404

    filename = f"{_safe_filename(title)}.pdf"
    pdf_response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    if used_url:
        pdf_response.headers["X-Source-PDF-URL"] = used_url
    return pdf_response


@app.get("/probe_pdf")
@limiter.limit("40/minute")
def probe_pdf():
    pdf_url = (request.args.get("pdf_url") or "").strip()
    doi = (request.args.get("doi") or "").strip()
    source_url = (request.args.get("source_url") or "").strip()
    err = (
        _validate_input_length(pdf_url, MAX_URL_LEN, "pdf_url")
        or _validate_input_length(source_url, MAX_URL_LEN, "source_url")
        or _validate_input_length(doi, MAX_DOI_LEN, "doi")
    )
    if err:
        return jsonify({"ok": False, "message": err}), 400
    candidates = _build_pdf_candidates(pdf_url=pdf_url, doi=doi, source_url=source_url)
    ok, resolved, message = _probe_pdf_candidates(candidates)
    return jsonify(
        {
            "ok": ok,
            "message": message,
            "resolved_url": resolved,
            "checked_candidates": candidates,
        }
    )


@app.get("/export_bibtex")
@limiter.limit("20/minute")
def export_bibtex_endpoint():
    payload = export_bibtex(_db_load_selected(_get_session_id()))
    return Response(
        payload.encode("utf-8"),
        mimetype="text/plain; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="selected_references.bib"'},
    )


@app.get("/export_ris")
@limiter.limit("20/minute")
def export_ris_endpoint():
    payload = export_ris(_db_load_selected(_get_session_id()))
    return Response(
        payload.encode("utf-8"),
        mimetype="text/plain; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="selected_references.ris"'},
    )


@app.get("/api/work")
@limiter.limit("120/minute")
@cache.cached(timeout=120, query_string=True)
def api_work_metadata():
    """
    JSON метаданных: ровно один из query-параметров doi, url или title.
    Для интеграций (тонкий клиент, скрипты, Zotero и т.д.).
    """
    doi = (request.args.get("doi") or "").strip()
    url = (request.args.get("url") or "").strip()
    title = (request.args.get("title") or "").strip()
    n_params = sum(bool(x) for x in (doi, url, title))
    if n_params == 0:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Передайте один из параметров: doi, url или title",
                }
            ),
            400,
        )
    if n_params > 1:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Укажите только один параметр: doi, url или title",
                }
            ),
            400,
        )

    if doi:
        meta = extract_metadata_from_doi(doi)
        return jsonify({"ok": True, "mode": "doi", "metadata": asdict(meta)})

    if url:
        verr = _validate_input_length(url, MAX_URL_LEN, "url")
        if verr:
            return jsonify({"ok": False, "error": verr}), 400
        if not is_public_http_url(url):
            return jsonify({"ok": False, "error": "URL не разрешён политикой безопасности"}), 400
        meta = extract_metadata_from_url(url)
        return jsonify({"ok": True, "mode": "url", "metadata": asdict(meta)})

    verr = _validate_input_length(title, MAX_TITLE_LEN, "title")
    if verr:
        return jsonify({"ok": False, "error": verr}), 400
    cands = search_metadata_candidates_from_title(title, max_results=4)
    if not cands:
        p = extract_metadata_from_title(title)
        if not (p.title or p.doi):
            return jsonify({"ok": False, "error": "Не найдено", "metadata": None}), 404
        cands = [p]
    primary = cands[0]
    out: dict = {"ok": True, "mode": "title", "metadata": asdict(primary)}
    if len(cands) > 1:
        out["candidates"] = [asdict(c) for c in cands]
    return jsonify(out)


@app.get("/history")
@limiter.limit("60/minute")
def search_history_page():
    if _anonymous_mode():
        return (
            "<p>История поиска отключена (SP_ANONYMOUS=1).</p><p><a href='/'>На главную</a></p>",
            403,
        )
    sid = _get_session_id()
    rows = _get_db().execute(
        """
        SELECT created_at, mode, url, doi, title_query, result_title, ok, error_message
        FROM search_history
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT 40
        """,
        (sid,),
    ).fetchall()
    cards: List[str] = []
    for r in rows:
        raw_snippet = (r["title_query"] or "").strip() or (r["doi"] or "").strip() or (r["url"] or "").strip()
        snippet = raw_snippet[:200] + ("…" if len(raw_snippet) > 200 else "")
        if r["ok"]:
            outcome = (r["result_title"] or "—")[:200]
            outcome_html = escape(outcome)
        else:
            outcome_html = escape((r["error_message"] or "ошибка")[:300])
        cards.append(
            f"""<div class="card" style="margin-bottom:10px;padding:12px;">
            <div style="color:var(--muted);font-size:13px;">{escape(r["created_at"] or "")}</div>
            <div><b>{escape(r["mode"] or "")}</b></div>
            <div style="margin-top:6px;">Запрос: {escape(snippet) if snippet else "—"}</div>
            <div style="margin-top:6px;">Итог: {outcome_html}</div>
            </div>"""
        )
    body = "".join(cards) if cards else "<p class='empty-text'>Пока нет сохранённых запросов в этой сессии.</p>"
    return f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>История поиска</title>
      <style>{APP_THEME_CSS}</style>
    </head>
    <body>
      <div class="container">
        <div class="topbar">
          <div class="topbar-left">
            <div class="brand">📚 История</div>
            <a class="btn nav-link" href="/">Главная</a>
          </div>
          <div class="topbar-right">
            <button id="themeToggle" class="btn theme-toggle">🌙 Dark</button>
          </div>
        </div>
        <div class="hero"><h2>Последние запросы</h2>
        <p>До 40 записей для текущей сессии браузера. Отключение: <code>SP_ANONYMOUS=1</code>.</p></div>
        {body}
      </div>
      <script>
        (function() {{
          const root = document.documentElement;
          const key = "sp_theme";
          const btn = document.getElementById("themeToggle");
          function apply(theme) {{
            if (theme === "dark") {{
              root.setAttribute("data-theme", "dark");
              if (btn) btn.textContent = "☀️ Light";
            }} else {{
              root.removeAttribute("data-theme");
              if (btn) btn.textContent = "🌙 Dark";
            }}
          }}
          apply(localStorage.getItem(key) || "light");
          if (btn) {{
            btn.addEventListener("click", function() {{
              const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
              localStorage.setItem(key, next);
              apply(next);
            }});
          }}
        }})();
      </script>
    </body>
    </html>
    """


@app.post("/parse_batch")
@limiter.limit("8/hour")
def parse_batch():
    raw = (request.form.get("batch") or "").strip()
    if len(raw) > MAX_BATCH_CHARS:
        return f"Слишком большой текст пакета (max {MAX_BATCH_CHARS} символов).", 400
    citation_style = _citation_style(request.form.get("citation_style") or "gost")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()][:MAX_BATCH_LINES]
    if not lines:
        return "Введите хотя бы одну непустую строку (DOI, URL или название).", 400

    blocks: List[str] = []
    for i, line in enumerate(lines, start=1):
        try:
            if re.match(r"^10\.\d+", line):
                m = extract_metadata_from_doi(line)
                mode_line = "doi"
            elif line.lower().startswith("http://") or line.lower().startswith("https://"):
                if not is_public_http_url(line):
                    blocks.append(
                        f'<div class="card" style="border-left:4px solid var(--danger);"><b>Строка {i}</b> — URL запрещён политикой безопасности.</div>'
                    )
                    continue
                m = extract_metadata_from_url(line)
                mode_line = "url"
            else:
                m = extract_metadata_from_title(line)
                mode_line = "title"
            cit = format_citation(m, style=citation_style)
            blocks.append(
                f"""<div class="card"><div><b>Строка {i}</b> ({escape(mode_line)})</div>
                <pre class="mono-box" style="margin-top:8px;white-space:pre-wrap;">{escape(cit)}</pre></div>"""
            )
            _db_log_search(_get_session_id(), f"batch:{mode_line}", "", "", line, (m.title or "")[:500])
        except Exception as e:
            blocks.append(
                f"""<div class="card" style="border-left:4px solid var(--warning);"><b>Строка {i}</b> — ошибка: {escape(str(e))}</div>"""
            )
            _db_log_search(
                _get_session_id(),
                "batch",
                "",
                "",
                line,
                "",
                ok=False,
                error_message=str(e)[:500],
            )

    joined = "\n".join(blocks)
    return f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Пакетный разбор</title>
      <style>{APP_THEME_CSS}</style>
    </head>
    <body>
      <div class="container">
        <div class="topbar">
          <div class="topbar-left">
            <div class="brand">📚 Пакетный разбор</div>
            <a class="btn nav-link" href="/">Главная</a>
          </div>
          <div class="topbar-right">
            <button id="themeToggle" class="btn theme-toggle">🌙 Dark</button>
          </div>
        </div>
        <div class="hero"><h2>Результаты ({len(lines)} строк)</h2>
        <p>Стиль: {escape(citation_style.upper())}. Добавление в общий список из этого режима не выполняется — используйте одиночный Parse.</p></div>
        {joined}
      </div>
      <script>
        (function() {{
          const root = document.documentElement;
          const key = "sp_theme";
          const btn = document.getElementById("themeToggle");
          function apply(theme) {{
            if (theme === "dark") {{
              root.setAttribute("data-theme", "dark");
              if (btn) btn.textContent = "☀️ Light";
            }} else {{
              root.removeAttribute("data-theme");
              if (btn) btn.textContent = "🌙 Dark";
            }}
          }}
          apply(localStorage.getItem(key) || "light");
          if (btn) {{
            btn.addEventListener("click", function() {{
              const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
              localStorage.setItem(key, next);
              apply(next);
            }});
          }}
        }})();
      </script>
    </body>
    </html>
    """


@app.post("/parse")
@limiter.limit("20/minute")
def parse():
    url = (request.form.get("url") or "").strip()
    title_input = (request.form.get("title") or "").strip()
    doi = (request.form.get("doi") or "").strip()
    citation_style = (request.form.get("citation_style") or "gost").strip().lower()
    max_variants_raw = (request.form.get("max_variants") or "5").strip()
    try:
        max_variants = max(1, min(10, int(max_variants_raw)))
    except ValueError:
        max_variants = 5
    citation_style = _citation_style(citation_style)

    err = _validate_input_length(url, MAX_URL_LEN, "url") or _validate_input_length(title_input, MAX_TITLE_LEN, "title") or _validate_input_length(doi, MAX_DOI_LEN, "doi")
    if err:
        return err, 400
    if url and not is_public_http_url(url):
        return "URL не разрешен политикой безопасности.", 400

    if not url and not title_input and not doi:
        return "Введите URL или название или DOI", 400

    mode_used = "unknown"
    candidates: List[PaperMetadata] = []
    interpretation = None
    listing_match_failed = False
    generic_listing_match_failed = False

    if doi:
        mode_used = "doi"
        meta = extract_metadata_from_doi(doi)
        # Для DOI возвращаем ровно одну целевую статью (точное совпадение по идентификатору).
        candidates = [meta]
    elif url:
        mode_used = "url"
        meta = None
        tit = (title_input or "").strip()
        candidates_built = False
        if tit and is_journal_rubric_list_url(url):
            resolved = resolve_article_from_journal_listing(url, tit)
            if resolved and (resolved.title or resolved.doi):
                meta = resolved
                mode_used = "url_listing_title_match"
            else:
                listing_match_failed = True
        if (
            meta is None
            and tit
            and len(tit) >= 4
            and is_cyberleninka_non_article_page(url)
        ):
            cl_list = search_cyberleninka_candidates_by_title(tit, per_page=max(25, max_variants))
            if cl_list:
                meta = cl_list[0]
                mode_used = "url_cyberleninka_title_search"
                related = search_metadata_candidates_from_title(meta.title or tit, max_results=max_variants)
                candidates = _merge_unique_papers(cl_list, related, max_variants)
            else:
                interpretation = explain_title_interpretation(tit)
                candidates = search_metadata_candidates_from_title(tit, max_results=max_variants)
                meta = candidates[0] if candidates else extract_metadata_from_title(tit)
                mode_used = "url_cyberleninka_title_fallback"
            candidates_built = True
        if meta is None:
            meta = extract_metadata_from_url(url)
        if not candidates_built:
            if not meta.title and not meta.doi:
                guessed_title = None
                try:
                    html = fetch_html(url)
                    guessed_title = extract_title_from_html_basic(html)
                except Exception:
                    guessed_title = None
                if not guessed_title:
                    guessed_title = guess_title_from_url(url)
                if guessed_title:
                    meta2 = extract_metadata_from_title(guessed_title)
                    if meta2 and (meta2.title or meta2.doi):
                        mode_used = "title_fallback"
                        meta2.source_url = url
                        meta = meta2
            tit_cmp = (title_input or "").strip()
            if (
                tit_cmp
                and len(tit_cmp) >= 6
                and mode_used
                not in (
                    "url_listing_title_match",
                    "url_cyberleninka_title_search",
                    "url_cyberleninka_title_fallback",
                )
            ):
                page_score = (meta.confidence or {}).get("page_article_score")
                try:
                    ps_val = float(page_score) if page_score is not None else None
                except (TypeError, ValueError):
                    ps_val = None
                page_is_article = ps_val is not None and ps_val >= 0.45
                meta_title = (meta.title or "").strip()
                sim = similarity_normalized_titles(tit_cmp, meta_title) if meta_title else 0.0
                needs_listing_scan = (not meta_title) or (not page_is_article) or (sim < 0.42)
                if needs_listing_scan:
                    resolved_g = resolve_article_from_listing_by_title(url, tit_cmp)
                    u_meta = (resolved_g.source_url or "").split("#")[0].rstrip("/").lower() if resolved_g else ""
                    u_in = (url or "").split("#")[0].rstrip("/").lower()
                    if resolved_g and u_meta and u_meta != u_in and (resolved_g.title or resolved_g.doi):
                        meta = resolved_g
                        mode_used = "url_generic_title_match"
                    elif needs_listing_scan:
                        generic_listing_match_failed = True
            related = search_metadata_candidates_from_title(meta.title or "", max_results=max_variants) if meta.title else []
            candidates = _merge_unique_papers([meta], related, max_variants)
    else:
        mode_used = "title_multi"
        interpretation = explain_title_interpretation(title_input)
        candidates = search_metadata_candidates_from_title(title_input, max_results=max_variants)
        meta = candidates[0] if candidates else extract_metadata_from_title(title_input)

    best_info = _fulltext_info(meta)
    bibliography_text = format_bibliography_list(candidates, style=citation_style)
    status_class = "status-bad"
    status_icon = "⛔"
    if best_info.get("status") == "direct_pdf_candidate":
        status_class = "status-ok"
        status_icon = "✅"
    elif best_info.get("status") == "open_access_candidate":
        status_class = "status-warn"
        status_icon = "🟡"
    page_cls_block = ""
    page_score = (meta.confidence or {}).get("page_article_score")
    page_reason = (meta.confidence or {}).get("page_article_reason")
    if page_score is not None:
        is_article = "да" if float(page_score) >= 0.45 else "нет"
        page_cls_block = f"""
        <div class="card">
        <h3 class="section-title">Классификатор страницы</h3>
        <pre class="mono-box">Похожа на страницу статьи: {is_article}
Оценка: {escape(str(page_score))}
Причина: {escape(str(page_reason or "нет"))}</pre>
        </div>
        """

    qual_block = ""
    _mc = meta.confidence or {}
    if float(_mc.get("doi_invalid") or 0) >= 1.0:
        qual_block += """
        <div class="card" style="border-left:4px solid var(--warning);">
          <h3 class="section-title">Проверка DOI</h3>
          <p>Строка не похожа на корректный DOI (часто при копировании из PDF или OCR). Проверьте префикс <code>10.</code>, суффикс после «/» и отсутствие пробелов.</p>
        </div>
        """
    if float(_mc.get("retracted_openalex") or 0) >= 1.0:
        qual_block += """
        <div class="card" style="border-left:4px solid var(--danger);">
          <h3 class="section-title">Возможная рестракция</h3>
          <p>По данным OpenAlex запись помечена как retracted. Уточните статус на сайте издателя.</p>
        </div>
        """

    interpretation_block = ""
    if interpretation and interpretation.get("variants"):
        variants = ", ".join(interpretation.get("variants", [])[:4])
        interpretation_block = f"""
        <div class="card">
        <h3 class="section-title">Я вас понял так</h3>
        <pre class="mono-box">Исходный запрос: {escape(interpretation.get("original", ""))}
Нормализованные варианты: {escape(variants)}
Предполагаемая корректировка: {escape(interpretation.get("suggested") or "не требуется")}</pre>
        </div>
        """

    journal_rubric_block = ""
    if url:
        journal_rubric_block = _journal_rubric_usage_block(
            url,
            title_input,
            listing_match_failed=listing_match_failed,
            page_is_listing=is_journal_rubric_list_url(url),
        )

    generic_listing_block = _generic_listing_hint_block(
        generic_match_failed=generic_listing_match_failed,
        title_input=title_input or "",
    )

    language_hint_block = ""
    if (title_input or "").strip() and candidates and mode_used != "doi":
        language_hint_block = _language_catalog_hint_block(title_input, candidates)

    selected_papers = _db_load_selected(_get_session_id())
    selected_block = ""
    if selected_papers:
        selected_block = f"""
        <div class="card">
          <h3 class="section-title">Итоговый выбранный список ({len(selected_papers)})</h3>
          <pre class="mono-box">{escape(format_bibliography_list(selected_papers, style=citation_style))}</pre>
          <div class="mt-10 action-row">
            <a class="btn btn-soft" href="/export_bibtex">Экспорт BibTeX</a>
            <a class="btn btn-soft" href="/export_ris">Экспорт RIS</a>
            <button type="button" class="btn btn-soft" id="popLastSelectedBtn">Убрать последний</button>
            <button type="button" class="btn btn-warn" id="clearAllSelectedBtn">Очистить весь список</button>
          </div>
        </div>
        """

    variants_html = ""
    for idx, item in enumerate(candidates, start=1):
        score = f"{item.search_score:.3f}" if item.search_score is not None else "n/a"
        full = _fulltext_info(item)
        item_status_class = "status-bad"
        item_status_icon = "⛔"
        if full.get("status") == "direct_pdf_candidate":
            item_status_class = "status-ok"
            item_status_icon = "✅"
        elif full.get("status") == "open_access_candidate":
            item_status_class = "status-warn"
            item_status_icon = "🟡"
        links = "<br/>".join(
            f'<a href="{escape(link)}" target="_blank">{escape(link)}</a>' for link in full["links"][:3]
        ) or "нет альтернативных ссылок"

        payload = quote_plus(json.dumps(asdict(item), ensure_ascii=False))
        save_btn = (
            f'<button class="btn btn-soft save-candidate-btn" '
            f'data-payload="{escape(payload, quote=True)}" '
            f'>Добавить в итоговый список</button>'
        )
        pdf_btn = (
            f'<button class="btn btn-primary download-pdf-btn" '
            f'data-pdf-url="{escape(item.pdf_url or "", quote=True)}" '
            f'data-doi="{escape(item.doi or "", quote=True)}" '
            f'data-source-url="{escape(item.source_url or "", quote=True)}" '
            f'data-title="{escape(item.title or f"article_{idx}", quote=True)}" '
            f'>Скачать полный PDF</button>'
        )

        rank_expl = explain_candidate_ranking(
            (title_input or "").strip() or None,
            (doi or "").strip() or None,
            item,
        )
        open_attr = "open" if idx <= 1 else ""
        variants_html += f"""
        <details class="variant-collapse" {open_attr}>
          <summary>
            <span>Вариант {idx}</span>
            <span class="summary-score">score: {escape(score)}</span>
          </summary>
          <div class="variant-body">
            <div class="row"><b>Citation:</b> {escape(format_citation(item, style=citation_style))}</div>
            <div class="row"><span class="status-chip {item_status_class}">{item_status_icon} {escape(full["label"])}</span></div>
            <div class="row"><b>Почему этот вариант в списке:</b> {escape(rank_expl)}</div>
            <div class="row"><b>Комментарий:</b> {escape(full["reason"])}</div>
            <div class="row"><b>Ссылки:</b><br/>{links}</div>
            <div class="row action-row">{pdf_btn} {save_btn}</div>
            <div class="hint">Кнопка добавляет этот вариант в общий список для экспорта BibTeX/RIS.</div>
          </div>
        </details>
        """

    _db_log_search(
        _get_session_id(),
        mode_used,
        url,
        doi,
        title_input,
        (meta.title or "")[:500],
    )

    return f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>BiblioParser Result</title>
      <style>{APP_THEME_CSS}</style>
    </head>
    <body>
      <div class="container">
        <div class="topbar">
          <div class="topbar-left">
            <div class="brand">📚 BiblioParser</div>
            <div class="result-nav">
              <a class="btn nav-link" href="/">Главная</a>
              <a class="btn nav-link" href="/#about">О проекте</a>
              <a class="btn nav-link" href="/#howto">Как пользоваться</a>
              <a class="btn" href="/">Новый поиск</a>
              <a class="btn btn-soft" href="/export_bibtex">BibTeX</a>
              <a class="btn btn-soft" href="/export_ris">RIS</a>
              <a class="btn btn-soft" href="/history">История</a>
            </div>
          </div>
          <div class="topbar-right">
            <button id="themeToggle" class="btn theme-toggle">🌙 Dark</button>
          </div>
        </div>

        <div class="hero">
          <h2>Результат анализа</h2>
          <p>Проверь найденные данные, выбери нужные источники и экспортируй список.</p>
        </div>

        <div class="card">
          <a class="btn" href="/">← Назад к поиску</a>
          <div class="meta-row mt-12">
            <div class="meta-item">
              <div class="k">Mode</div>
              <div class="v">{escape(mode_used)}</div>
            </div>
            <div class="meta-item">
              <div class="k">Стиль</div>
              <div class="v">{escape(citation_style.upper())}</div>
            </div>
          </div>
        </div>

        {interpretation_block}
        {journal_rubric_block}
        {generic_listing_block}
        {language_hint_block}
        {page_cls_block}
        {qual_block}

        <div class="card">
          <h3 class="section-title">Блок полного текста (лучший результат)</h3>
          <div class="mb-8">
            <span class="status-chip {status_class}">{status_icon} {escape(best_info["label"])}</span>
          </div>
          <pre class="mono-box">Статус: {escape(best_info["label"])}
Комментарий: {escape(best_info["reason"])}</pre>
        </div>

        <div class="card">
          <h3 class="section-title">Библиографический список (текущий результат)</h3>
          <pre class="mono-box">{escape(bibliography_text)}</pre>
          <pre id="bibliography_data" style="display:none;">{escape(bibliography_text)}</pre>
          <div class="mt-10 action-row">
            <button class="btn btn-soft" onclick="copyBibliography()">Скопировать список</button>
          </div>
        </div>

        {selected_block}

        <div class="card">
          <h3 class="section-title">Найденные варианты</h3>
          {variants_html or "<p class='empty-text'>Нет вариантов.</p>"}
        </div>
      </div>

      <div id="toast" class="toast"></div>

      <script>
        (function() {{
          const root = document.documentElement;
          const key = "sp_theme";
          const btn = document.getElementById("themeToggle");
          function apply(theme) {{
            if (theme === "dark") {{
              root.setAttribute("data-theme", "dark");
              if (btn) btn.textContent = "☀️ Light";
            }} else {{
              root.removeAttribute("data-theme");
              if (btn) btn.textContent = "🌙 Dark";
            }}
          }}
          apply(localStorage.getItem(key) || "light");
          if (btn) {{
            btn.addEventListener("click", function() {{
              const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
              localStorage.setItem(key, next);
              apply(next);
            }});
          }}
        }})();
        function showToast(message, isError=false) {{
          const toast = document.getElementById("toast");
          toast.textContent = message;
          toast.style.display = "block";
          toast.style.background = isError ? "#b32020" : "#222";
          setTimeout(() => {{
            toast.style.display = "none";
          }}, 2800);
        }}
        function copyBibliography() {{
          const node = document.getElementById("bibliography_data");
          const value = (node ? node.textContent : "").trim();
          navigator.clipboard.writeText(value)
            .then(() => showToast("Библиографический список скопирован"))
            .catch(() => showToast("Не удалось скопировать список", true));
        }}
        function sanitizeFilename(input) {{
          const cleaned = (input || "article").replace(/[^\\w\\-]+/g, "_").replace(/^_+|_+$/g, "");
          return (cleaned || "article").slice(0, 80) + ".pdf";
        }}
        function downloadPdf(pdfUrl, doi, sourceUrl, title) {{
          const params = new URLSearchParams({{
            pdf_url: pdfUrl || "",
            doi: doi || "",
            source_url: sourceUrl || "",
            title: title || "article",
          }});
          // Открываем вкладку синхронно по клику, чтобы браузер не блокировал загрузку.
          const downloadWindow = window.open("about:blank", "_blank");
          fetch("/probe_pdf?" + params.toString())
            .then(async (r) => {{
              const data = await r.json();
              if (!data.ok) {{
                throw new Error(data.message || "Не удалось получить PDF");
              }}
              if (downloadWindow) {{
                downloadWindow.location.href = "/download_pdf?" + params.toString();
              }} else {{
                window.location.href = "/download_pdf?" + params.toString();
              }}
              showToast("Начата загрузка PDF");
            }})
            .catch((e) => {{
              if (downloadWindow) {{
                downloadWindow.close();
              }}
              showToast(e.message || "Не удалось скачать PDF", true);
            }});
        }}
        function saveCandidate(encodedPayload) {{
          const url = "/save_candidate?data=" + encodedPayload;
          fetch(url).then(r => r.json()).then(data => {{
            showToast(data.message || "Готово");
            window.location.reload();
          }}).catch(() => showToast("Не удалось сохранить вариант", true));
        }}
        function bindSelectedListButtons() {{
          const popBtn = document.getElementById("popLastSelectedBtn");
          const clearBtn = document.getElementById("clearAllSelectedBtn");
          if (popBtn) {{
            popBtn.addEventListener("click", () => {{
              fetch("/pop_last_selected")
                .then((r) => r.json())
                .then((data) => {{
                  showToast(data.message || "Готово", !data.ok);
                  if (data.ok) window.location.reload();
                }})
                .catch(() => showToast("Не удалось обновить список", true));
            }});
          }}
          if (clearBtn) {{
            clearBtn.addEventListener("click", () => {{
              if (!confirm("Удалить все статьи из итогового списка?")) return;
              fetch("/clear_selected")
                .then((r) => r.json())
                .then((data) => {{
                  showToast(data.message || "Готово", !data.ok);
                  if (data.ok) window.location.reload();
                }})
                .catch(() => showToast("Не удалось очистить список", true));
            }});
          }}
        }}
        function bindActionButtons() {{
          document.querySelectorAll(".download-pdf-btn").forEach((btn) => {{
            btn.addEventListener("click", () => {{
              downloadPdf(
                btn.dataset.pdfUrl || "",
                btn.dataset.doi || "",
                btn.dataset.sourceUrl || "",
                btn.dataset.title || "article"
              );
            }});
          }});
          document.querySelectorAll(".save-candidate-btn").forEach((btn) => {{
            btn.addEventListener("click", () => {{
              saveCandidate(btn.dataset.payload || "");
            }});
          }});
        }}
        bindActionButtons();
        bindSelectedListButtons();
      </script>
    </body>
    </html>
    """


_init_db()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

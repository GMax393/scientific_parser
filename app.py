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
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from inference_pipeline import (
    PaperMetadata,
    explain_title_interpretation,
    export_bibtex,
    export_ris,
    extract_metadata_from_doi,
    extract_metadata_from_title,
    extract_metadata_from_url,
    extract_title_from_html_basic,
    fetch_html,
    format_bibliography_list,
    format_citation,
    guess_title_from_url,
    search_metadata_candidates_from_title,
)
from net_security import is_public_http_url

app = Flask(__name__)
LOGGER = logging.getLogger(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me-in-production")

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "app_state.db")
MAX_URL_LEN = 2048
MAX_TITLE_LEN = 350
MAX_DOI_LEN = 255
MAX_PAYLOAD_LEN = 50000
MAX_PDF_BYTES = 25 * 1024 * 1024

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["120 per hour", "30 per minute"],
    storage_uri="memory://",
)

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
def health():
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


def _query_unpaywall_pdf_url(doi: str) -> Optional[str]:
    doi = (doi or "").strip()
    if not doi:
        return None
    try:
        r = requests.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": "scientific.parser.demo@gmail.com"},
            timeout=20,
            headers={"User-Agent": "scientific-parser/1.0"},
        )
        if r.status_code != 200:
            return None
        data = r.json() or {}
    except Exception:
        return None

    best = data.get("best_oa_location") or {}
    for key in ("url_for_pdf", "url"):
        value = (best.get(key) or "").strip()
        if value:
            return value

    for loc in data.get("oa_locations") or []:
        if not isinstance(loc, dict):
            continue
        for key in ("url_for_pdf", "url"):
            value = (loc.get(key) or "").strip()
            if value:
                return value
    return None


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
    add(_extract_arxiv_pdf_candidate(source_url))
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


def _probe_pdf_candidates(candidates: List[str]) -> Tuple[bool, Optional[str], str]:
    for candidate in candidates:
        try:
            parsed = urlparse(candidate)
            origin = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else ""
            r = requests.get(
                candidate,
                timeout=30,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "application/pdf,*/*;q=0.8",
                    "Referer": origin or "https://www.google.com/",
                },
                allow_redirects=True,
                stream=True,
            )
            if r.status_code != 200:
                continue
            content_type = (r.headers.get("Content-Type") or "").lower()
            first_chunk = b""
            for chunk in r.iter_content(chunk_size=16):
                first_chunk = chunk or b""
                break
            starts_like_pdf = first_chunk.startswith(b"%PDF")
            if "pdf" in content_type or starts_like_pdf:
                return True, candidate, "PDF доступен для скачивания."
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


@app.get("/")
def index():
    return f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Scientific Parser Demo</title>
      <style>{APP_THEME_CSS}</style>
    </head>
    <body>
      <div class="container">
        <div class="topbar">
          <div class="topbar-left">
            <div class="brand">📚 Scientific Parser</div>
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
            <button id="themeToggle" class="btn theme-toggle">🌙 Dark</button>
          </div>
        </div>

        <div class="hero">
          <h1>Scientific Parser Demo</h1>
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
                    </select>
                  </div>
                </div>

                <div style="margin-top:14px;">
                  <button id="parseSubmitBtn" class="btn btn-primary">Parse</button>
                </div>
              </form>
            </div>
          </section>
          <section id="panel-about" class="tab-panel" data-tab-panel="about">
            <div class="card info-card">
              <h3 class="section-title">О проекте</h3>
              <p><b>Scientific Parser</b> помогает быстро извлекать метаданные научных статей по `URL`, `DOI` или названию.</p>
              <p>Сервис показывает несколько релевантных вариантов, проверяет доступность полного текста и помогает собрать аккуратный библиографический список для учебной или исследовательской работы.</p>
            </div>
          </section>
          <section id="panel-howto" class="tab-panel" data-tab-panel="howto">
            <div class="card info-card">
              <h3 class="section-title">Как пользоваться</h3>
              <p>Введи `URL`, `DOI` или название статьи, затем нажми `Parse` и дождись результатов анализа.</p>
              <p>После этого выбери подходящий вариант, при необходимости скачай PDF, добавь запись в итоговый список и экспортируй ссылки в нужном формате.</p>
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
    return "Итоговый список очищен. Вернитесь назад и обновите страницу."


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
    if citation_style not in ("gost", "apa", "ieee"):
        citation_style = "gost"

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

    if doi:
        mode_used = "doi"
        meta = extract_metadata_from_doi(doi)
        # Для DOI возвращаем ровно одну целевую статью (точное совпадение по идентификатору).
        candidates = [meta]
    elif url:
        mode_used = "url"
        meta = extract_metadata_from_url(url)
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
            <div class="row"><b>Комментарий:</b> {escape(full["reason"])}</div>
            <div class="row"><b>Ссылки:</b><br/>{links}</div>
            <div class="row action-row">{pdf_btn} {save_btn}</div>
            <div class="hint">Кнопка добавляет этот вариант в общий список для экспорта BibTeX/RIS.</div>
          </div>
        </details>
        """

    return f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Scientific Parser Result</title>
      <style>{APP_THEME_CSS}</style>
    </head>
    <body>
      <div class="container">
        <div class="topbar">
          <div class="topbar-left">
            <div class="brand">📚 Scientific Parser</div>
            <div class="result-nav">
              <a class="btn nav-link" href="/">Главная</a>
              <a class="btn nav-link" href="/#about">О проекте</a>
              <a class="btn nav-link" href="/#howto">Как пользоваться</a>
              <a class="btn" href="/">Новый поиск</a>
              <a class="btn btn-soft" href="/export_bibtex">BibTeX</a>
              <a class="btn btn-soft" href="/export_ris">RIS</a>
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
        {page_cls_block}

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
      </script>
    </body>
    </html>
    """


_init_db()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

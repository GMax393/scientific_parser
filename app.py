import base64
import json
import logging
import os
import re
import sqlite3
import time
import uuid
from dataclasses import asdict
from html import escape
from typing import Dict, List, Optional, Tuple
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
    export_csl_json,
    export_latex_thebibliography,
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
    journal_indices_for,
    paper_completeness,
    paper_provenance,
    parse_citation_string,
    query_has_cyrillic,
    resolve_article_from_journal_listing,
    resolve_article_from_listing_by_title,
    search_cyberleninka_candidates_by_title,
    search_metadata_candidates_from_title,
    similarity_normalized_titles,
    validate_draft_text,
)
from net_security import explain_public_http_url, is_public_http_url, probe_http_redirect_chain

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
MAX_CITATION_LINE = 4000
MAX_DRAFT_VALIDATE_LINES = 200

CITATION_STYLES = ("gost", "apa", "ieee", "journal_auto", "springer", "nature")

_JOURNAL_INDEX_LABELS = {"vak": "ВАК", "scopus": "Scopus", "wos": "WoS", "rsci": "РИНЦ"}


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
.journal-index-pill {
  font-size: 12px;
  padding: 3px 10px;
  border-radius: 999px;
  background: #e0f2fe;
  border: 1px solid #7dd3fc;
  color: #0369a1;
}
[data-theme="dark"] .journal-index-pill {
  background: #0c4a6e;
  border-color: #38bdf8;
  color: #e0f2fe;
}
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
.busy-bar {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: color-mix(in srgb, var(--accent) 18%, transparent);
  z-index: 200;
  pointer-events: none;
  opacity: 0;
  overflow: hidden;
  transition: opacity 0.25s ease;
}
.busy-bar::before {
  content: "";
  position: absolute;
  top: 0;
  left: -40%;
  height: 100%;
  width: 40%;
  background: linear-gradient(90deg, transparent, var(--accent), #60a5fa, transparent);
  animation: busy-bar-slide 1.1s linear infinite;
}
.busy-bar.is-active { opacity: 1; }
@keyframes busy-bar-slide {
  0% { left: -40%; }
  100% { left: 100%; }
}
.btn[disabled],
.btn.is-busy {
  opacity: 0.78;
  cursor: progress;
}
.btn.is-busy {
  position: relative;
  padding-right: 34px;
  pointer-events: none;
}
.btn.is-busy::after {
  content: "";
  position: absolute;
  right: 12px;
  top: 50%;
  width: 14px;
  height: 14px;
  margin-top: -7px;
  border: 2px solid currentColor;
  border-top-color: transparent;
  border-radius: 50%;
  animation: btn-spin 0.7s linear infinite;
}
@keyframes btn-spin {
  to { transform: rotate(360deg); }
}
.batch-status {
  margin-top: 14px;
  background: var(--card);
  border: 1px solid var(--border);
  border-left: 4px solid var(--accent);
  border-radius: 12px;
  padding: 12px 14px;
  box-shadow: var(--shadow);
}
.batch-status-row {
  display: flex;
  align-items: center;
  gap: 12px;
}
.batch-status .spinner {
  flex: 0 0 auto;
  width: 18px;
  height: 18px;
  border: 2px solid var(--accent);
  border-top-color: transparent;
  border-radius: 50%;
  animation: btn-spin 0.7s linear infinite;
}
.batch-status[data-state="done"] {
  border-left-color: #0f766e;
}
.batch-status[data-state="done"] .spinner {
  border-color: #0f766e;
  border-top-color: transparent;
  animation: none;
  background: #0f766e;
  border-style: solid;
  position: relative;
}
.batch-status[data-state="done"] .spinner::after {
  content: "✓";
  position: absolute;
  inset: 0;
  color: #fff;
  font-size: 12px;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
}
.batch-status[data-state="error"] { border-left-color: #b91c1c; }
.batch-status[data-state="error"] .spinner {
  border-color: #b91c1c;
  animation: none;
  background: #b91c1c;
  position: relative;
}
.batch-status[data-state="error"] .spinner::after {
  content: "!";
  position: absolute;
  inset: 0;
  color: #fff;
  font-size: 12px;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
}
.batch-status .label { font-weight: 600; font-size: 14px; color: var(--text); }
.batch-status .sub { color: var(--muted); font-size: 12px; margin-top: 2px; }
.batch-progress-track {
  position: relative;
  height: 6px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--accent) 12%, transparent);
  overflow: hidden;
  margin-top: 10px;
}
.batch-progress-fill {
  position: absolute;
  inset: 0 auto 0 0;
  width: 0;
  background: linear-gradient(90deg, var(--accent), #60a5fa);
  border-radius: 999px;
  transition: width 0.35s ease;
}
.batch-line-status {
  margin-top: 8px;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.batch-line-status .pill {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  padding: 3px 8px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: color-mix(in srgb, var(--card) 90%, var(--accent) 10%);
  color: var(--muted);
}
.batch-line-status .pill.ok { color: #065f46; background: #ecfdf5; border-color: #a7f3d0; }
.batch-line-status .pill.err { color: #991b1b; background: #fef2f2; border-color: #fecaca; }
.batch-line-status .pill.run {
  color: #1e3a8a;
  background: #eef2ff;
  border-color: #c7d2fe;
}
[data-theme="dark"] .batch-line-status .pill.ok { color: #6ee7b7; background: #042f2e; border-color: #115e59; }
[data-theme="dark"] .batch-line-status .pill.err { color: #fda4af; background: #3f1018; border-color: #7f1d1d; }
[data-theme="dark"] .batch-line-status .pill.run { color: #93c5fd; background: #1e293b; border-color: #1d4ed8; }
.batch-error-list {
  margin-top: 10px;
  border-top: 1px solid var(--border);
  padding-top: 10px;
}
.batch-error-list ul { margin: 4px 0 0; padding-left: 18px; }
.batch-error-list li { color: var(--text); font-size: 13px; margin-bottom: 4px; }
.batch-error-list code { word-break: break-all; }
.batch-dropzone {
  position: relative;
  border: 1px dashed color-mix(in srgb, var(--accent) 35%, var(--border) 65%);
  border-radius: 10px;
  padding: 8px;
  background: color-mix(in srgb, var(--card) 92%, var(--accent) 8%);
  transition: background 0.2s ease, border-color 0.2s ease;
}
.batch-dropzone.is-drag-over {
  background: color-mix(in srgb, var(--card) 75%, var(--accent) 25%);
  border-color: var(--accent);
}
.batch-dropzone textarea {
  width: 100%;
  padding: 10px;
  border-radius: 8px;
  border: 1px solid var(--border);
  font-family: inherit;
  background: var(--card);
  color: var(--text);
  resize: vertical;
}
.batch-dropzone .dropzone-hint {
  margin-top: 6px;
  color: var(--muted);
  font-size: 12px;
}
.link-btn {
  background: none;
  border: 0;
  padding: 0;
  color: #1d4ed8;
  cursor: pointer;
  text-decoration: underline;
  font-size: inherit;
}
[data-theme="dark"] .link-btn { color: #93c5fd; }
.batch-counters {
  margin-top: 6px;
  font-size: 12px;
  color: var(--muted);
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
}
.batch-counters .dot { opacity: 0.5; }
.batch-limit-warn {
  color: #b45309;
  background: #fff7ed;
  border: 1px solid #fed7aa;
  border-radius: 999px;
  padding: 2px 8px;
}
[data-theme="dark"] .batch-limit-warn {
  color: #fcd34d;
  background: #3f2d05;
  border-color: #78350f;
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


# Общий JS-хелпер: глобальный busy-bar и состояние is-busy для любой кнопки.
# Использование:
#   SP.busy.show()      -- включить полоску активности сверху страницы
#   SP.busy.hide()      -- выключить
#   SP.busy.run(btn, async () => { ... })
#                       -- выполнить асинхронную операцию с автоматическим
#                          отображением busy-bar и блокировкой конкретной кнопки
APP_BUSY_JS = r"""
(function () {
  if (window.SP && window.SP.busy) return;
  let pending = 0;
  function ensureBar() {
    let bar = document.getElementById("globalBusyBar");
    if (!bar) {
      bar = document.createElement("div");
      bar.id = "globalBusyBar";
      bar.className = "busy-bar";
      document.body.appendChild(bar);
    }
    return bar;
  }
  function show() {
    pending += 1;
    ensureBar().classList.add("is-active");
  }
  function hide() {
    pending = Math.max(0, pending - 1);
    if (pending === 0) ensureBar().classList.remove("is-active");
  }
  function setBtnBusy(btn, busy, busyLabel) {
    if (!btn) return;
    if (busy) {
      if (!btn.dataset._origText) btn.dataset._origText = btn.textContent;
      if (busyLabel) btn.textContent = busyLabel;
      btn.classList.add("is-busy");
      btn.setAttribute("aria-busy", "true");
      btn.disabled = true;
    } else {
      btn.classList.remove("is-busy");
      btn.removeAttribute("aria-busy");
      btn.disabled = false;
      if (btn.dataset._origText) {
        btn.textContent = btn.dataset._origText;
        delete btn.dataset._origText;
      }
    }
  }
  async function run(btn, fn, opts) {
    const busyLabel = (opts && opts.busyLabel) || null;
    setBtnBusy(btn, true, busyLabel);
    show();
    try {
      return await fn();
    } finally {
      hide();
      setBtnBusy(btn, false);
    }
  }
  window.SP = window.SP || {};
  window.SP.busy = { show, hide, setBtnBusy, run };
})();
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
        # Мягкая миграция: новый столбец duration_ms для лёгкой телеметрии.
        cols = {row[1] for row in conn.execute("PRAGMA table_info(search_history)").fetchall()}
        if "duration_ms" not in cols:
            conn.execute("ALTER TABLE search_history ADD COLUMN duration_ms INTEGER")
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
    duration_ms: Optional[int] = None,
) -> None:
    if _anonymous_mode():
        return
    try:
        _get_db().execute(
            """
            INSERT INTO search_history(session_id, mode, url, doi, title_query, result_title, ok, error_message, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                int(duration_ms) if duration_ms is not None else None,
            ),
        )
        _get_db().commit()
    except Exception:
        LOGGER.exception("search_history insert failed")


def _maybe_log_url_fetch_rejection(session_id: str, url: str, meta: PaperMetadata) -> None:
    """Журнал отказов при загрузке страницы по URL (403/5xx, сеть, SSRF после редиректа)."""
    if not (url or "").strip():
        return
    ce = meta.confidence or {}
    recovered = bool((meta.title or "").strip() or (meta.doi or "").strip())
    rej_mode: Optional[str] = None
    rej_detail: Optional[str] = None
    if ce.get("fetch_blocked") and not recovered:
        rej_mode, rej_detail = "reject:ssrf", str(ce.get("fetch_blocked_reason") or "fetch_blocked")
    elif ce.get("fetch_network_error") and not recovered:
        rej_mode, rej_detail = "reject:network", "network_error"
    elif ce.get("fetch_http_status") is not None:
        try:
            code = int(float(ce["fetch_http_status"]))
        except (TypeError, ValueError):
            code = None
        if code is not None and not recovered and (code == 403 or code >= 500):
            rej_mode, rej_detail = "reject:http", f"HTTP {code}"
    if rej_mode:
        _db_log_search(
            session_id,
            rej_mode,
            url,
            "",
            "",
            "",
            ok=False,
            error_message=(rej_detail or "")[:500],
        )


def _render_url_security_card(url: str) -> str:
    """Карточка для экрана результата: объяснение SSRF-проверки и цепочка редиректов (HEAD)."""
    expl = explain_public_http_url(url)
    probe: Dict[str, object] = {}
    if expl.get("ok"):
        probe = probe_http_redirect_chain(url)
    lines = [
        f"Схема http/https: {'✓' if expl.get('scheme_ok') else '✗'}",
        f"Netloc / хост: {'✓' if expl.get('host_present') else '✗'}",
        f"Без учётных данных в URL: {'✓' if expl.get('no_credentials') else '✗'}",
        f"Хост не в блок-листе: {'✓' if expl.get('host_allowed') else '✗'}",
        f"DNS: {'✓' if expl.get('dns_ok') else '✗'}",
        f"Публичные IP после резолва: {'✓' if expl.get('ips_public') else '✗'}",
    ]
    if expl.get("resolved_ips"):
        lines.append("IP: " + ", ".join(expl["resolved_ips"]))
    if not expl.get("ok"):
        lines.append(f"Итог: блокировка ({expl.get('reason') or 'unknown'})")
    else:
        lines.append("Итог: разрешено к исходящему HTTP-запросу (до следования редиректам).")
    probe_lines: List[str] = []
    if probe:
        hops = probe.get("hops") or []
        probe_lines.append("Цепочка URL (HEAD, до смены хоста по Location):")
        for h in hops:
            probe_lines.append("  → " + str(h))
        if probe.get("last_status") is not None:
            probe_lines.append(f"Последний код ответа: {probe.get('last_status')}")
        if probe.get("blocked_redirect"):
            probe_lines.append("Редирект на непубличный URL остановлен.")
        if probe.get("probe_error"):
            probe_lines.append("Ошибка probe: " + str(probe.get("probe_error")))
    body = escape("\n".join(lines + [""] + probe_lines).strip())
    return f"""
        <div class="card">
          <h3 class="section-title">Проверка URL до загрузки HTML</h3>
          <p class="hint" style="margin-top:0;">Политика SSRF: только публичные http(s), без учётных данных в URL; каждый IP после DNS должен быть «белым». Ниже — та же проверка, что выполняется перед <code>fetch_html</code>.</p>
          <pre class="mono-box">{body}</pre>
        </div>
        """


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


def _render_checklist_block(items: List[PaperMetadata]) -> str:
    """
    Чеклист полноты по каждой принятой записи: «нет года», «нет страниц»,
    «формат DOI подозрителен» и т.п. Используется и в одиночном, и в пакетном
    выводе, чтобы заранее показать, что стоит поправить вручную перед экспортом.
    """
    if not items:
        return ""
    rows: List[str] = []
    for idx, item in enumerate(items, start=1):
        report = paper_completeness(item)
        if report["ok"] and not report["warnings"]:
            chip = '<span class="status-chip status-ok">✓ полный</span>'
        elif not report["missing"]:
            chip = f'<span class="status-chip status-warn">⚠ {len(report["warnings"])} замечаний</span>'
        else:
            chip = f'<span class="status-chip status-bad">⛔ нет {len(report["missing"])} обязательных</span>'

        details: List[str] = []
        for entry in report["missing"]:
            details.append(f'<span class="pill err">нет: {escape(entry["label"])}</span>')
        for entry in report["warnings"]:
            hint = f' — {escape(entry["hint"])}' if entry.get("hint") else ""
            details.append(f'<span class="pill run">мягко: {escape(entry["label"])}{hint}</span>')
        title_short = (item.title or item.doi or "запись").strip()
        if len(title_short) > 120:
            title_short = title_short[:117] + "…"
        rows.append(
            f"""
            <div class="row" style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-top:8px;">
              <span style="font-weight:600;min-width:24px;">{idx}.</span>
              {chip}
              <span style="color:var(--muted);font-size:13px;flex:1;">{escape(title_short)}</span>
            </div>
            <div class="batch-line-status" style="margin-top:6px;">{''.join(details) if details else ''}</div>
            """
        )
    if not rows:
        return ""
    return f"""
        <div class="card">
          <h3 class="section-title">Чеклист перед экспортом</h3>
          <p class="hint" style="margin-top:0;">
            Подсветка обязательных и условно-обязательных полей. «Мягкие» замечания
            (страницы, DOI, том) не блокируют экспорт, но стоит проверить вручную.
          </p>
          {''.join(rows)}
        </div>
    """


def _journal_index_chips_html(item: PaperMetadata) -> str:
    idxs = journal_indices_for(item)
    if not idxs:
        return ""
    chips = "".join(
        f'<span class="pill journal-index-pill">{escape(_JOURNAL_INDEX_LABELS.get(i, i.upper()))}</span>'
        for i in idxs
    )
    return (
        f'<div class="row" style="margin-top:6px;display:flex;flex-wrap:wrap;align-items:center;gap:8px;">'
        f"<b>Индексация журнала:</b> {chips}</div>"
    )


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
                  <input class="with-icon" name="url" id="mainUrlInput" placeholder="https://..." />
        </div>
                <div id="urlPreflightPanel" class="mono-box" style="margin-top:8px;display:none;font-size:13px;white-space:pre-wrap;"></div>

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
              <h3 class="section-title">Распознать строку цитирования</h3>
              <p class="hint" style="margin-top:0;">
                Вставьте готовую библиографическую строку (русский или английский текст, с DOI или без).
                Поля разбираются эвристически; при наличии данных запись сопоставляется с Crossref и при кириллице — с КиберЛенинкой.
              </p>
              <label class="label"><b>Строка</b></label>
              <div class="batch-dropzone">
                <textarea id="citationLineInput" rows="4" placeholder="Иванов И.И., Петров П.П. Название статьи // Журнал. 2020. Т. 5. № 2. С. 10–20."></textarea>
              </div>
              <div style="margin-top:12px;">
                <label class="label"><b>Стиль цитаты в ответе</b></label>
                <select id="citationParseStyle">
                  <option value="gost">ГОСТ-like</option>
                  <option value="apa">APA</option>
                  <option value="ieee">IEEE</option>
                  <option value="journal_auto">Журнал (авто по названию)</option>
                  <option value="springer">Springer-подобный</option>
                  <option value="nature">Nature-подобный</option>
                </select>
              </div>
              <div style="margin-top:12px;">
                <button type="button" class="btn btn-soft" id="citationParseBtn">Разобрать строку</button>
              </div>
              <pre class="mono-box mt-12" id="citationParseResult" style="display:none;white-space:pre-wrap;"></pre>
              <div id="citationParseMeta" class="hint mt-10" style="display:none;"></div>
            </div>

            <div class="card" style="margin-top:14px;">
              <h3 class="section-title">Проверить черновик списка литературы</h3>
              <p class="hint" style="margin-top:0;">
                Каждая непустая строка обрабатывается отдельно: распознавание, сопоставление с каталогами, отметка
                <b>обязательных пропусков</b> и <b>мягких замечаний</b>, плюс блок <b>«как оформить»</b> — цитата по уже обогащённым полям в выбранном стиле (её можно копировать в работу после проверки).
                Нумерация <code>1.</code>, <code>[1]</code> снимается автоматически.
              </p>
              <label class="label"><b>Текст списка</b></label>
              <div class="batch-dropzone">
                <textarea id="draftValidateInput" rows="8" placeholder="1. Первая ссылка в свободной форме&#10;2. Вторая ссылка…"></textarea>
              </div>
              <div style="margin-top:12px;">
                <label class="label"><b>Стиль для блока «как оформить»</b></label>
                <select id="draftCitationStyle">
                  <option value="gost">ГОСТ-like</option>
                  <option value="apa">APA</option>
                  <option value="ieee">IEEE</option>
                  <option value="journal_auto">Журнал (авто по названию)</option>
                  <option value="springer">Springer-подобный</option>
                  <option value="nature">Nature-подобный</option>
                </select>
              </div>
              <div style="margin-top:12px;">
                <button type="button" class="btn btn-soft" id="draftValidateBtn">Проверить список</button>
              </div>
              <div id="draftValidateSummary" class="batch-line-status mt-10" style="display:none;"></div>
              <div id="draftValidateBody" class="mt-10" style="display:none;"></div>
            </div>

            <div id="lastResultCard" class="card" style="margin-top:14px;border-left:4px solid var(--accent);" hidden>
              <h3 class="section-title">Последний результат (офлайн)</h3>
              <p class="hint" style="margin-top:0;">
                Сохраняется локально в браузере, чтобы продолжить работу без сети
                или быстро восстановить недавнюю цитату.
              </p>
              <div id="lastResultMeta" class="hint"></div>
              <pre class="mono-box mt-10" id="lastResultText"></pre>
              <div class="mt-10 action-row">
                <button type="button" class="btn btn-soft" id="lastResultCopyBtn">Скопировать</button>
                <button type="button" class="btn btn-soft" id="lastResultRedoBtn">Повторить запрос</button>
                <button type="button" class="btn btn-warn" id="lastResultClearBtn">Удалить</button>
              </div>
            </div>

            <div class="card" style="margin-top:14px;">
              <h3 class="section-title">Массовый импорт</h3>
              <p style="color:var(--muted);font-size:14px;">До {MAX_BATCH_LINES} строк: DOI (<code>10....</code>), URL или свободное название. После обработки выдаётся <b>единый библиографический список</b> в выбранном стиле — его можно скопировать, скачать BibTeX/RIS или сразу добавить в общий итоговый список.</p>
              <form id="batchForm" method="POST" action="/parse_batch">
                <label class="label"><b>Строки</b></label>
                <div id="batchDropZone" class="batch-dropzone">
                  <textarea id="batchInput" name="batch" rows="7" placeholder="10.1000/182&#10;https://doi.org/10.1000/182&#10;Toward Verified Artificial Intelligence"></textarea>
                  <div class="dropzone-hint">
                    Перетащите сюда <b>.txt</b> или <b>.csv</b> (одна запись на строку), либо
                    <button type="button" class="link-btn" id="batchPickFileBtn">выберите файл</button>.
                    <input type="file" id="batchFileInput" accept=".txt,.csv,text/plain,text/csv" hidden />
                  </div>
                </div>
                <div class="batch-counters">
                  <span id="batchLineCount">0 строк</span>
                  <span class="dot">·</span>
                  <span id="batchCharCount">0 / {MAX_BATCH_CHARS} символов</span>
                  <span id="batchLimitWarn" class="batch-limit-warn" hidden>превышен лимит на {MAX_BATCH_LINES} строк — лишние будут отброшены</span>
                </div>
                <div style="margin-top:10px;">
                  <label class="label"><b>Стиль цитирования</b></label>
                  <select id="batchCitationStyle" name="citation_style">
                    <option value="gost">ГОСТ-like</option>
                    <option value="apa">APA</option>
                    <option value="ieee">IEEE</option>
                    <option value="journal_auto">Журнал (авто по названию)</option>
                    <option value="springer">Springer-подобный</option>
                    <option value="nature">Nature-подобный</option>
                  </select>
                </div>
                <div style="margin-top:12px;display:flex;flex-wrap:wrap;gap:8px;align-items:center;">
                  <button type="submit" id="batchSubmitBtn" class="btn btn-soft">Разобрать пакет</button>
                  <button type="button" id="batchCancelBtn" class="btn" style="display:none;">Отменить</button>
                  <button type="button" id="batchClearInputBtn" class="btn">Очистить</button>
                  <button type="button" id="batchShareBtn" class="btn" hidden>Скопировать ссылку</button>
                  <span id="batchHint" style="color:var(--muted);font-size:12px;"></span>
                </div>
              </form>

              <div id="batchStatus" class="batch-status" hidden>
                <div class="batch-status-row">
                  <div class="spinner" aria-hidden="true"></div>
                  <div style="flex:1 1 auto;">
                    <div class="label" id="batchStatusLabel">Готовлю пакет…</div>
                    <div class="sub" id="batchStatusSub">0 / 0</div>
                  </div>
                  <div id="batchStatusCounter" class="status-chip" style="font-variant-numeric:tabular-nums;">0%</div>
                </div>
                <div class="batch-progress-track" aria-hidden="true">
                  <div class="batch-progress-fill" id="batchProgressFill"></div>
                </div>
                <div id="batchLineStatus" class="batch-line-status" aria-live="polite"></div>
              </div>

              <div id="batchResults" class="card" hidden style="margin-top:14px;">
                <h3 class="section-title">Готовый библиографический список</h3>
                <div id="batchResultsMeta" class="hint" style="margin-bottom:10px;"></div>
                <pre class="mono-box" id="batchBibliographyText"></pre>
                <div class="mt-10 action-row">
                  <button type="button" class="btn btn-soft" id="batchCopyListBtn">Скопировать список</button>
                  <button type="button" class="btn btn-soft" id="batchDownloadTxtBtn">Скачать TXT</button>
                  <button type="button" class="btn btn-soft" id="batchDownloadBibBtn">Скачать BibTeX</button>
                  <button type="button" class="btn btn-soft" id="batchDownloadRisBtn">Скачать RIS</button>
                  <button type="button" class="btn btn-soft" id="batchDownloadLatexBtn">Скачать LaTeX</button>
                  <button type="button" class="btn btn-soft" id="batchDownloadCslBtn">Скачать CSL JSON</button>
                  <button type="button" class="btn btn-primary" id="batchSaveAllBtn">Добавить весь список в итоговый</button>
                </div>
                <div id="batchChecklist" hidden></div>
                <details class="variant-collapse mt-12" id="batchPerLineWrap" hidden>
                  <summary><span>Подробно по строкам</span></summary>
                  <div class="variant-body" id="batchPerLineBody"></div>
                </details>
                <details class="variant-collapse mt-12" id="batchErrorsWrap" hidden>
                  <summary><span style="color:#b91c1c;">Не удалось обработать</span></summary>
                  <div class="variant-body" id="batchErrorsBody"></div>
                </details>
              </div>
            </div>
          </section>
          <section id="panel-about" class="tab-panel" data-tab-panel="about">
            <div class="card info-card">
              <h3 class="section-title">О проекте</h3>
              <p><b>BiblioParser</b> извлекает метаданные по URL, DOI или названию и собирает библиографию.</p>
              <p><b>Где участвует ML.</b> Классификатор блоков страницы: TF‑IDF (слово и символьные n‑граммы) + признаки DOM (тег, классы, глубина, плотность ссылок и др.) предсказывает тип каждого текстового блока — заголовок, авторы, год, журнал, DOI и т.д., чтобы вытащить поля со страницы журнала (не класс «References» целиком). Дополнительно можно включить эмбеддинги Sentence‑BERT по полю текста блока: при обучении задайте <code>USE_SBERT=1</code> и зависимости из <code>requirements-ml.txt</code>. LayoutLM и аналоги требуют отдельной разметки/инфраструктуры и не входят в базовый пайплайн.</p>
              <p><b>Нормализация запроса по названию.</b> CAPS приводятся к нормальному виду, строятся варианты раскладки (в т.ч. «ghbdtn» → «привет» через подстановку раскладки), показывается блок «Я вас понял так»; опечатки частично снимаются нечётким сопоставлением с заголовками из CrossRef.</p>
              <p><b>Безопасность и кэш.</b> Исходящие URL проверяются на SSRF в <code>net_security.py</code> (схема, запрет приватных IP после DNS, опционально <code>ALLOW_ONION=1</code> для onion‑хостов и <code>TOR_SOCKS_PROXY</code> для запросов через Tor). В процессе используется <code>lru_cache</code> для повторяющихся API‑запросов; при переменной <code>REDIS_URL</code> подключаются Redis и Flask‑Caching (см. также <code>RATELIMIT_STORAGE_URI</code> для лимитов).</p>
              <p><b>Интеграции.</b> Одиночная метаданная карточка: <code>/api/work?doi=…</code> или <code>url=…</code> или <code>title=…</code>; проверка URL до загрузки: <code>/api/url_check?url=…&amp;probe=1</code>. Пакетный NDJSON-поток (как у потоковой формы): <code>POST /api/batch</code> с JSON <code>{{"batch":"…","citation_style":"gost"}}</code>. Разбор одной библиографической строки: <code>POST /parse_citation</code> с JSON <code>{{"citation":"…","citation_style":"gost"}}</code> (поля <code>citation</code> / <code>line</code>). Проверка черновика списка литературы: <code>POST /validate_draft</code> с JSON <code>{{"text":"…","max_lines":80}}</code>. Экспорт выбранного списка: <code>/export_bibtex</code>, <code>/export_ris</code>, <code>/export_csl</code> (CSL JSON для Pandoc/Hugo). Расширение Chrome: <code>extensions/chrome</code>. Telegram: <code>telegram_bot.py</code> (<code>TELEGRAM_BOT_TOKEN</code>, <code>SP_PUBLIC_URL</code>; команды <code>/cite</code>, <code>/draft</code>).</p>
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
              <p>После Parse открой варианты: у каждого есть блок <b>«Почему этот вариант в списке»</b> — краткое пояснение ранжирования (сходство заголовка, год, DOI, search_score). Если для журнала есть запись в локальном каталоге, под полнотой показываются чипы <b>ВАК / Scopus / WoS / РИНЦ</b>.</p>
              <p>На главной доступны <b>разбор одной библиографической строки</b> и <b>проверка черновика списка литературы</b> (те же возможности, что <code>POST /parse_citation</code> и <code>POST /validate_draft</code>). В Telegram-боте: команды <code>/cite</code> и <code>/draft</code>.</p>
              <p><b>Экспорт BibTeX / RIS / CSL JSON.</b> Добавь нужные варианты кнопкой «Добавить в итоговый список». В шапке результата или в блоке списка скачай <b>BibTeX</b>, <b>RIS</b>, <b>LaTeX</b> или <b>CSL JSON</b> — последний удобен для Pandoc (<code>--citeproc</code>) и движков CSL. Для скриптов: <code>/api/work?doi=…</code>.</p>
              <p>Список сохраняется между поисками; на экране результата можно <b>убрать последний</b> добавленный источник или <b>очистить весь список</b>. Запросы и отдельно <b>отклонённые URL</b> (SSRF, 403/5xx без восстановления метаданных) — в <a href="/history">История</a>, вкладка «Отклонённые URL» (отключение: <code>SP_ANONYMOUS=1</code>).</p>
            </div>
          </section>
        </div>
      </div>
      <div id="globalBusyBar" class="busy-bar"></div>
      <div id="toast" class="toast"></div>
      <div id="loadingOverlay" class="loading-overlay">
        <div class="loading-panel">
          <div class="sk-title skeleton"></div>
          <div class="sk-line skeleton"></div>
          <div class="sk-line skeleton"></div>
          <div class="sk-line skeleton" style="width:80%;"></div>
        </div>
      </div>
      <script>{APP_BUSY_JS}</script>
      <script>
        function spShowToast(message, isError) {{
          const toast = document.getElementById("toast");
          if (!toast) return;
          toast.textContent = message;
          toast.style.display = "block";
          toast.style.background = isError ? "#b32020" : "#222";
          setTimeout(function() {{ toast.style.display = "none"; }}, 2800);
        }}
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
            window.SP.busy.show();
            window.SP.busy.setBtnBusy(submitBtn, true, "Идёт анализ…");
            overlay.style.display = "flex";
          }});
        }})();
        (function() {{
          const inp = document.getElementById("mainUrlInput");
          const panel = document.getElementById("urlPreflightPanel");
          if (!inp || !panel) return;
          let timer = null;
          function render(j) {{
            if (!j || !j.explain) {{
              panel.style.display = "none";
              return;
            }}
            const e = j.explain;
            const lines = [];
            lines.push("Проверка до отправки формы (как на сервере до загрузки HTML):");
            lines.push("  Схема http(s): " + (e.scheme_ok ? "✓" : "✗"));
            lines.push("  Хост / netloc: " + (e.host_present ? "✓" : "✗"));
            lines.push("  Без учётных данных в URL: " + (e.no_credentials ? "✓" : "✗"));
            lines.push("  Итог SSRF: " + (e.ok ? "✓ публичный URL" : ("✗ " + (e.reason || ""))));
            const rp = j.redirect_probe || {{}};
            const hops = rp.hops || [];
            if (hops.length) {{
              lines.push("Цепочка (HEAD):");
              hops.forEach(function(h) {{ lines.push("  → " + h); }});
              if (rp.last_status != null) lines.push("Последний код: " + rp.last_status);
              if (rp.probe_error) lines.push("Ошибка HEAD: " + rp.probe_error);
            }}
            panel.textContent = lines.join("\\n");
            panel.style.display = "block";
          }}
          async function check() {{
            const u = (inp.value || "").trim();
            if (!u.toLowerCase().startsWith("http")) {{
              panel.style.display = "none";
              return;
            }}
            try {{
              const r = await fetch("/api/url_check?probe=1&url=" + encodeURIComponent(u));
              const j = await r.json();
              render(j);
            }} catch (err) {{
              panel.textContent = "Не удалось выполнить проверку URL.";
              panel.style.display = "block";
            }}
          }}
          inp.addEventListener("input", function() {{
            if (timer) clearTimeout(timer);
            timer = setTimeout(check, 450);
          }});
          inp.addEventListener("blur", function() {{
            if (timer) clearTimeout(timer);
            check();
          }});
        }})();
        (function() {{
          const btn = document.getElementById("citationParseBtn");
          const ta = document.getElementById("citationLineInput");
          const st = document.getElementById("citationParseStyle");
          const out = document.getElementById("citationParseResult");
          const meta = document.getElementById("citationParseMeta");
          if (!btn || !ta || !out || !meta) return;
          btn.addEventListener("click", function() {{
            const line = (ta.value || "").trim();
            if (!line) {{
              spShowToast("Введите строку цитирования", true);
              return;
            }}
            window.SP.busy.run(btn, async function() {{
              try {{
                const r = await fetch("/parse_citation", {{
                  method: "POST",
                  headers: {{ "Content-Type": "application/json", "Accept": "application/json" }},
                  body: JSON.stringify({{
                    citation: line,
                    citation_style: (st && st.value) ? st.value : "gost",
                  }}),
                }});
                const data = await r.json();
                if (!data.ok) {{
                  spShowToast(data.error || "Ошибка разбора", true);
                  out.style.display = "none";
                  meta.style.display = "none";
                  return;
                }}
                out.style.display = "block";
                out.textContent = data.citation || "";
                meta.style.display = "block";
                const via = (data.parse_info && data.parse_info.matched_via) ? data.parse_info.matched_via : "—";
                const jidx = (data.journal_indices && data.journal_indices.length)
                  ? data.journal_indices.join(", ")
                  : "нет данных";
                const nc = (typeof data.candidates_count === "number") ? data.candidates_count : 0;
                meta.textContent = "Сопоставление: " + via
                  + " · Индексация журнала: " + jidx
                  + " · Кандидатов в каталогах: " + nc;
                spShowToast("Строка разобрана");
              }} catch (e) {{
                spShowToast("Сеть или сервер недоступны", true);
              }}
            }}, {{ busyLabel: "Разбираю…" }});
          }});
        }})();
        (function() {{
          const btn = document.getElementById("draftValidateBtn");
          const ta = document.getElementById("draftValidateInput");
          const stDraft = document.getElementById("draftCitationStyle");
          const sum = document.getElementById("draftValidateSummary");
          const body = document.getElementById("draftValidateBody");
          if (!btn || !ta || !sum || !body) return;
          function esc(s) {{
            return (s == null ? "" : String(s))
              .replace(/&/g, "&amp;")
              .replace(/</g, "&lt;")
              .replace(/>/g, "&gt;")
              .replace(/"/g, "&quot;");
          }}
          btn.addEventListener("click", function() {{
            const text = (ta.value || "").trim();
            if (!text) {{
              spShowToast("Введите текст списка литературы", true);
              return;
            }}
            window.SP.busy.run(btn, async function() {{
              try {{
                const r = await fetch("/validate_draft", {{
                  method: "POST",
                  headers: {{ "Content-Type": "application/json", "Accept": "application/json" }},
                  body: JSON.stringify({{
                    text: text,
                    citation_style: (stDraft && stDraft.value) ? stDraft.value : "gost",
                  }}),
                }});
                const data = await r.json();
                if (!data.ok) {{
                  spShowToast(data.error || "Ошибка проверки", true);
                  sum.style.display = "none";
                  body.style.display = "none";
                  return;
                }}
                const s = data.summary || {{}};
                sum.style.display = "flex";
                sum.innerHTML =
                  '<span class="pill ok">OK: ' + (s.ok || 0) + '</span>'
                  + '<span class="pill run">Частично: ' + (s.partial || 0) + '</span>'
                  + '<span class="pill err">Не найдено: ' + (s.not_found || 0) + '</span>';
                const stLabel = (data.citation_style || "gost").toUpperCase();
                const rows = (data.items || []).map(function(it) {{
                  const st0 = it.status || "";
                  const cls = st0 === "ok" ? "ok" : (st0 === "partial" ? "run" : "err");
                  const reasons = (it.reasons || []).map(function(x) {{ return esc(x); }}).join("; ");
                  const soft = (it.soft_warnings || []).map(function(x) {{ return esc(x); }}).join("; ");
                  const sug = (it.suggested_citation || "").trim();
                  const jidx = (it.journal_indices || []).join(", ");
                  let sugBlock = "";
                  if (sug) {{
                    sugBlock = '<div class="label" style="margin-top:8px;">Как оформить (' + esc(stLabel) + ')</div>'
                      + '<pre class="mono-box" style="margin-top:4px;white-space:pre-wrap;">' + esc(sug) + '</pre>';
                  }}
                  return '<div class="card" style="padding:10px;margin-bottom:8px;border-left:4px solid var(--border);">'
                    + '<div class="batch-line-status"><span class="pill ' + cls + '">#' + esc(it.index) + " — " + esc(st0) + '</span>'
                    + '<span class="pill">' + esc(it.matched_via || "—") + '</span>'
                    + (jidx ? '<span class="pill run">' + esc(jidx) + '</span>' : "")
                    + '</div>'
                    + '<div class="hint" style="margin-top:6px;word-break:break-word;"><b>Было в черновике:</b> ' + esc(it.input || "") + '</div>'
                    + (reasons ? '<div class="hint" style="margin-top:4px;color:var(--danger);"><b>Пропуски:</b> ' + reasons + '</div>' : "")
                    + (soft ? '<div class="hint" style="margin-top:4px;color:var(--warning);"><b>Замечания:</b> ' + soft + '</div>' : "")
                    + sugBlock
                    + "</div>";
                }}).join("");
                body.innerHTML = rows || '<p class="empty-text">Нет строк.</p>';
                body.style.display = "block";
                spShowToast("Проверка завершена");
              }} catch (e) {{
                spShowToast("Сеть или сервер недоступны", true);
              }}
            }}, {{ busyLabel: "Проверяю…" }});
          }});
        }})();
        (function() {{
          // Восстановление последнего удачного одиночного результата из localStorage.
          // Нужно для офлайн-демонстрации и быстрого «вернуть последнюю цитату».
          const card = document.getElementById("lastResultCard");
          if (!card) return;
          const KEY = "sp_last_result_v1";
          let raw = null;
          try {{ raw = JSON.parse(localStorage.getItem(KEY) || "null"); }} catch (e) {{ raw = null; }}
          if (!raw || !raw.citation) return;
          const meta = document.getElementById("lastResultMeta");
          const txt = document.getElementById("lastResultText");
          const copyBtn = document.getElementById("lastResultCopyBtn");
          const redoBtn = document.getElementById("lastResultRedoBtn");
          const clearBtn = document.getElementById("lastResultClearBtn");
          card.hidden = false;
          const ts = raw.savedAt ? new Date(raw.savedAt).toLocaleString() : "";
          const styleLabel = (raw.style || "gost").toUpperCase();
          meta.textContent = (ts ? (ts + " · ") : "") + "Стиль: " + styleLabel
            + (raw.mode ? (" · режим: " + raw.mode) : "");
          txt.textContent = raw.citation;
          copyBtn.addEventListener("click", function() {{
            navigator.clipboard.writeText(raw.citation || "")
              .then(function() {{ spShowToast("Цитата скопирована"); }})
              .catch(function() {{ spShowToast("Не удалось скопировать", true); }});
          }});
          redoBtn.addEventListener("click", function() {{
            const params = new URLSearchParams();
            if (raw.doi) params.set("doi", raw.doi);
            if (raw.url) params.set("url", raw.url);
            if (raw.title) params.set("title", raw.title);
            window.location.href = "/?" + params.toString() + "#home";
          }});
          clearBtn.addEventListener("click", function() {{
            localStorage.removeItem(KEY);
            card.hidden = true;
            spShowToast("Локальный кэш очищен");
          }});
        }})();
        (function() {{
          // Интерактивный пакетный импорт: NDJSON-поток с прогрессом + единый
          // библиографический список, BibTeX/RIS, групповое сохранение.
          const form = document.getElementById("batchForm");
          if (!form) return;
          const submitBtn = document.getElementById("batchSubmitBtn");
          const cancelBtn = document.getElementById("batchCancelBtn");
          const status = document.getElementById("batchStatus");
          const statusLabel = document.getElementById("batchStatusLabel");
          const statusSub = document.getElementById("batchStatusSub");
          const statusCounter = document.getElementById("batchStatusCounter");
          const progressFill = document.getElementById("batchProgressFill");
          const lineStatus = document.getElementById("batchLineStatus");
          const results = document.getElementById("batchResults");
          const resultsMeta = document.getElementById("batchResultsMeta");
          const bibText = document.getElementById("batchBibliographyText");
          const perLineWrap = document.getElementById("batchPerLineWrap");
          const perLineBody = document.getElementById("batchPerLineBody");
          const errorsWrap = document.getElementById("batchErrorsWrap");
          const errorsBody = document.getElementById("batchErrorsBody");
          const copyBtn = document.getElementById("batchCopyListBtn");
          const dlTxt = document.getElementById("batchDownloadTxtBtn");
          const dlBib = document.getElementById("batchDownloadBibBtn");
          const dlRis = document.getElementById("batchDownloadRisBtn");
          const dlLatex = document.getElementById("batchDownloadLatexBtn");
          const dlCsl = document.getElementById("batchDownloadCslBtn");
          const saveAllBtn = document.getElementById("batchSaveAllBtn");
          const hint = document.getElementById("batchHint");
          const checklistEl = document.getElementById("batchChecklist");
          const dropZone = document.getElementById("batchDropZone");
          const inputEl = document.getElementById("batchInput");
          const filePicker = document.getElementById("batchFileInput");
          const pickFileBtn = document.getElementById("batchPickFileBtn");
          const lineCountEl = document.getElementById("batchLineCount");
          const charCountEl = document.getElementById("batchCharCount");
          const limitWarn = document.getElementById("batchLimitWarn");
          const clearInputBtn = document.getElementById("batchClearInputBtn");
          const shareBtn = document.getElementById("batchShareBtn");
          const MAX_LINES = {MAX_BATCH_LINES};
          const MAX_CHARS = {MAX_BATCH_CHARS};

          let abortCtrl = null;
          let lastResult = null;

          function setStatusState(state) {{
            status.dataset.state = state || "";
          }}
          function resetStatus(total) {{
            status.hidden = false;
            setStatusState("running");
            statusLabel.textContent = total > 0
              ? "Обрабатываю строки пакета…"
              : "Готовлю пакет…";
            statusSub.textContent = "0 / " + (total || 0);
            statusCounter.textContent = "0%";
            progressFill.style.width = "0%";
            lineStatus.innerHTML = "";
            for (let i = 1; i <= (total || 0); i += 1) {{
              const pill = document.createElement("span");
              pill.className = "pill";
              pill.dataset.idx = String(i);
              pill.textContent = "#" + i;
              lineStatus.appendChild(pill);
            }}
          }}
          function markLine(idx, state, title) {{
            const pill = lineStatus.querySelector('[data-idx="' + idx + '"]');
            if (!pill) return;
            pill.classList.remove("ok", "err", "run");
            if (state) pill.classList.add(state);
            if (title) pill.title = title;
          }}
          function updateProgress(done, total) {{
            const pct = total > 0 ? Math.round((done / total) * 100) : 0;
            statusSub.textContent = done + " / " + total;
            statusCounter.textContent = pct + "%";
            progressFill.style.width = pct + "%";
          }}
          function downloadBlob(text, name, mime) {{
            const blob = new Blob([text], {{ type: (mime || "text/plain") + ";charset=utf-8" }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = name;
            document.body.appendChild(a);
            a.click();
            setTimeout(function() {{
              document.body.removeChild(a);
              URL.revokeObjectURL(url);
            }}, 200);
          }}
          function escapeHtml(s) {{
            return String(s == null ? "" : s)
              .replace(/&/g, "&amp;")
              .replace(/</g, "&lt;")
              .replace(/>/g, "&gt;")
              .replace(/"/g, "&quot;")
              .replace(/'/g, "&#39;");
          }}
          function renderChecklist(savedItems) {{
            if (!checklistEl) return;
            const rows = (savedItems || []).map(function(it, i) {{
              const c = (it && it.completeness) || null;
              if (!c) return "";
              const ok = !!c.ok && (!(c.warnings || []).length);
              const missLen = (c.missing || []).length;
              const warnLen = (c.warnings || []).length;
              let chip;
              if (ok) chip = '<span class="status-chip status-ok">✓ полный</span>';
              else if (!missLen) chip = '<span class="status-chip status-warn">⚠ ' + warnLen + ' замечаний</span>';
              else chip = '<span class="status-chip status-bad">⛔ нет ' + missLen + ' обязательных</span>';

              const pills = [].concat(
                (c.missing || []).map(function(m) {{
                  return '<span class="pill err">нет: ' + escapeHtml(m.label) + '</span>';
                }}),
                (c.warnings || []).map(function(w) {{
                  const hint = w.hint ? (' — ' + escapeHtml(w.hint)) : "";
                  return '<span class="pill run">мягко: ' + escapeHtml(w.label) + hint + '</span>';
                }})
              ).join("");
              const titleShort = ((it.metadata && (it.metadata.title || it.metadata.doi)) || it.input || "запись").slice(0, 120);
              return (
                '<div class="row" style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-top:8px;">' +
                  '<span style="font-weight:600;min-width:24px;">' + (i + 1) + '.</span>' +
                  chip +
                  '<span style="color:var(--muted);font-size:13px;flex:1;">' + escapeHtml(titleShort) + '</span>' +
                '</div>' +
                '<div class="batch-line-status" style="margin-top:6px;">' + pills + '</div>'
              );
            }}).filter(Boolean).join("");
            if (!rows) {{
              checklistEl.hidden = true;
              checklistEl.innerHTML = "";
              return;
            }}
            checklistEl.hidden = false;
            checklistEl.innerHTML = (
              '<details class="variant-collapse mt-12" open>' +
                '<summary><span>Чеклист перед экспортом</span></summary>' +
                '<div class="variant-body">' +
                  '<p class="hint" style="margin-top:0;">' +
                    'Подсветка обязательных и условно-обязательных полей. «Мягкие» замечания не блокируют экспорт.' +
                  '</p>' +
                  rows +
                '</div>' +
              '</details>'
            );
          }}
          function renderResults(data, savedItems) {{
            lastResult = Object.assign({{}}, data, {{ savedItems: savedItems }});
            results.hidden = false;
            const total = data.total || 0;
            const ok = data.ok_count || 0;
            const errs = data.error_count || 0;
            resultsMeta.textContent = "Стиль: " + (data.style || "gost").toUpperCase()
              + ". Обработано: " + ok + " из " + total
              + (errs ? (", не удалось: " + errs) : "") + ".";
            bibText.textContent = data.bibliography || "";
            const lines = (savedItems || []).map(function(it, i) {{
              const cit = (it && it.citation) || "";
              const dur = (it && it.duration_ms) ? ' <span class="hint">(' + it.duration_ms + ' мс)</span>' : "";
              return '<div class="row"><b>' + (i + 1) + '.</b> ' + escapeHtml(cit) + dur + "</div>";
            }}).join("");
            if (lines) {{
              perLineBody.innerHTML = lines;
              perLineWrap.hidden = false;
            }} else {{
              perLineWrap.hidden = true;
            }}
            const errList = (data.errors || []).map(function(e) {{
              return "<li>Строка " + e.index + ": <code>" + escapeHtml(e.input) + "</code> — " + escapeHtml(e.error) + "</li>";
            }}).join("");
            if (errList) {{
              errorsBody.innerHTML = "<ul>" + errList + "</ul>";
              errorsWrap.hidden = false;
            }} else {{
              errorsWrap.hidden = true;
            }}
            renderChecklist(savedItems);
            if (shareBtn) shareBtn.hidden = false;
            saveAllBtn.disabled = ok === 0;
          }}
          function resetUI() {{
            status.hidden = true;
            results.hidden = true;
            lineStatus.innerHTML = "";
            cancelBtn.style.display = "none";
            hint.textContent = "";
            lastResult = null;
          }}

          form.addEventListener("submit", async function(ev) {{
            ev.preventDefault();
            const fd = new FormData(form);
            const text = (fd.get("batch") || "").toString().trim();
            if (!text) {{
              spShowToast("Введите хотя бы одну строку", true);
              return;
            }}
            const lineCount = text.split(/\\r?\\n/).filter(function(s) {{ return s.trim(); }}).length;
            resultsMeta && (resultsMeta.textContent = "");
            resetStatus(lineCount);
            results.hidden = true;
            cancelBtn.style.display = "inline-block";
            hint.textContent = lineCount > 4
              ? "Парсинг занимает несколько секунд на строку, дождитесь завершения."
              : "";
            window.SP.busy.show();
            window.SP.busy.setBtnBusy(submitBtn, true, "Обрабатываю…");

            abortCtrl = new AbortController();
            const savedItems = [];
            try {{
              const resp = await fetch("/parse_batch_stream", {{
                method: "POST",
                body: fd,
                signal: abortCtrl.signal,
              }});
              if (!resp.ok || !resp.body) {{
                const errText = await resp.text();
                throw new Error(errText || ("HTTP " + resp.status));
              }}
              const reader = resp.body.getReader();
              const decoder = new TextDecoder();
              let buf = "";
              let total = lineCount;
              let done = 0;
              while (true) {{
                const chunk = await reader.read();
                if (chunk.done) break;
                buf += decoder.decode(chunk.value, {{ stream: true }});
                let newlineIdx;
                while ((newlineIdx = buf.indexOf("\\n")) >= 0) {{
                  const raw = buf.slice(0, newlineIdx).trim();
                  buf = buf.slice(newlineIdx + 1);
                  if (!raw) continue;
                  let evt;
                  try {{ evt = JSON.parse(raw); }} catch (e) {{ continue; }}
                  if (evt.event === "start") {{
                    total = evt.total || total;
                    resetStatus(total);
                  }} else if (evt.event === "item") {{
                    done += 1;
                    if (evt.ok) {{
                      const titleHint = (evt.metadata && (evt.metadata.title || evt.metadata.doi)) || evt.input || "";
                      markLine(evt.index, "ok", String(titleHint).slice(0, 240));
                      savedItems.push({{ citation: evt.citation, metadata: evt.metadata, input: evt.input, mode: evt.mode }});
                    }} else {{
                      markLine(evt.index, "err", evt.error || "ошибка");
                    }}
                    statusLabel.textContent = "Строка " + evt.index + " / " + total
                      + (evt.ok ? " — готово" : " — не удалось");
                    updateProgress(done, total);
                  }} else if (evt.event === "done") {{
                    setStatusState("done");
                    const ok = evt.ok_count || 0;
                    const errs = evt.error_count || 0;
                    statusLabel.textContent = errs > 0
                      ? ("Готово: " + ok + " из " + (evt.total || total) + " (есть ошибки)")
                      : ("Готово: " + ok + " из " + (evt.total || total));
                    updateProgress(evt.total || total, evt.total || total);
                    renderResults(evt, savedItems);
                  }}
                }}
              }}
            }} catch (e) {{
              if (e.name === "AbortError") {{
                setStatusState("error");
                statusLabel.textContent = "Отменено пользователем";
              }} else {{
                setStatusState("error");
                statusLabel.textContent = "Ошибка: " + (e.message || e);
                spShowToast("Не удалось обработать пакет", true);
              }}
            }} finally {{
              cancelBtn.style.display = "none";
              window.SP.busy.hide();
              window.SP.busy.setBtnBusy(submitBtn, false);
              abortCtrl = null;
            }}
          }});

          cancelBtn.addEventListener("click", function() {{
            if (abortCtrl) abortCtrl.abort();
          }});

          copyBtn.addEventListener("click", function() {{
            const value = (bibText.textContent || "").trim();
            if (!value) return;
            navigator.clipboard.writeText(value)
              .then(function() {{ spShowToast("Список скопирован"); }})
              .catch(function() {{ spShowToast("Не удалось скопировать", true); }});
          }});
          dlTxt.addEventListener("click", function() {{
            downloadBlob((bibText.textContent || ""), "bibliography.txt", "text/plain");
          }});
          dlBib.addEventListener("click", function() {{
            if (!lastResult) return;
            downloadBlob(lastResult.bibtex || "", "batch_references.bib", "application/x-bibtex");
          }});
          dlRis.addEventListener("click", function() {{
            if (!lastResult) return;
            downloadBlob(lastResult.ris || "", "batch_references.ris", "application/x-research-info-systems");
          }});
          if (dlLatex) {{
            dlLatex.addEventListener("click", function() {{
              if (!lastResult) return;
              downloadBlob(lastResult.latex || "", "batch_references.tex", "text/x-latex");
            }});
          }}
          if (dlCsl) {{
            dlCsl.addEventListener("click", function() {{
              if (!lastResult) return;
              downloadBlob(lastResult.csl || "", "batch_references_csl.json", "application/json");
            }});
          }}
          saveAllBtn.addEventListener("click", function() {{
            if (!lastResult || !Array.isArray(lastResult.savedItems) || !lastResult.savedItems.length) return;
            const items = lastResult.savedItems.map(function(it) {{ return it.metadata; }}).filter(Boolean);
            window.SP.busy.run(saveAllBtn, async function() {{
              try {{
                const resp = await fetch("/save_batch", {{
                  method: "POST",
                  headers: {{ "Content-Type": "application/json" }},
                  body: JSON.stringify({{ items: items }}),
                }});
                const data = await resp.json();
                if (data && data.ok) {{
                  const dup = data.duplicates || 0;
                  let msg = "Добавлено в итоговый список: " + (data.added || 0);
                  if (dup) msg += " (дубликатов уже было: " + dup + ")";
                  spShowToast(msg);
                }} else {{
                  spShowToast((data && data.message) || "Не удалось сохранить", true);
                }}
              }} catch (e) {{
                spShowToast("Ошибка сети при сохранении", true);
              }}
            }}, {{ busyLabel: "Сохраняю…" }});
          }});

          // --- Счётчик строк/символов и индикация лимитов -------------------
          function updateCounters() {{
            const value = inputEl.value || "";
            const lines = value.split(/\\r?\\n/).filter(function(s) {{ return s.trim(); }}).length;
            const chars = value.length;
            lineCountEl.textContent = lines + " строк";
            charCountEl.textContent = chars + " / " + MAX_CHARS + " символов";
            const exceeded = lines > MAX_LINES || chars > MAX_CHARS;
            limitWarn.hidden = !exceeded;
          }}
          inputEl.addEventListener("input", updateCounters);
          updateCounters();

          // --- Загрузка файла .txt/.csv -------------------------------------
          function ingestFile(file) {{
            if (!file) return;
            if (file.size > MAX_CHARS * 2) {{
              spShowToast("Файл слишком большой", true);
              return;
            }}
            const reader = new FileReader();
            reader.onload = function() {{
              let text = String(reader.result || "");
              // Простейший CSV-парсер: берём только первую колонку, кавычки убираем.
              if ((file.name || "").toLowerCase().endsWith(".csv")) {{
                text = text.split(/\\r?\\n/).map(function(row) {{
                  const m = row.match(/^\\s*\"?([^\",;]+)\"?/);
                  return m ? m[1].trim() : row.trim();
                }}).filter(Boolean).join("\\n");
              }}
              const existing = (inputEl.value || "").trim();
              inputEl.value = existing
                ? (existing + "\\n" + text.trim())
                : text.trim();
              updateCounters();
              spShowToast("Импортировано из файла: " + (file.name || "источник"));
            }};
            reader.onerror = function() {{ spShowToast("Не удалось прочитать файл", true); }};
            reader.readAsText(file, "utf-8");
          }}
          if (pickFileBtn) {{
            pickFileBtn.addEventListener("click", function() {{ filePicker.click(); }});
          }}
          if (filePicker) {{
            filePicker.addEventListener("change", function(ev) {{
              const f = ev.target.files && ev.target.files[0];
              if (f) ingestFile(f);
              filePicker.value = "";
            }});
          }}
          if (dropZone) {{
            ["dragenter", "dragover"].forEach(function(name) {{
              dropZone.addEventListener(name, function(ev) {{
                ev.preventDefault(); ev.stopPropagation();
                dropZone.classList.add("is-drag-over");
              }});
            }});
            ["dragleave", "drop"].forEach(function(name) {{
              dropZone.addEventListener(name, function(ev) {{
                ev.preventDefault(); ev.stopPropagation();
                dropZone.classList.remove("is-drag-over");
              }});
            }});
            dropZone.addEventListener("drop", function(ev) {{
              const dt = ev.dataTransfer;
              if (dt && dt.files && dt.files.length) {{
                ingestFile(dt.files[0]);
              }} else if (dt) {{
                const text = dt.getData("text/plain");
                if (text) {{
                  const existing = (inputEl.value || "").trim();
                  inputEl.value = existing ? (existing + "\\n" + text.trim()) : text.trim();
                  updateCounters();
                }}
              }}
            }});
          }}

          if (clearInputBtn) {{
            clearInputBtn.addEventListener("click", function() {{
              inputEl.value = "";
              updateCounters();
              status.hidden = true;
              results.hidden = true;
              if (shareBtn) shareBtn.hidden = true;
              if (checklistEl) {{ checklistEl.innerHTML = ""; checklistEl.hidden = true; }}
            }});
          }}

          if (shareBtn) {{
            shareBtn.addEventListener("click", function() {{
              const value = (inputEl.value || "").trim();
              if (!value) return;
              try {{
                const enc = btoa(unescape(encodeURIComponent(value)));
                const url = window.location.origin + window.location.pathname + "?batch=" + encodeURIComponent(enc);
                navigator.clipboard.writeText(url)
                  .then(function() {{ spShowToast("Ссылка на пакет скопирована"); }})
                  .catch(function() {{ spShowToast("Не удалось скопировать ссылку", true); }});
              }} catch (e) {{
                spShowToast("Не удалось закодировать ссылку", true);
              }}
            }});
          }}

          // --- Deep link: автозаполнение textarea из ?batch=base64 ----------
          (function() {{
            try {{
              const q = new URLSearchParams(window.location.search);
              const enc = q.get("batch");
              if (!enc) return;
              const raw = decodeURIComponent(escape(atob(enc)));
              if (!raw) return;
              inputEl.value = raw;
              updateCounters();
              const tabBtn = document.querySelector('[data-tab-target="home"]');
              if (tabBtn) tabBtn.click();
              if (shareBtn) shareBtn.hidden = false;
              spShowToast("Загружен пакет из ссылки");
            }} catch (e) {{ /* ignore broken links */ }}
          }})();
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
        return jsonify({"ok": True, "added": 0, "duplicates": 1, "message": "Уже добавлено"})
    return jsonify({"ok": True, "added": 1, "duplicates": 0, "message": "Добавлено в итоговый список"})


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


@app.get("/export_csl")
@limiter.limit("20/minute")
def export_csl_endpoint():
    payload = export_csl_json(_db_load_selected(_get_session_id()))
    return Response(
        payload.encode("utf-8"),
        mimetype="application/json; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="selected_references_csl.json"'},
    )


@app.get("/export_latex")
@limiter.limit("20/minute")
def export_latex_endpoint():
    """
    Готовый фрагмент LaTeX `\\begin{thebibliography}` с одним
    `\\bibitem{...}` на запись — можно сразу вставить в .tex без .bib.
    """
    style = _citation_style(request.args.get("style") or "gost")
    payload = export_latex_thebibliography(_db_load_selected(_get_session_id()), style=style)
    return Response(
        payload.encode("utf-8"),
        mimetype="text/plain; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="selected_references.tex"'},
    )


@app.get("/api/url_check")
@limiter.limit("60/minute")
def api_url_check():
    """
    Проверка URL до загрузки HTML: объяснение SSRF и опционально цепочка редиректов (HEAD).
    Query: url=..., probe=1|0
    """
    url = (request.args.get("url") or "").strip()
    verr = _validate_input_length(url, MAX_URL_LEN, "url")
    if verr:
        return jsonify({"ok": False, "error": verr}), 400
    expl = explain_public_http_url(url)
    want_probe = (request.args.get("probe") or "").strip().lower() in ("1", "true", "yes", "on")
    probe_payload: Dict[str, object] = {}
    if want_probe:
        probe_payload = probe_http_redirect_chain(url) if expl.get("ok") else {"skipped": True, "reason": expl.get("reason")}
    return jsonify({"ok": True, "allowed": bool(expl.get("ok")), "explain": expl, "redirect_probe": probe_payload})


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
        expl = explain_public_http_url(url)
        if not expl.get("ok"):
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "URL не разрешён политикой безопасности",
                        "url_security": {"explain": expl},
                    }
                ),
                400,
            )
        meta = extract_metadata_from_url(url)
        return jsonify(
            {
                "ok": True,
                "mode": "url",
                "metadata": asdict(meta),
                "url_security": {"explain": expl},
            }
        )

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


def _aggregate_history_stats(session_id: str) -> Dict[str, object]:
    """
    Лёгкая телеметрия по сессии: сколько запросов в разрезе режима,
    доля успешных, среднее и медианное время. Используется на странице
    «История» как мини-дашборд для самого пользователя/защиты.
    """
    db = _get_db()
    overall = db.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN ok = 1 THEN 1 ELSE 0 END) AS ok_count,
            AVG(duration_ms) AS avg_ms,
            MIN(duration_ms) AS min_ms,
            MAX(duration_ms) AS max_ms
        FROM search_history
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    rows = db.execute(
        """
        SELECT mode,
               COUNT(*) AS n,
               SUM(CASE WHEN ok = 1 THEN 1 ELSE 0 END) AS ok_n,
               AVG(duration_ms) AS avg_ms
        FROM search_history
        WHERE session_id = ?
        GROUP BY mode
        ORDER BY n DESC
        """,
        (session_id,),
    ).fetchall()
    durations = [
        int(r["duration_ms"]) for r in db.execute(
            "SELECT duration_ms FROM search_history WHERE session_id = ? AND duration_ms IS NOT NULL",
            (session_id,),
        ).fetchall()
        if r["duration_ms"] is not None
    ]
    median_ms = None
    if durations:
        durations.sort()
        n = len(durations)
        median_ms = durations[n // 2] if n % 2 else (durations[n // 2 - 1] + durations[n // 2]) // 2
    by_mode = []
    for r in rows:
        n = int(r["n"] or 0)
        ok_n = int(r["ok_n"] or 0)
        by_mode.append({
            "mode": r["mode"] or "—",
            "n": n,
            "ok_n": ok_n,
            "err_rate": 0.0 if n == 0 else round(1 - ok_n / n, 3),
            "avg_ms": int(r["avg_ms"]) if r["avg_ms"] is not None else None,
        })
    return {
        "total": int(overall["total"] or 0),
        "ok_count": int(overall["ok_count"] or 0),
        "avg_ms": int(overall["avg_ms"]) if overall["avg_ms"] is not None else None,
        "min_ms": int(overall["min_ms"]) if overall["min_ms"] is not None else None,
        "max_ms": int(overall["max_ms"]) if overall["max_ms"] is not None else None,
        "median_ms": median_ms,
        "by_mode": by_mode,
    }


@app.get("/history")
@limiter.limit("60/minute")
def search_history_page():
    if _anonymous_mode():
        return (
            "<p>История поиска отключена (SP_ANONYMOUS=1).</p><p><a href='/'>На главную</a></p>",
            403,
        )
    sid = _get_session_id()
    tab = (request.args.get("tab") or "all").strip().lower()
    reject_only = tab == "rejected"
    where_extra = " AND mode LIKE 'reject:%' " if reject_only else " AND COALESCE(mode,'') NOT LIKE 'reject:%' "
    rows = _get_db().execute(
        f"""
        SELECT created_at, mode, url, doi, title_query, result_title, ok, error_message, duration_ms
        FROM search_history
        WHERE session_id = ? {where_extra}
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
        dur_label = ""
        try:
            ms = r["duration_ms"]
            if ms is not None:
                dur_label = f' · <span class="hint">{int(ms)} мс</span>'
        except Exception:
            dur_label = ""
        cards.append(
            f"""<div class="card" style="margin-bottom:10px;padding:12px;">
            <div style="color:var(--muted);font-size:13px;">{escape(r["created_at"] or "")}{dur_label}</div>
            <div><b>{escape(r["mode"] or "")}</b></div>
            <div style="margin-top:6px;">Запрос: {escape(snippet) if snippet else "—"}</div>
            <div style="margin-top:6px;">Причина / итог: {outcome_html}</div>
            </div>"""
        )
    body = "".join(cards) if cards else (
        "<p class='empty-text'>Нет записей в этой вкладке.</p>"
        if reject_only
        else "<p class='empty-text'>Пока нет сохранённых запросов в этой сессии.</p>"
    )

    stats = _aggregate_history_stats(sid)
    err_rate = 0.0 if stats["total"] == 0 else round(1 - stats["ok_count"] / stats["total"], 3)
    avg_label = f"{stats['avg_ms']} мс" if stats["avg_ms"] is not None else "—"
    median_label = f"{stats['median_ms']} мс" if stats["median_ms"] is not None else "—"
    overall_html = f"""
        <div class="meta-row" style="grid-template-columns:repeat(4, minmax(0, 1fr));gap:8px;">
          <div class="meta-item"><div class="k">Запросов</div><div class="v">{stats['total']}</div></div>
          <div class="meta-item"><div class="k">Доля ошибок</div><div class="v">{int(err_rate * 100)}%</div></div>
          <div class="meta-item"><div class="k">Среднее</div><div class="v">{avg_label}</div></div>
          <div class="meta-item"><div class="k">Медиана</div><div class="v">{median_label}</div></div>
        </div>
    """
    mode_rows = "".join(
        f"<tr><td>{escape(b['mode'])}</td><td>{b['n']}</td><td>{int(b['err_rate'] * 100)}%</td>"
        f"<td>{b['avg_ms'] if b['avg_ms'] is not None else '—'}{' мс' if b['avg_ms'] is not None else ''}</td></tr>"
        for b in stats["by_mode"]
    ) or "<tr><td colspan='4' class='empty-text'>Нет данных</td></tr>"
    by_mode_html = f"""
        <div class="card mt-12">
          <h3 class="section-title">По режимам</h3>
          <table style="width:100%;border-collapse:collapse;font-size:13px;">
            <thead><tr style="text-align:left;color:var(--muted);">
              <th style="padding:6px;">Режим</th>
              <th style="padding:6px;">Запросов</th>
              <th style="padding:6px;">Ошибок</th>
              <th style="padding:6px;">Среднее</th>
            </tr></thead>
            <tbody>{mode_rows}</tbody>
          </table>
        </div>
    """

    tab_all_cls = "" if reject_only else " btn-primary"
    tab_rej_cls = " btn-primary" if reject_only else ""
    hero_heading = "Отклонённые URL" if reject_only else "Последние запросы и сводка по сессии"
    hero_sub = (
        "Записи режимов <code>reject:*</code>: SSRF, HTTP 403 и 5xx без восстановления метаданных, сетевые сбои."
        if reject_only
        else "Вкладка «Запросы» скрывает служебные отказы; они на «Отклонённые URL». До 40 записей; сводка — по всей сессии."
    )
    list_heading = "Отклонённые URL (журнал)" if reject_only else "Журнал запросов"

    return f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>История поиска</title>
      <style>{APP_THEME_CSS}</style>
    </head>
    <body>
      <div id="globalBusyBar" class="busy-bar"></div>
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
        <div class="hero">
          <h2>{hero_heading}</h2>
          <p>{hero_sub} Отключение истории: <code>SP_ANONYMOUS=1</code>.</p>
          <div class="action-row" style="margin-top:12px;">
            <a class="btn btn-soft{tab_all_cls}" href="/history?tab=all">Запросы</a>
            <a class="btn btn-soft{tab_rej_cls}" href="/history?tab=rejected">Отклонённые URL</a>
          </div>
        </div>
        <div class="card">
          <h3 class="section-title">Сводка</h3>
          {overall_html}
        </div>
        {by_mode_html}
        <div class="card mt-12">
          <h3 class="section-title">{list_heading}</h3>
          {body}
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


def _batch_parse_lines(raw: str):
    """
    Подготовка пакетного разбора: проверка лимитов и нарезка на строки.
    Возвращает (citation_style, lines, error_message). При ошибке lines пуст.
    """
    if len(raw) > MAX_BATCH_CHARS:
        return None, [], f"Слишком большой текст пакета (max {MAX_BATCH_CHARS} символов)."
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()][:MAX_BATCH_LINES]
    if not lines:
        return None, [], "Введите хотя бы одну непустую строку (DOI, URL или название)."
    return None, lines, None


def _batch_process_line(line: str):
    """
    Обработать одну строку пакета. Возвращает (PaperMetadata|None, mode_str, error_str|None).
    """
    try:
        if re.match(r"^10\.\d+", line):
            return extract_metadata_from_doi(line), "doi", None
        if line.lower().startswith(("http://", "https://")):
            if not is_public_http_url(line):
                return None, "url", "URL запрещён политикой безопасности"
            return extract_metadata_from_url(line), "url", None
        return extract_metadata_from_title(line), "title", None
    except Exception as e:
        return None, "unknown", str(e)


@app.post("/parse_batch")
@limiter.limit("8/hour")
def parse_batch():
    """
    Резервный неинтерактивный пакетный разбор (без JS): сразу возвращает
    готовый библиографический список + блоки BibTeX/RIS для копирования.
    Основной поток в браузере использует /parse_batch_stream и интерактивный
    блок результатов в индексной странице.
    """
    raw = (request.form.get("batch") or "").strip()
    citation_style = _citation_style(request.form.get("citation_style") or "gost")
    _, lines, err = _batch_parse_lines(raw)
    if err:
        return err, 400

    items: List[PaperMetadata] = []
    errors: List[Tuple[int, str, str]] = []
    sid = _get_session_id()
    for i, line in enumerate(lines, start=1):
        meta, mode_line, error = _batch_process_line(line)
        if meta is not None and (meta.title or meta.doi or meta.authors):
            items.append(meta)
            _db_log_search(sid, f"batch:{mode_line}", "", "", line, (meta.title or "")[:500])
        else:
            err_msg = error or "Не удалось распознать запись"
            errors.append((i, line, err_msg))
            batch_mode = "reject:ssrf" if mode_line == "url" and "безопасности" in err_msg else "batch"
            url_col = line if mode_line == "url" else ""
            _db_log_search(
                sid,
                batch_mode,
                url_col,
                "",
                "" if url_col else line,
                "",
                ok=False,
                error_message=err_msg[:500],
            )

    combined = format_bibliography_list(items, style=citation_style)
    bibtex = export_bibtex(items)
    ris = export_ris(items)
    latex_block = export_latex_thebibliography(items, style=citation_style)

    checklist_html = _render_checklist_block(items)

    error_html = ""
    if errors:
        rows = "".join(
            f"<li>Строка {i}: <code>{escape(src)}</code> — {escape(msg)}</li>"
            for i, src, msg in errors
        )
        error_html = (
            f'<div class="card" style="border-left:4px solid var(--warning);">'
            f"<h3 class=\"section-title\">Строки, которые не удалось обработать</h3>"
            f"<ul>{rows}</ul></div>"
        )

    return f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Пакетный разбор</title>
      <style>{APP_THEME_CSS}</style>
    </head>
    <body>
      <div id="globalBusyBar" class="busy-bar"></div>
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
        <div class="hero"><h2>Готовый библиографический список</h2>
        <p>Стиль: {escape(citation_style.upper())}. Обработано: {len(items)} из {len(lines)}.</p></div>

        <div class="card">
          <h3 class="section-title">Библиографический список</h3>
          <pre class="mono-box" id="bibliography_data">{escape(combined)}</pre>
          <div class="mt-10 action-row">
            <button class="btn btn-soft" id="copyListBtn">Скопировать список</button>
            <a class="btn btn-soft" href="data:text/plain;charset=utf-8,{quote_plus(combined)}" download="bibliography.txt">Скачать TXT</a>
          </div>
        </div>

        <div class="card">
          <h3 class="section-title">Экспорт</h3>
          <div class="action-row">
            <a class="btn btn-soft" href="data:text/plain;charset=utf-8,{quote_plus(bibtex)}" download="batch_references.bib">Скачать BibTeX</a>
            <a class="btn btn-soft" href="data:text/plain;charset=utf-8,{quote_plus(ris)}" download="batch_references.ris">Скачать RIS</a>
            <a class="btn btn-soft" href="data:text/plain;charset=utf-8,{quote_plus(latex_block)}" download="batch_references.tex">Скачать LaTeX</a>
          </div>
          <details class="variant-collapse mt-12">
            <summary><span>BibTeX (предпросмотр)</span></summary>
            <div class="variant-body"><pre class="mono-box">{escape(bibtex)}</pre></div>
          </details>
          <details class="variant-collapse mt-12">
            <summary><span>RIS (предпросмотр)</span></summary>
            <div class="variant-body"><pre class="mono-box">{escape(ris)}</pre></div>
          </details>
          <details class="variant-collapse mt-12">
            <summary><span>LaTeX <code>thebibliography</code> (предпросмотр)</span></summary>
            <div class="variant-body"><pre class="mono-box">{escape(latex_block)}</pre></div>
          </details>
        </div>

        {checklist_html}
        {error_html}
      </div>
      <div id="toast" class="toast"></div>
      <script>{APP_BUSY_JS}</script>
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
          const copyBtn = document.getElementById("copyListBtn");
          if (copyBtn) {{
            copyBtn.addEventListener("click", function() {{
              const node = document.getElementById("bibliography_data");
              const value = (node ? node.textContent : "").trim();
              navigator.clipboard.writeText(value).then(function() {{
                const t = document.getElementById("toast");
                t.textContent = "Список скопирован";
                t.style.display = "block";
                setTimeout(function() {{ t.style.display = "none"; }}, 2000);
              }});
            }});
          }}
        }})();
      </script>
    </body>
    </html>
    """


def _batch_ndjson_stream_factory(raw: str, citation_style: str):
    _, lines, err = _batch_parse_lines(raw)
    if err:
        return None, err

    def _emit(payload: dict) -> str:
        return json.dumps(payload, ensure_ascii=False) + "\n"

    def stream():
        sid = _get_session_id()
        items: List[PaperMetadata] = []
        errors: List[dict] = []
        total = len(lines)
        yield _emit({"event": "start", "total": total, "style": citation_style})
        for i, line in enumerate(lines, start=1):
            t0 = time.monotonic()
            meta, mode_line, error = _batch_process_line(line)
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            if meta is not None and (meta.title or meta.doi or meta.authors):
                cit = format_citation(meta, style=citation_style)
                items.append(meta)
                _db_log_search(
                    sid,
                    f"batch:{mode_line}",
                    "",
                    "",
                    line,
                    (meta.title or "")[:500],
                    duration_ms=elapsed_ms,
                )
                yield _emit({
                    "event": "item",
                    "index": i,
                    "total": total,
                    "ok": True,
                    "mode": mode_line,
                    "input": line,
                    "citation": cit,
                    "metadata": asdict(meta),
                    "completeness": paper_completeness(meta),
                    "provenance": paper_provenance(meta),
                    "duration_ms": elapsed_ms,
                })
            else:
                err_msg = error or "Не удалось распознать запись"
                errors.append({"index": i, "input": line, "error": err_msg})
                batch_mode = "reject:ssrf" if mode_line == "url" and "безопасности" in err_msg else "batch"
                url_col = line if mode_line == "url" else ""
                _db_log_search(
                    sid,
                    batch_mode,
                    url_col,
                    "",
                    "" if url_col else line,
                    "",
                    ok=False,
                    error_message=err_msg[:500],
                    duration_ms=elapsed_ms,
                )
                yield _emit({
                    "event": "item",
                    "index": i,
                    "total": total,
                    "ok": False,
                    "mode": mode_line,
                    "input": line,
                    "error": err_msg,
                    "duration_ms": elapsed_ms,
                })

        combined = format_bibliography_list(items, style=citation_style)
        bibtex = export_bibtex(items)
        ris = export_ris(items)
        latex_block = export_latex_thebibliography(items, style=citation_style)
        csl_block = export_csl_json(items)
        yield _emit({
            "event": "done",
            "total": total,
            "ok_count": len(items),
            "error_count": len(errors),
            "style": citation_style,
            "bibliography": combined,
            "bibtex": bibtex,
            "ris": ris,
            "latex": latex_block,
            "csl": csl_block,
            "errors": errors,
        })

    return stream, None


@app.post("/parse_batch_stream")
@limiter.limit("12/hour")
def parse_batch_stream():
    """
    Потоковая обработка пакета: NDJSON-события `start` / `item` / `done`.
    Каждый ответ — одна строка JSON, фронтенд читает их по мере поступления
    и обновляет статус-бар, не дожидаясь завершения всего пакета.
    """
    raw = (request.form.get("batch") or "").strip()
    citation_style = _citation_style(request.form.get("citation_style") or "gost")
    stream_fn, err = _batch_ndjson_stream_factory(raw, citation_style)
    if err:
        return err, 400

    return Response(
        stream_with_context(stream_fn()),
        mimetype="application/x-ndjson; charset=utf-8",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache, no-transform",
        },
    )


@app.post("/api/batch")
@limiter.limit("12/hour")
def api_batch_ndjson_stream():
    """
    Тот же NDJSON-поток, что и /parse_batch_stream, для расширений и скриптов:
    тело запроса JSON `{"batch": "...", "citation_style": "gost"}` или form-data.
    """
    raw = ""
    citation_style = _citation_style("gost")
    body = request.get_json(silent=True)
    if isinstance(body, dict):
        raw = (body.get("batch") or "").strip()
        citation_style = _citation_style(body.get("citation_style") or citation_style)
    if not raw:
        raw = (request.form.get("batch") or "").strip()
        citation_style = _citation_style(request.form.get("citation_style") or citation_style)
    stream_fn, err = _batch_ndjson_stream_factory(raw, citation_style)
    if err:
        return jsonify({"ok": False, "error": err}), 400
    return Response(
        stream_with_context(stream_fn()),
        mimetype="application/x-ndjson; charset=utf-8",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache, no-transform",
        },
    )


@app.post("/save_batch")
@limiter.limit("20/hour")
def save_batch():
    """
    Групповое сохранение пакетного результата в общий итоговый список сессии.
    Принимает JSON {"items": [PaperMetadata, ...]}; возвращает количество добавленных.
    """
    if len(request.get_data(cache=True) or b"") > MAX_PAYLOAD_LEN * 4:
        return jsonify({"ok": False, "message": "Слишком большой payload"}), 400
    try:
        payload = request.get_json(silent=True) or {}
        items = payload.get("items") or []
    except Exception:
        return jsonify({"ok": False, "message": "Некорректные данные"}), 400
    if not isinstance(items, list) or not items:
        return jsonify({"ok": False, "message": "Список пуст"}), 400
    if len(items) > MAX_BATCH_LINES * 2:
        return jsonify({"ok": False, "message": "Слишком много элементов"}), 400

    sid = _get_session_id()
    added = 0
    duplicates = 0
    invalid = 0
    for entry in items:
        if not isinstance(entry, dict):
            invalid += 1
            continue
        try:
            paper = PaperMetadata(**entry)
        except Exception:
            invalid += 1
            continue
        if _db_save_selected(sid, paper):
            added += 1
        else:
            duplicates += 1
    return jsonify(
        {
            "ok": True,
            "added": added,
            "duplicates": duplicates,
            "invalid": invalid,
            "total": len(items),
        }
    )


@app.post("/parse_citation")
@limiter.limit("20/minute")
def parse_citation_endpoint():
    """
    Разбор свободной библиографической строки (JSON или form).
    Возвращает метаданные, служебную информацию о сопоставлении и цитату в выбранном стиле.
    """
    line = ""
    citation_style = "gost"
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        line = (payload.get("citation") or payload.get("line") or "").strip()
        citation_style = _citation_style(str(payload.get("citation_style") or "gost"))
    else:
        line = (request.form.get("citation") or request.form.get("line") or "").strip()
        citation_style = _citation_style(request.form.get("citation_style") or "gost")

    if len(line) > MAX_CITATION_LINE:
        return jsonify({"ok": False, "error": f"Слишком длинная строка (макс. {MAX_CITATION_LINE} символов)"}), 400
    if not line:
        return jsonify({"ok": False, "error": "Введите строку цитирования"}), 400

    try:
        meta, info = parse_citation_string(line)
    except Exception as e:
        LOGGER.exception("parse_citation_endpoint")
        return jsonify({"ok": False, "error": str(e)}), 500

    cands = info.get("candidates") or []
    parse_info = {
        "input": info.get("input"),
        "parsed": info.get("parsed"),
        "matched_via": info.get("matched_via"),
        "candidates": cands,
    }
    return jsonify(
        {
            "ok": True,
            "metadata": asdict(meta),
            "parse_info": parse_info,
            "citation": format_citation(meta, style=citation_style),
            "journal_indices": journal_indices_for(meta),
            "candidates_count": len(cands),
        }
    )


@app.post("/validate_draft")
@limiter.limit("15/minute")
def validate_draft_endpoint():
    """
    Проверка черновика списка литературы: построчно parse_citation_string + paper_completeness.
    """
    text = ""
    max_lines = 80
    citation_style = "gost"
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        text = (payload.get("text") or payload.get("draft") or "").strip()
        try:
            max_lines = max(1, min(MAX_DRAFT_VALIDATE_LINES, int(payload.get("max_lines") or 80)))
        except (TypeError, ValueError):
            max_lines = 80
        citation_style = _citation_style(str(payload.get("citation_style") or "gost"))
    else:
        text = (request.form.get("text") or request.form.get("draft") or "").strip()
        raw_ml = (request.form.get("max_lines") or "").strip()
        if raw_ml:
            try:
                max_lines = max(1, min(MAX_DRAFT_VALIDATE_LINES, int(raw_ml)))
            except ValueError:
                max_lines = 80
        citation_style = _citation_style(request.form.get("citation_style") or "gost")

    if len(request.get_data(cache=True) or b"") > MAX_PAYLOAD_LEN:
        return jsonify({"ok": False, "error": "Слишком большой запрос"}), 400
    if len(text) > MAX_PAYLOAD_LEN:
        return jsonify({"ok": False, "error": f"Слишком длинный текст (макс. {MAX_PAYLOAD_LEN} символов)"}), 400
    if not text:
        return jsonify({"ok": False, "error": "Введите текст списка литературы"}), 400

    try:
        items = validate_draft_text(text, max_lines=max_lines, citation_style=citation_style)
    except Exception as e:
        LOGGER.exception("validate_draft_endpoint")
        return jsonify({"ok": False, "error": str(e)}), 500

    summary = {"ok": 0, "partial": 0, "not_found": 0}
    for it in items:
        st = str(it.get("status") or "")
        if st in summary:
            summary[st] += 1
    return jsonify(
        {
            "ok": True,
            "items": items,
            "summary": summary,
            "total": len(items),
            "max_lines": max_lines,
            "citation_style": citation_style,
        }
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
    citation_style = _citation_style(citation_style)

    sid = _get_session_id()
    err = _validate_input_length(url, MAX_URL_LEN, "url") or _validate_input_length(title_input, MAX_TITLE_LEN, "title") or _validate_input_length(doi, MAX_DOI_LEN, "doi")
    if err:
        return err, 400
    if url and not is_public_http_url(url):
        expl = explain_public_http_url(url)
        reason = expl.get("reason") or "blocked"
        _db_log_search(
            sid,
            "reject:ssrf",
            url,
            "",
            "",
            "",
            ok=False,
            error_message=str(reason)[:500],
        )
        return "URL не разрешен политикой безопасности.", 400

    if not url and not title_input and not doi:
        return "Введите URL или название или DOI", 400

    request_started_at = time.monotonic()
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
    url_security_block = _render_url_security_card(url) if url else ""
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
            <a class="btn btn-soft" href="/export_latex?style={escape(citation_style)}">Экспорт LaTeX</a>
            <a class="btn btn-soft" href="/export_csl">Экспорт CSL JSON</a>
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
        prov = paper_provenance(item)
        prov_html = ""
        if prov:
            chips = "".join(
                f'<span class="pill" title="Источник поля {escape(field)}">{escape(field)}: {escape(src)}</span>'
                for field, src in prov.items()
            )
            prov_html = f'<div class="row"><b>Источники полей:</b></div><div class="batch-line-status">{chips}</div>'
        report = paper_completeness(item)
        if report["ok"] and not report["warnings"]:
            comp_chip = '<span class="status-chip status-ok">✓ полный</span>'
        elif not report["missing"]:
            comp_chip = (
                f'<span class="status-chip status-warn">⚠ {len(report["warnings"])} замечаний</span>'
            )
        else:
            comp_chip = (
                f'<span class="status-chip status-bad">⛔ нет {len(report["missing"])} обязательных</span>'
            )
        completeness_html = (
            f'<div class="row"><b>Полнота:</b> {comp_chip} '
            f'<span class="hint">({report["filled"]}/{report["total"]} полей)</span></div>'
        )
        journal_idx_html = _journal_index_chips_html(item)
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
            {completeness_html}
            {journal_idx_html}
            {prov_html}
            <div class="row"><b>Почему этот вариант в списке:</b> {escape(rank_expl)}</div>
            <div class="row"><b>Комментарий:</b> {escape(full["reason"])}</div>
            <div class="row"><b>Ссылки:</b><br/>{links}</div>
            <div class="row action-row">{pdf_btn} {save_btn}</div>
            <div class="hint">Кнопка добавляет этот вариант в общий список для экспорта BibTeX / RIS / CSL JSON.</div>
          </div>
        </details>
        """

    _maybe_log_url_fetch_rejection(sid, url, meta)

    _db_log_search(
        sid,
        mode_used,
        url,
        doi,
        title_input,
        (meta.title or "")[:500],
        duration_ms=int((time.monotonic() - request_started_at) * 1000),
    )

    checklist_block = _render_checklist_block(candidates[:1] if candidates else [])

    # Снимок для офлайн-кэша в localStorage: только то, что нужно для повтора
    # запроса и быстрой вставки готовой цитаты, без больших объектов.
    last_result_payload = json.dumps(
        {
            "savedAt": int(time.time() * 1000),
            "style": citation_style,
            "mode": mode_used,
            "doi": doi or None,
            "url": url or None,
            "title": title_input or None,
            "citation": format_citation(meta, style=citation_style) if meta else None,
            "result_title": (meta.title or "") if meta else "",
        },
        ensure_ascii=False,
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
      <div id="globalBusyBar" class="busy-bar"></div>
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
              <a class="btn btn-soft" href="/export_latex?style={escape(citation_style)}">LaTeX</a>
              <a class="btn btn-soft" href="/export_csl">CSL JSON</a>
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
        {url_security_block}
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
            <button type="button" class="btn btn-soft" data-action="copy-bibliography">Скопировать список</button>
          </div>
        </div>

        {checklist_block}

        {selected_block}

        <div class="card">
          <h3 class="section-title">Найденные варианты</h3>
          {variants_html or "<p class='empty-text'>Нет вариантов.</p>"}
        </div>
      </div>

      <div id="toast" class="toast"></div>

      <script>{APP_BUSY_JS}</script>
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
        function copyBibliography(triggerBtn) {{
          const node = document.getElementById("bibliography_data");
          const value = (node ? node.textContent : "").trim();
          window.SP.busy.run(triggerBtn || null, async function() {{
            try {{
              await navigator.clipboard.writeText(value);
              showToast("Библиографический список скопирован");
            }} catch (e) {{
              showToast("Не удалось скопировать список", true);
            }}
          }}, {{ busyLabel: "Копирую…" }});
        }}
        function downloadPdf(triggerBtn, pdfUrl, doi, sourceUrl, title) {{
          const params = new URLSearchParams({{
            pdf_url: pdfUrl || "",
            doi: doi || "",
            source_url: sourceUrl || "",
            title: title || "article",
          }});
          // Открываем вкладку синхронно по клику, чтобы браузер не блокировал загрузку.
          const downloadWindow = window.open("about:blank", "_blank");
          window.SP.busy.run(triggerBtn, async function() {{
            try {{
              const r = await fetch("/probe_pdf?" + params.toString());
              const data = await r.json();
              if (!data.ok) {{
                if (downloadWindow) downloadWindow.close();
                throw new Error(data.message || "Не удалось получить PDF");
              }}
              if (downloadWindow) {{
                downloadWindow.location.href = "/download_pdf?" + params.toString();
              }} else {{
                window.location.href = "/download_pdf?" + params.toString();
              }}
              showToast("Начата загрузка PDF");
            }} catch (e) {{
              showToast(e.message || "Не удалось скачать PDF", true);
            }}
          }}, {{ busyLabel: "Готовлю PDF…" }});
        }}
        function saveCandidate(triggerBtn, encodedPayload) {{
          window.SP.busy.run(triggerBtn, async function() {{
            try {{
              const r = await fetch("/save_candidate?data=" + encodedPayload);
              const data = await r.json();
              showToast(data.message || "Готово");
              if (data.ok) window.location.reload();
            }} catch (e) {{
              showToast("Не удалось сохранить вариант", true);
            }}
          }}, {{ busyLabel: "Сохраняю…" }});
        }}
        function bindSelectedListButtons() {{
          const popBtn = document.getElementById("popLastSelectedBtn");
          const clearBtn = document.getElementById("clearAllSelectedBtn");
          if (popBtn) {{
            popBtn.addEventListener("click", () => {{
              window.SP.busy.run(popBtn, async function() {{
                try {{
                  const r = await fetch("/pop_last_selected");
                  const data = await r.json();
                  showToast(data.message || "Готово", !data.ok);
                  if (data.ok) window.location.reload();
                }} catch (e) {{
                  showToast("Не удалось обновить список", true);
                }}
              }}, {{ busyLabel: "Удаляю…" }});
            }});
          }}
          if (clearBtn) {{
            clearBtn.addEventListener("click", () => {{
              if (!confirm("Удалить все статьи из итогового списка?")) return;
              window.SP.busy.run(clearBtn, async function() {{
                try {{
                  const r = await fetch("/clear_selected");
                  const data = await r.json();
                  showToast(data.message || "Готово", !data.ok);
                  if (data.ok) window.location.reload();
                }} catch (e) {{
                  showToast("Не удалось очистить список", true);
                }}
              }}, {{ busyLabel: "Очищаю…" }});
            }});
          }}
        }}
        function bindActionButtons() {{
          document.querySelectorAll(".download-pdf-btn").forEach((btn) => {{
            btn.addEventListener("click", () => {{
              downloadPdf(
                btn,
                btn.dataset.pdfUrl || "",
                btn.dataset.doi || "",
                btn.dataset.sourceUrl || "",
                btn.dataset.title || "article"
              );
            }});
          }});
          document.querySelectorAll(".save-candidate-btn").forEach((btn) => {{
            btn.addEventListener("click", () => {{
              saveCandidate(btn, btn.dataset.payload || "");
            }});
          }});
          document.querySelectorAll("[data-action='copy-bibliography']").forEach((btn) => {{
            btn.addEventListener("click", () => copyBibliography(btn));
          }});
        }}
        bindActionButtons();
        bindSelectedListButtons();
        (function() {{
          // Сохраняем «снимок» одиночного результата в localStorage,
          // чтобы на главной странице была карточка «Последний результат» —
          // удобно при потере сети или для быстрой вставки последней цитаты.
          try {{
            const payload = {last_result_payload};
            if (payload && payload.citation) {{
              localStorage.setItem("sp_last_result_v1", JSON.stringify(payload));
            }}
          }} catch (e) {{ /* ignore quota errors */ }}
        }})();
      </script>
    </body>
    </html>
    """


_init_db()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

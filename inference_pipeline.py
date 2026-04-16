# inference_pipeline.py

import json
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urljoin
from difflib import SequenceMatcher
from functools import lru_cache
import logging
import time
import joblib
import requests
from bs4 import BeautifulSoup
import pandas as pd
from net_security import is_public_http_url
try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None

MODEL_PATH = "models/block_classifier.joblib"
LOGGER = logging.getLogger(__name__)
API_TIMEOUTS = (4, 8)
SEARCH_TIME_BUDGET_SEC = 9.0

DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
NON_WORD_RE = re.compile(r"[^\w\s\-]", re.UNICODE)

EN_TO_RU_LAYOUT = str.maketrans(
    "`qwertyuiop[]asdfghjkl;'zxcvbnm,./"
    '~QWERTYUIOP{}ASDFGHJKL:"ZXCVBNM<>?',
    "ёйцукенгшщзхъфывапролджэячсмитьбю."
    "ЁЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ,",
)
RU_TO_EN_LAYOUT = str.maketrans(
    "ёйцукенгшщзхъфывапролджэячсмитьбю."
    "ЁЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ,",
    "`qwertyuiop[]asdfghjkl;'zxcvbnm,./"
    '~QWERTYUIOP{}ASDFGHJKL:"ZXCVBNM<>?',
)


@dataclass
class PaperMetadata:
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    year: Optional[str] = None
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    pdf_url: Optional[str] = None
    source_url: Optional[str] = None
    enriched_by: Optional[str] = None  # e.g. "crossref"
    confidence: Optional[Dict[str, float]] = None  # per-field confidence
    search_score: Optional[float] = None
    matched_query: Optional[str] = None


def fetch_html(url: str, timeout: int = 20) -> str:
    if not is_public_http_url(url):
        raise ValueError("Blocked non-public URL")
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
            "Referer": "https://www.google.com/",
        }
    )
    r = s.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def extract_blocks_from_html(html: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")

    blocks: List[Dict[str, Any]] = []
    elements = soup.find_all(["h1", "h2", "h3", "p", "div", "span", "li", "td"])

    def dom_depth(elem) -> int:
        depth = 0
        p = elem.parent
        while p is not None and getattr(p, "name", None):
            depth += 1
            p = p.parent
        return depth

    def sibling_index(elem) -> int:
        parent = elem.parent
        if parent is None:
            return 0
        siblings = parent.find_all(elem.name, recursive=False)
        for idx, sibling in enumerate(siblings):
            if sibling is elem:
                return idx
        return 0

    def link_density(elem, text_len: int) -> float:
        if text_len <= 0:
            return 0.0
        link_text = " ".join(a.get_text(" ", strip=True) for a in elem.find_all("a"))
        return min(1.0, len(link_text) / max(text_len, 1))

    for elem in elements:
        text = elem.get_text().strip()
        if not text or len(text) < 5:
            continue

        style = (elem.get("style") or "").lower()
        is_visible = not ("display:none" in style or "visibility:hidden" in style)

        blocks.append(
            {
                "text": text,
                "tag": elem.name,
                "classes": elem.get("class", []),
                "text_length": len(text),
                "word_count": len(text.split()),
                "is_visible": int(is_visible),
                "dom_depth": dom_depth(elem),
                "sibling_index": sibling_index(elem),
                "link_density": link_density(elem, len(text)),
            }
        )

    return blocks


@lru_cache(maxsize=2048)
def _crossref_work_message_cached(doi: str) -> Optional[dict]:
    url = f"https://api.crossref.org/works/{doi}"
    headers = {"User-Agent": "scientific-parser/1.0 (mailto: example@example.com)"}
    for timeout in API_TIMEOUTS:
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            if r.status_code != 200:
                continue
            return (r.json() or {}).get("message") or {}
        except requests.RequestException as e:
            LOGGER.warning("CrossRef works timeout/failure for DOI %s: %s", doi, e)
            time.sleep(0.15)
    return None


@lru_cache(maxsize=2048)
def _crossref_search_items_cached(query: str, rows: int) -> Tuple[dict, ...]:
    params = {"query.bibliographic": query, "rows": rows}
    headers = {"User-Agent": "scientific-parser/1.0 (mailto: example@example.com)"}
    for timeout in API_TIMEOUTS:
        try:
            r = requests.get(
                "https://api.crossref.org/works",
                params=params,
                timeout=timeout,
                headers=headers,
            )
            if r.status_code != 200:
                continue
            items = ((r.json() or {}).get("message") or {}).get("items") or []
            if not isinstance(items, list):
                return tuple()
            return tuple(items)
        except requests.RequestException as e:
            LOGGER.warning("CrossRef search timeout/failure for query '%s': %s", query, e)
            time.sleep(0.15)
    return tuple()


@lru_cache(maxsize=2048)
def _datacite_attributes_cached(doi: str) -> Optional[dict]:
    url = f"https://api.datacite.org/dois/{doi}"
    headers = {"User-Agent": "scientific-parser/1.0"}
    for timeout in API_TIMEOUTS:
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            if r.status_code != 200:
                continue
            data = (r.json() or {}).get("data") or {}
            return data.get("attributes") or {}
        except requests.RequestException as e:
            LOGGER.warning("DataCite timeout/failure for DOI %s: %s", doi, e)
            time.sleep(0.15)
    return None


def extract_doi_rule_based(html: str) -> Optional[str]:
    # 0) прямо в сыром HTML (иногда DOI есть в атрибутах, а не в тексте)
    m0 = DOI_RE.search(html)
    if m0:
        return m0.group(0)

    soup = BeautifulSoup(html, "html.parser")

    # 1) meta
    m = soup.find("meta", {"name": "citation_doi"})
    if m and m.get("content"):
        return m.get("content").strip()

    # 2) ссылки вида doi.org/...
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "doi.org/" in href:
            m2 = DOI_RE.search(href)
            if m2:
                return m2.group(0)

    # 3) видимый текст
    text = soup.get_text(" ")
    m3 = DOI_RE.search(text)
    if m3:
        return m3.group(0)

    return None

def normalize_doi(doi: str) -> str:
    doi = doi.strip()
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    doi = doi.replace("https://dx.doi.org/", "").replace("http://dx.doi.org/", "")
    return doi.strip()


def _normalize_text_for_compare(text: str) -> str:
    text = (text or "").strip().lower().replace("ё", "е")
    text = NON_WORD_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _looks_like_capslock_input(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    upper_count = sum(1 for ch in letters if ch.isupper())
    return upper_count / max(len(letters), 1) >= 0.8


def _convert_keyboard_layout(text: str, mapping: Dict[int, str]) -> str:
    return text.translate(mapping)


def _build_title_query_variants(title: str) -> List[str]:
    base = (title or "").strip()
    if not base:
        return []

    variants: List[str] = []

    def add_variant(value: str) -> None:
        v = re.sub(r"\s+", " ", (value or "").strip())
        if len(v) < 3:
            return
        if v not in variants:
            variants.append(v)

    add_variant(base)
    add_variant(base.replace("ё", "е").replace("Ё", "Е"))
    add_variant(NON_WORD_RE.sub(" ", base))

    if _looks_like_capslock_input(base):
        add_variant(base.capitalize())
        add_variant(base.lower())

    add_variant(_convert_keyboard_layout(base, EN_TO_RU_LAYOUT))
    add_variant(_convert_keyboard_layout(base, RU_TO_EN_LAYOUT))

    compact = re.sub(r"\s+", " ", base)
    add_variant(compact)

    return variants[:8]


def _suggest_title_correction(query: str, candidate_titles: List[str]) -> Optional[str]:
    q = (query or "").strip()
    if len(q) < 4 or not candidate_titles:
        return None

    best_title = None
    best_score = 0.0

    q_norm = _normalize_text_for_compare(q)
    for title in candidate_titles:
        t_norm = _normalize_text_for_compare(title)
        if not t_norm:
            continue
        if fuzz:
            score = float(fuzz.WRatio(q_norm, t_norm))
        else:
            score = float(SequenceMatcher(None, q_norm, t_norm).ratio() * 100.0)
        if score > best_score:
            best_score = score
            best_title = title

    # если нашли почти точное совпадение, но с опечатками/раскладкой
    if best_title and 86.0 <= best_score < 99.9:
        return best_title
    return None


def explain_title_interpretation(title: str) -> Dict[str, Any]:
    variants = _build_title_query_variants(title)
    if not variants:
        return {
            "original": (title or "").strip(),
            "variants": [],
            "suggested": None,
        }

    collected_titles: List[str] = []
    for q in variants[:3]:
        items = _crossref_search_items_cached(q, 5)
        for item in items:
            t = ((item.get("title") or [None])[0] if isinstance(item.get("title"), list) else None)
            if t:
                collected_titles.append(t)
    suggested = _suggest_title_correction(title, collected_titles)
    return {
        "original": (title or "").strip(),
        "variants": variants,
        "suggested": suggested,
    }


def _extract_authors_from_crossref_item(item: dict) -> Optional[List[str]]:
    if not isinstance(item.get("author"), list):
        return None

    authors = []
    for author in item["author"]:
        family = (author.get("family") or "").strip()
        given = (author.get("given") or "").strip()
        full_name = (" ".join([given, family])).strip()
        if full_name:
            authors.append(full_name)
    return authors or None


def _extract_year_from_crossref_item(item: dict) -> Optional[str]:
    issued = (item.get("issued") or {}).get("date-parts")
    if isinstance(issued, list) and issued and isinstance(issued[0], list) and issued[0]:
        return str(issued[0][0])
    return None


def _extract_pdf_url_from_crossref_item(item: dict) -> Optional[str]:
    links = item.get("link")
    if isinstance(links, list):
        for link in links:
            content_type = (link.get("content-type") or "").lower()
            if "pdf" in content_type and link.get("URL"):
                return link["URL"]
            url = (link.get("URL") or "").lower()
            if url.endswith(".pdf"):
                return link.get("URL")
    return None


def _paper_from_crossref_item(
    item: dict,
    *,
    enriched_by: str,
    search_score: Optional[float] = None,
    matched_query: Optional[str] = None,
) -> PaperMetadata:
    titles = item.get("title") or []
    title = titles[0] if isinstance(titles, list) and titles else None

    containers = item.get("container-title") or []
    journal = containers[0] if isinstance(containers, list) and containers else None

    doi = item.get("DOI")

    return PaperMetadata(
        title=title,
        authors=_extract_authors_from_crossref_item(item),
        year=_extract_year_from_crossref_item(item),
        journal=journal,
        volume=(item.get("volume") or None),
        issue=(item.get("issue") or None),
        pages=(item.get("page") or None),
        doi=normalize_doi(doi) if doi else None,
        pdf_url=_extract_pdf_url_from_crossref_item(item),
        source_url=f"https://doi.org/{normalize_doi(doi)}" if doi else None,
        enriched_by=enriched_by,
        confidence={},
        search_score=search_score,
        matched_query=matched_query,
    )


def crossref_enrich(doi: str, timeout: int = 15) -> Optional[PaperMetadata]:
    doi = normalize_doi(doi)
    msg = _crossref_work_message_cached(doi)
    if not msg:
        return None
    return _paper_from_crossref_item(msg, enriched_by="crossref")

def guess_title_from_url(url: str) -> Optional[str]:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").strip("/")

    # ScienceDirect: /science/article/pii/S.... -> это НЕ title, это PII
    if "sciencedirect.com" in host and "/pii/" in ("/" + path.lower() + "/"):
        return None

    if not path:
        return None

    slug = path.split("/")[-1]
    slug = slug.replace("-", " ").replace("_", " ").strip()

    # если slug слишком “технический” (много цифр/мало букв) — не считаем его title
    letters = sum(ch.isalpha() for ch in slug)
    digits = sum(ch.isdigit() for ch in slug)
    if letters < 8 or digits > letters:
        return None

    return slug


def _best_crossref_item_by_title(query_title: str, items: List[dict]) -> Optional[dict]:
    q = (query_title or "").strip().lower()
    if not q or not items:
        return None

    best_item = None
    best_score = 0.0

    for it in items:
        titles = it.get("title") or []
        t = titles[0] if isinstance(titles, list) and titles else ""
        t_norm = (t or "").strip().lower()

        # похожесть строк + вес CrossRef score (если есть)
        sim = SequenceMatcher(None, q, t_norm).ratio()
        cr_score = float(it.get("score") or 0.0)

        score = sim * 10.0 + (cr_score / 100.0)  # sim главный

        if score > best_score:
            best_score = score
            best_item = it

    # минимальный порог похожести, чтобы не брать “левые” статьи
    if best_item:
        titles = best_item.get("title") or []
        t = titles[0] if isinstance(titles, list) and titles else ""
        sim = SequenceMatcher(None, q, (t or "").strip().lower()).ratio()
        if sim >= 0.75:
            return best_item

    return None


def search_crossref_candidates_by_title(title: str, rows: int = 12, timeout: int = 15) -> List[PaperMetadata]:
    original_query = _normalize_text_for_compare(title)
    query_variants = _build_title_query_variants(title)
    if not original_query or not query_variants:
        return []

    best_by_doi: Dict[str, PaperMetadata] = {}
    best_without_doi: List[PaperMetadata] = []
    seen_queries = set()
    retrieved_titles: List[str] = []
    started = time.monotonic()

    def process_query(q: str) -> None:
        if (time.monotonic() - started) > SEARCH_TIME_BUDGET_SEC:
            return
        if q in seen_queries:
            return
        seen_queries.add(q)

        params = {"query.bibliographic": q, "rows": rows}
        items = _crossref_search_items_cached(q, rows)
        if not items:
            return
        q_norm = _normalize_text_for_compare(q)
        for item in items:
            paper = _paper_from_crossref_item(item, enriched_by="crossref_search", matched_query=q)
            if not paper.title:
                continue
            retrieved_titles.append(paper.title)

            title_norm = _normalize_text_for_compare(paper.title)
            sim_original = SequenceMatcher(None, original_query, title_norm).ratio()
            sim_variant = SequenceMatcher(None, q_norm, title_norm).ratio()
            sim = max(sim_original, sim_variant)
            if sim < 0.45:
                continue

            crossref_score = float(item.get("score") or 0.0) / 100.0
            final_score = sim * 0.8 + crossref_score * 0.2
            paper.search_score = round(final_score, 4)

            if paper.doi:
                existing = best_by_doi.get(paper.doi)
                if existing is None or (paper.search_score or 0.0) > (existing.search_score or 0.0):
                    best_by_doi[paper.doi] = paper
            else:
                best_without_doi.append(paper)

    for q in query_variants[:4]:
        process_query(q)

    # Умная подсказка по опечаткам на базе реальных заголовков из CrossRef
    corrected = _suggest_title_correction(title, retrieved_titles)
    if corrected:
        process_query(corrected)

    ranked = list(best_by_doi.values()) + best_without_doi
    ranked.sort(key=lambda x: x.search_score or 0.0, reverse=True)
    return ranked[:10]


def search_crossref_candidates_relaxed(title: str, rows: int = 10) -> List[PaperMetadata]:
    query_variants = _build_title_query_variants(title)
    if not query_variants:
        return []

    best_by_key: Dict[str, PaperMetadata] = {}
    started = time.monotonic()
    for q in query_variants[:3]:
        if (time.monotonic() - started) > SEARCH_TIME_BUDGET_SEC:
            break
        items = _crossref_search_items_cached(q, rows)
        for item in items:
            paper = _paper_from_crossref_item(item, enriched_by="crossref_search_relaxed", matched_query=q)
            if not paper.title:
                continue
            title_sim = SequenceMatcher(
                None,
                _normalize_text_for_compare(q),
                _normalize_text_for_compare(paper.title),
            ).ratio()
            crossref_score = float(item.get("score") or 0.0) / 100.0
            paper.search_score = round(max(0.15, title_sim * 0.6 + crossref_score * 0.4), 4)
            key = paper.doi or f"title::{_normalize_text_for_compare(paper.title)}"
            prev = best_by_key.get(key)
            if prev is None or (paper.search_score or 0.0) > (prev.search_score or 0.0):
                best_by_key[key] = paper
    ranked = list(best_by_key.values())
    ranked.sort(key=lambda x: x.search_score or 0.0, reverse=True)
    return ranked[:10]


def crossref_search_doi_by_title(title: str, timeout: int = 15) -> Optional[str]:
    candidates = search_crossref_candidates_by_title(title, rows=8, timeout=timeout)
    for candidate in candidates:
        if candidate.doi:
            return candidate.doi
    return None

def predict_blocks(model, blocks: List[Dict[str, Any]]) -> Tuple[List[str], List[List[float]], List[str]]:
    rows = []
    for b in blocks:
        text = b.get("text", "")
        digits = sum(ch.isdigit() for ch in text)
        uppers = sum(ch.isupper() for ch in text if ch.isalpha())
        alpha = sum(ch.isalpha() for ch in text)
        rows.append(
            {
                "text": text,
                "text_char": text,
                "tag": b.get("tag", ""),
                "classes": " ".join(b.get("classes") or []),
                "text_length": b.get("text_length", len(text)),
                "word_count": b.get("word_count", len(text.split())),
                "is_visible": int(b.get("is_visible", 1)),
                "dom_depth": float(b.get("dom_depth", 0)),
                "sibling_index": float(b.get("sibling_index", 0)),
                "link_density": float(b.get("link_density", 0.0)),
                "digit_ratio": float(digits / max(len(text), 1)),
                "upper_ratio": float(uppers / max(alpha, 1)),
                "page_url": "",
                "source_type": "",
                "domain": "",
            }
        )

    X_df = pd.DataFrame(rows)

    preds = model.predict(X_df).tolist()
    proba = model.predict_proba(X_df).tolist()
    classes = list(model.classes_)
    return preds, proba, classes


def choose_best_text(blocks, preds, proba, classes, target_label: str) -> Tuple[Optional[str], float]:
    if target_label not in classes:
        return None, 0.0

    idx = classes.index(target_label)

    best_text = None
    best_score = 0.0

    for b, y, p in zip(blocks, preds, proba):
        score = float(p[idx])

        # чуть усиливаем h1/h2 для title
        if target_label == "title" and b.get("tag") in ("h1", "h2"):
            score *= 1.15

        # чуть усиливаем короткие блоки для year/doi
        if target_label in ("year", "doi") and b.get("word_count", 0) <= 12:
            score *= 1.10

        if score > best_score:
            best_score = score
            best_text = b.get("text")

    return best_text, best_score


def parse_year(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    return m.group(1) if m else None


def parse_doi(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = DOI_RE.search(text)
    return normalize_doi(m.group(0)) if m else None


def _split_author(author: str) -> Tuple[str, str]:
    chunks = [p for p in re.split(r"\s+", (author or "").strip()) if p]
    if not chunks:
        return "", ""
    if len(chunks) == 1:
        return chunks[0], ""
    family = chunks[-1]
    given = " ".join(chunks[:-1])
    return family, given


def _format_author_gost(author: str) -> str:
    family, given = _split_author(author)
    if not family and not given:
        return ""
    initials = "".join(f"{x[0].upper()}." for x in given.split() if x)
    return f"{family} {initials}".strip()


def _format_author_apa(author: str) -> str:
    family, given = _split_author(author)
    if not family and not given:
        return ""
    initials = " ".join(f"{x[0].upper()}." for x in given.split() if x)
    return f"{family}, {initials}".strip(", ").strip()


def _format_author_ieee(author: str) -> str:
    family, given = _split_author(author)
    if not family and not given:
        return ""
    initials = " ".join(f"{x[0].upper()}." for x in given.split() if x)
    return f"{initials} {family}".strip()


def _format_authors(authors: Optional[List[str]], style: str) -> str:
    if not authors:
        return ""
    style = (style or "gost").lower()
    if style == "apa":
        values = [_format_author_apa(a) for a in authors if a]
        values = [v for v in values if v]
        if len(values) <= 7:
            return ", ".join(values)
        return ", ".join(values[:6] + ["...", values[-1]])
    if style == "ieee":
        values = [_format_author_ieee(a) for a in authors if a]
        values = [v for v in values if v]
        if len(values) <= 6:
            return ", ".join(values)
        return ", ".join(values[:6]) + ", et al."

    values = [_format_author_gost(a) for a in authors if a]
    values = [v for v in values if v]
    if len(values) <= 3:
        return ", ".join(values)
    return ", ".join(values[:3]) + " [и др.]"


def format_citation(m: PaperMetadata, style: str = "gost") -> str:
    style = (style or "gost").lower()

    if style == "apa":
        a = _format_authors(m.authors, "apa")
        year = m.year or "n.d."
        title = m.title or "Untitled"
        journal = m.journal or ""
        volume = f", {m.volume}" if m.volume else ""
        issue = f"({m.issue})" if m.issue else ""
        pages = f", {m.pages}" if m.pages else ""
        doi = f" https://doi.org/{m.doi}" if m.doi else ""
        left = f"{a} ({year}). {title}." if a else f"{title}. ({year})."
        right = f" {journal}{volume}{issue}{pages}.{doi}".strip()
        return (left + " " + right).strip()

    if style == "ieee":
        a = _format_authors(m.authors, "ieee")
        title = f"\"{m.title or 'Untitled'}\""
        journal = m.journal or "Unknown source"
        year = m.year or "n.d."
        vol = f", vol. {m.volume}" if m.volume else ""
        no = f", no. {m.issue}" if m.issue else ""
        pages = f", pp. {m.pages}" if m.pages else ""
        doi = f", doi: {m.doi}" if m.doi else ""
        prefix = f"{a}, " if a else ""
        return f"{prefix}{title}, {journal}{vol}{no}{pages}, {year}{doi}."

    # GOST-like

    parts = []
    gost_authors = _format_authors(m.authors, "gost")
    if gost_authors:
        parts.append(gost_authors)

    if m.title:
        parts.append(m.title)

    left = ". ".join([p for p in parts if p]).strip()

    right_parts = []
    if m.journal:
        right_parts.append(m.journal)
    if m.year:
        right_parts.append(m.year)
    if m.volume:
        right_parts.append(f"Т. {m.volume}")
    if m.issue:
        right_parts.append(f"№ {m.issue}")
    if m.pages:
        right_parts.append(f"С. {m.pages}")
    if m.doi:
        right_parts.append(f"DOI: {m.doi}")

    right = ". ".join([p for p in right_parts if p]).strip()
    if right and left:
        return f"{left} // {right}."
    if right:
        return f"{right}."
    return left or "Недостаточно данных для оформления ссылки."


def format_citation_ru_gost_like(m: PaperMetadata) -> str:
    return format_citation(m, style="gost")


def format_bibliography_list(items: List[PaperMetadata], style: str = "gost") -> str:
    if not items:
        return "1. Недостаточно данных для оформления ссылки."

    lines = []
    for i, item in enumerate(items, start=1):
        lines.append(f"{i}. {format_citation(item, style=style)}")
    return "\n".join(lines)


def export_bibtex(items: List[PaperMetadata]) -> str:
    entries = []
    for idx, item in enumerate(items, start=1):
        key_base = (item.authors[0].split()[-1] if item.authors else "source").lower()
        key = f"{re.sub(r'[^a-z0-9]+', '', key_base)}{item.year or 'nd'}{idx}"
        title = item.title or "Untitled"
        authors = " and ".join(item.authors or [])
        fields = [
            f"  title = {{{title}}}",
            f"  author = {{{authors}}}" if authors else None,
            f"  journal = {{{item.journal}}}" if item.journal else None,
            f"  year = {{{item.year}}}" if item.year else None,
            f"  volume = {{{item.volume}}}" if item.volume else None,
            f"  number = {{{item.issue}}}" if item.issue else None,
            f"  pages = {{{item.pages}}}" if item.pages else None,
            f"  doi = {{{item.doi}}}" if item.doi else None,
            f"  url = {{{item.source_url or ''}}}" if (item.source_url or item.doi) else None,
        ]
        filtered = ",\n".join(x for x in fields if x)
        entries.append(f"@article{{{key},\n{filtered}\n}}")
    return "\n\n".join(entries) if entries else "% empty bibliography"


def export_ris(items: List[PaperMetadata]) -> str:
    chunks: List[str] = []
    for item in items:
        lines = ["TY  - JOUR"]
        for author in item.authors or []:
            lines.append(f"AU  - {author}")
        if item.title:
            lines.append(f"TI  - {item.title}")
        if item.journal:
            lines.append(f"JO  - {item.journal}")
        if item.year:
            lines.append(f"PY  - {item.year}")
        if item.volume:
            lines.append(f"VL  - {item.volume}")
        if item.issue:
            lines.append(f"IS  - {item.issue}")
        if item.pages:
            lines.append(f"SP  - {item.pages}")
        if item.doi:
            lines.append(f"DO  - {item.doi}")
        if item.source_url:
            lines.append(f"UR  - {item.source_url}")
        lines.append("ER  -")
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks) if chunks else "TY  - GEN\nER  -"


def extract_pdf_url_from_html(html: str, source_url: Optional[str] = None) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")

    selectors = [
        ("meta", {"name": "citation_pdf_url"}, "content"),
        ("meta", {"name": "dc.identifier", "scheme": "DCTERMS.URI"}, "content"),
    ]
    for tag, attrs, key in selectors:
        node = soup.find(tag, attrs)
        if node and node.get(key):
            raw_url = node.get(key).strip()
            if raw_url.lower().endswith(".pdf") or "pdf" in raw_url.lower():
                return urljoin(source_url or "", raw_url)

    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        text = (a.get_text(" ") or "").lower()
        if ".pdf" in href.lower() or "download pdf" in text or text.strip() == "pdf":
            return urljoin(source_url or "", href)

    return None


def extract_cyberleninka_metadata(html: str, source_url: Optional[str] = None) -> PaperMetadata:
    soup = BeautifulSoup(html, "html.parser")

    title = None
    m_title = soup.find("meta", {"property": "og:title"})
    if m_title and m_title.get("content"):
        title = m_title.get("content").strip()
        title = re.sub(r"\s*[–-]\s*тема научной статьи.*$", "", title, flags=re.IGNORECASE)
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(" ").strip()

    authors = []
    for m_author in soup.find_all("meta", {"name": "citation_author"}):
        a = (m_author.get("content") or "").strip()
        if a:
            authors.append(a)
    authors = authors or None

    year = None
    m_date = soup.find("meta", {"name": "citation_publication_date"})
    if m_date and m_date.get("content"):
        year = parse_year(m_date.get("content"))
    if not year:
        m_pub = soup.find("meta", {"property": "article:published_time"})
        if m_pub and m_pub.get("content"):
            year = parse_year(m_pub.get("content"))

    journal = None
    m_journal = soup.find("meta", {"name": "citation_journal_title"})
    if m_journal and m_journal.get("content"):
        journal = m_journal.get("content").strip()

    doi = extract_doi_rule_based(html)
    doi = normalize_doi(doi) if doi else None
    pdf_url = extract_pdf_url_from_html(html, source_url=source_url)

    return PaperMetadata(
        title=title,
        authors=authors,
        year=year,
        journal=journal,
        doi=doi,
        pdf_url=pdf_url,
        source_url=source_url,
        enriched_by="rule_based_cyberleninka",
        confidence={},
    )


def classify_article_page(html: str, url: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ").lower()
    host = (urlparse(url).netloc or "").lower()
    strong_hosts = (
        "arxiv.org",
        "ieeexplore.ieee.org",
        "sciencedirect.com",
        "link.springer.com",
        "cyberleninka.ru",
        "acm.org",
    )

    score = 0.0
    reasons: List[str] = []

    if any(h in host for h in strong_hosts):
        score += 0.35
        reasons.append("известный научный домен")

    has_citation_meta = bool(soup.select('meta[name^="citation_"]'))
    if has_citation_meta:
        score += 0.25
        reasons.append("есть citation_* метаданные")

    has_doi = bool(extract_doi_rule_based(html))
    if has_doi:
        score += 0.20
        reasons.append("обнаружен DOI")

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    long_paragraphs = sum(1 for p in paragraphs if len(p) > 220)
    if long_paragraphs >= 3:
        score += 0.15
        reasons.append("длинный структурированный текст")

    if any(k in text for k in ["abstract", "аннотация", "keywords", "ключевые слова", "references", "литература"]):
        score += 0.15
        reasons.append("обнаружены признаки научной статьи")

    if any(k in text for k in ["privacy policy", "contact us", "login", "sign in"]) and score < 0.5:
        score -= 0.15
        reasons.append("есть признаки сервисной страницы")

    score = max(0.0, min(1.0, score))
    is_article = score >= 0.45
    return {
        "is_article": is_article,
        "score": round(score, 3),
        "reason": "; ".join(reasons) if reasons else "недостаточно сигналов",
    }


def extract_metadata_from_url(url: str) -> PaperMetadata:
    meta = PaperMetadata(source_url=url, confidence={})

    try:
        html = fetch_html(url)
    except Exception:
        # Фоллбек: пробуем угадать название из URL и найти DOI через CrossRef
        guessed = guess_title_from_url(url)
        if guessed:
            doi = crossref_search_doi_by_title(guessed)
            if doi:
                enriched = crossref_enrich(doi)
                if enriched:
                    enriched.source_url = url
                    return enriched
        return meta

    host = (urlparse(url).netloc or "").lower()
    page_cls = classify_article_page(html, url)
    meta.confidence["page_article_score"] = float(page_cls["score"])
    meta.confidence["page_is_article"] = 1.0 if page_cls["is_article"] else 0.0
    meta.confidence["page_article_reason"] = page_cls["reason"]

    # Предклассификатор страницы: если это явно не статья и домен не из приоритетных, не запускаем тяжёлый ML.
    priority_hosts = ("arxiv.org", "ieeexplore.ieee.org", "sciencedirect.com", "link.springer.com", "cyberleninka.ru")
    if (not page_cls["is_article"]) and (not any(h in host for h in priority_hosts)):
        guessed = extract_title_from_html_basic(html) or guess_title_from_url(url)
        if guessed:
            meta.title = guessed
        return meta

    pdf_from_html = extract_pdf_url_from_html(html, source_url=url)

    if "arxiv.org" in host:
        meta_rb = extract_arxiv_metadata(html)
        meta_rb.pdf_url = meta_rb.pdf_url or pdf_from_html
        meta_rb.confidence = meta_rb.confidence or {}
        meta_rb.confidence.update(meta.confidence or {})

        # если нашли DOI — можно обогатить через CrossRef
        if meta_rb.doi:
            enriched = crossref_enrich(meta_rb.doi)
            if enriched:
                enriched.source_url = url
                enriched.confidence = meta_rb.confidence or {}
                enriched.pdf_url = enriched.pdf_url or meta_rb.pdf_url
                return enriched

        meta_rb.source_url = url
        return meta_rb

    if "cyberleninka.ru" in host:
        meta_rb = extract_cyberleninka_metadata(html, source_url=url)
        meta_rb.pdf_url = meta_rb.pdf_url or pdf_from_html
        meta_rb.confidence = meta_rb.confidence or {}
        meta_rb.confidence.update(meta.confidence or {})

        if meta_rb.doi:
            enriched = crossref_enrich(meta_rb.doi)
            if enriched:
                enriched.source_url = url
                enriched.confidence = meta_rb.confidence or {}
                enriched.pdf_url = enriched.pdf_url or meta_rb.pdf_url
                # для русских источников title из страницы часто лучше локализован
                enriched.title = meta_rb.title or enriched.title
                return enriched

        return meta_rb

    # быстрый DOI rule-based до ML
    doi0 = extract_doi_rule_based(html)
    if doi0:
        meta.doi = normalize_doi(doi0)

    if meta.doi:
        enriched = crossref_enrich(meta.doi)
        if enriched:
            enriched.source_url = meta.source_url
            enriched.confidence = meta.confidence or {}
            enriched.pdf_url = enriched.pdf_url or pdf_from_html
            return enriched

    blocks = extract_blocks_from_html(html)
    model = joblib.load(MODEL_PATH)

    preds, proba, classes = predict_blocks(model, blocks)

    title_text, title_conf = choose_best_text(blocks, preds, proba, classes, "title")
    author_text, author_conf = choose_best_text(blocks, preds, proba, classes, "author")
    journal_text, journal_conf = choose_best_text(blocks, preds, proba, classes, "journal")
    year_text, year_conf = choose_best_text(blocks, preds, proba, classes, "year")
    doi_text, doi_conf = choose_best_text(blocks, preds, proba, classes, "doi")

    meta.title = title_text
    meta.journal = journal_text
    meta.year = parse_year(year_text) or meta.year
    meta.doi = meta.doi or parse_doi(doi_text)
    meta.pdf_url = pdf_from_html

    # авторов в твоей разметке очень мало (author=3 блока), поэтому пока делаем супер‑осторожно:
    # берём author_text как одну строку и пытаемся грубо разделить
    if author_text:
        # разделители: запятая, точка с запятой, " and "
        raw = re.split(r"\s*(?:,|;|\band\b)\s*", author_text.strip())
        raw = [x.strip() for x in raw if len(x.strip()) >= 3]
        meta.authors = raw[:10] if raw else None

    meta.confidence = {
        "title": float(title_conf),
        "author": float(author_conf),
        "journal": float(journal_conf),
        "year": float(year_conf),
        "doi": float(doi_conf),
    }

    # обогащение CrossRef по DOI (лайфхак от преподавателя)
    if meta.doi:
        enriched = crossref_enrich(meta.doi)
        if enriched:
            # CrossRef считаем более надёжным по полям, кроме source_url
            enriched.source_url = meta.source_url
            enriched.confidence = meta.confidence
            enriched.pdf_url = enriched.pdf_url or meta.pdf_url
            return enriched

    # если ML не дал ничего полезного — пробуем вытащить title из html и найти DOI через CrossRef
    if (not meta.doi and not meta.title):
        t = extract_title_from_html_basic(html)
        if t:
            doi2 = crossref_search_doi_by_title(t)
            if doi2:
                enriched = crossref_enrich(doi2)
                if enriched:
                    enriched.source_url = url
                    return enriched

    return meta

def search_metadata_candidates_from_title(title: str, max_results: int = 5) -> List[PaperMetadata]:
    title = (title or "").strip()
    if not title:
        return []

    candidates = search_crossref_candidates_by_title(title, rows=15)
    if candidates:
        return candidates[:max_results]
    relaxed = search_crossref_candidates_relaxed(title, rows=10)
    if relaxed:
        return relaxed[:max_results]
    return [PaperMetadata(title=title, source_url=None, confidence={})]


def extract_metadata_from_title(title: str) -> PaperMetadata:
    candidates = search_metadata_candidates_from_title(title, max_results=1)
    if candidates:
        return candidates[0]
    return PaperMetadata(title=(title or "").strip(), source_url=None, confidence={})

def extract_metadata_from_doi(doi: str) -> PaperMetadata:
    doi = normalize_doi(doi)

    meta = crossref_enrich(doi)
    if meta:
        meta.source_url = f"https://doi.org/{doi}"
        meta.confidence = meta.confidence or {}
        return meta

    meta2 = datacite_enrich(doi)
    if meta2:
        meta2.source_url = f"https://doi.org/{doi}"
        meta2.confidence = meta2.confidence or {}
        return meta2

    return PaperMetadata(doi=doi, source_url=f"https://doi.org/{doi}", enriched_by=None, confidence={})

def extract_title_from_html_basic(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")

    m = soup.find("meta", {"name": "citation_title"})
    if m and m.get("content"):
        t = m.get("content").strip()
        return t if len(t) >= 8 else None

    og = soup.find("meta", {"property": "og:title"})
    if og and og.get("content"):
        t = og.get("content").strip()
        return t if len(t) >= 8 else None

    if soup.title and soup.title.get_text():
        t = soup.title.get_text().strip()
        # часто в title ещё “| ScienceDirect”
        t = re.split(r"\s+[\|\-]\s+", t)[0].strip()
        t = re.sub(r"\s*ScienceDirect\s*$", "", t, flags=re.IGNORECASE).strip()
        return t if len(t) >= 8 else None

    return None

def extract_arxiv_metadata(html: str) -> PaperMetadata:
    soup = BeautifulSoup(html, "html.parser")

    title = None
    t = soup.select_one("h1.title")
    if t:
        title = t.get_text(" ").replace("Title:", "").strip()

    authors = None
    a = soup.select_one("div.authors")
    if a:
        txt = a.get_text(" ").replace("Authors:", "").strip()
        parts = [x.strip() for x in txt.split(",") if x.strip()]
        authors = parts or None

    year = None
    d = soup.select_one("div.dateline")
    if d:
        # [Submitted on 21 Dec 2022]
        m = re.search(r"\b(19\d{2}|20\d{2})\b", d.get_text(" "))
        if m:
            year = m.group(1)

    doi = extract_doi_rule_based(html)
    doi = normalize_doi(doi) if doi else None

    return PaperMetadata(
        title=title,
        authors=authors,
        year=year,
        journal="arXiv",
        doi=doi,
        pdf_url=extract_pdf_url_from_html(html, source_url="https://arxiv.org"),
        enriched_by="rule_based_arxiv",
        confidence={},
    )

def datacite_enrich(doi: str, timeout: int = 15) -> Optional[PaperMetadata]:
    doi = normalize_doi(doi)
    attrs = _datacite_attributes_cached(doi)
    if not attrs:
        return None

    title = None
    titles = attrs.get("titles") or []
    if titles and isinstance(titles, list):
        t0 = titles[0] or {}
        title = (t0.get("title") or "").strip() or None

    year = None
    pub_year = attrs.get("published")
    if isinstance(pub_year, str):
        m = re.search(r"\b(19\d{2}|20\d{2})\b", pub_year)
        year = m.group(1) if m else None

    authors = None
    creators = attrs.get("creators") or []
    if creators and isinstance(creators, list):
        a = []
        for c in creators:
            name = (c.get("name") or "").strip()
            if not name:
                continue

            # "Last, First" -> "First Last"
            if "," in name:
                parts = [p.strip() for p in name.split(",", 1)]
                if len(parts) == 2 and parts[0] and parts[1]:
                    name = f"{parts[1]} {parts[0]}".strip()

            a.append(name)
        authors = a or None

    journal = None
    container = (attrs.get("container") or {}).get("title")
    if isinstance(container, str) and container.strip():
        journal = container.strip()

    return PaperMetadata(
        title=title,
        authors=authors,
        year=year,
        journal=journal,
        doi=doi,
        volume=attrs.get("volumeNumber") or None,
        issue=attrs.get("issue") or None,
        pages=attrs.get("firstPage") or None,
        enriched_by="datacite",
        confidence={},
    )

if __name__ == "__main__":
    test_url = "https://www.sciencedirect.com/science/article/pii/S1738573325005388"
    m = extract_metadata_from_url(test_url)
    print(json.dumps(asdict(m), ensure_ascii=False, indent=2))
    print("\nCITATION:\n", format_citation_ru_gost_like(m))
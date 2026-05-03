# inference_pipeline.py

import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urljoin, quote
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
# Строгая проверка «похожести на DOI» для пользовательского ввода (до запроса к Crossref).
DOI_STRICT_INPUT_RE = re.compile(r"^10\.\d{4,}/\S+$", re.IGNORECASE)
_BOOKISH_BANNER_YEAR_RE = re.compile(r"\(\s*(?:19|20)\d{2}\s*\)")
NON_WORD_RE = re.compile(r"[^\w\s\-]", re.UNICODE)
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
_HTML_TAGS_RE = re.compile(r"<[^>]+>")

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
    proxies = None
    tor_proxy = (os.getenv("TOR_SOCKS_PROXY") or "").strip()
    if tor_proxy:
        proxies = {"http": tor_proxy, "https": tor_proxy}
    r = s.get(url, timeout=timeout, proxies=proxies)
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


def _openalex_mailto() -> str:
    return (os.getenv("OPENALEX_MAILTO") or "example@example.com").strip()


def _openalex_headers() -> dict:
    return {
        "User-Agent": f"scientific-parser/1.0 (mailto: {_openalex_mailto()})",
        "Accept": "application/json",
    }


@lru_cache(maxsize=1024)
def _openalex_works_search_cached(query: str, per_page: int) -> Tuple[dict, ...]:
    """
    OpenAlex fulltext search on works. Returns tuple of work dicts.
    https://api.openalex.org/works?search=...
    """
    q = (query or "").strip()
    if not q:
        return tuple()
    params = {
        "search": q,
        "per_page": str(max(1, min(per_page, 50))),
        "mailto": _openalex_mailto(),
    }
    for timeout in API_TIMEOUTS:
        try:
            r = requests.get(
                "https://api.openalex.org/works",
                params=params,
                timeout=timeout,
                headers=_openalex_headers(),
            )
            if r.status_code != 200:
                continue
            data = r.json() or {}
            results = data.get("results")
            if not isinstance(results, list):
                return tuple()
            out: List[dict] = []
            for w in results:
                if isinstance(w, dict):
                    out.append(w)
            return tuple(out)
        except requests.RequestException as e:
            LOGGER.warning("OpenAlex search timeout/failure for query '%s': %s", q, e)
            time.sleep(0.15)
    return tuple()


def _location_landing_url(loc: Optional[dict]) -> Optional[str]:
    if not loc or not isinstance(loc, dict):
        return None
    url = (loc.get("landing_page_url") or "").strip()
    return url or None


def _location_pdf_url(loc: Optional[dict]) -> Optional[str]:
    if not loc or not isinstance(loc, dict):
        return None
    pdf_url = (loc.get("pdf_url") or "").strip()
    return pdf_url or None


def _extract_journal_from_openalex_work(work: dict) -> Optional[str]:
    loc = work.get("primary_location") if isinstance(work.get("primary_location"), dict) else None
    src = loc.get("source") if isinstance(loc, dict) and isinstance(loc.get("source"), dict) else None
    name = (src.get("display_name") or "").strip() if src else ""
    return name or None


def _extract_pdf_url_from_openalex_work(work: dict) -> Optional[str]:
    best = work.get("best_oa_location") if isinstance(work.get("best_oa_location"), dict) else None
    pdf1 = _location_pdf_url(best)
    if pdf1:
        return pdf1
    prim = work.get("primary_location") if isinstance(work.get("primary_location"), dict) else None
    pdf2 = _location_pdf_url(prim)
    if pdf2:
        return pdf2
    locs = work.get("locations")
    if isinstance(locs, list):
        for loc in locs:
            pdf3 = _location_pdf_url(loc if isinstance(loc, dict) else None)
            if pdf3:
                return pdf3
    return None


def _extract_source_url_from_openalex_work(work: dict) -> Optional[str]:
    """
    Prefer a human-readable landing page; DOI resolver as fallback.
    """
    doi = work.get("doi") or ""
    doi = doi.strip() if isinstance(doi, str) else ""
    doi_norm = normalize_doi(doi.replace("https://doi.org/", "")) if doi else ""

    prim = work.get("primary_location") if isinstance(work.get("primary_location"), dict) else None
    landing = _location_landing_url(prim)
    if landing:
        return landing

    best = work.get("best_oa_location") if isinstance(work.get("best_oa_location"), dict) else None
    landing2 = _location_landing_url(best)
    if landing2:
        return landing2

    locs = work.get("locations")
    if isinstance(locs, list):
        for loc in locs:
            landing3 = _location_landing_url(loc if isinstance(loc, dict) else None)
            if landing3:
                return landing3

    if doi_norm:
        return f"https://doi.org/{doi_norm}"
    oid = work.get("id") or ""
    if isinstance(oid, str) and oid.startswith("http"):
        return oid
    return None


def _extract_authors_from_openalex_work(work: dict) -> Optional[List[str]]:
    authorships = work.get("authorships")
    if not isinstance(authorships, list):
        return None
    authors: List[str] = []
    for ash in authorships:
        if not isinstance(ash, dict):
            continue
        ia = ash.get("author") if isinstance(ash.get("author"), dict) else {}
        dn = (ia.get("display_name") or "").strip()
        if dn:
            authors.append(dn)
            continue
        given = (ia.get("given_name") or "").strip()
        family = (ia.get("family_name") or "").strip()
        full = (" ".join([given, family])).strip()
        if full:
            authors.append(full)
    out = authors[:40]
    return out or None


def _extract_year_from_openalex_work(work: dict) -> Optional[str]:
    y = work.get("publication_year")
    try:
        if y is None:
            return None
        return str(int(y))
    except Exception:
        return None


def _extract_biblio_pages_from_openalex_work(work: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    biblio = work.get("biblio") if isinstance(work.get("biblio"), dict) else {}
    volume = biblio.get("volume")
    issue = biblio.get("issue")
    first = biblio.get("first_page")
    last = biblio.get("last_page")
    vol_s = str(volume).strip() if volume not in (None, "") else None
    issue_s = str(issue).strip() if issue not in (None, "") else None
    pages_s = None
    if first not in (None, "") and last not in (None, ""):
        pages_s = f"{first}-{last}"
    elif first not in (None, ""):
        pages_s = str(first).strip()
    return vol_s, issue_s, pages_s


def _paper_from_openalex_work(
    work: dict,
    *,
    enriched_by: str,
    matched_query: str,
    relevance: float,
    title_sim: float,
) -> PaperMetadata:
    title = (work.get("display_name") or "").strip() or None

    doi_raw = work.get("doi") or ""
    doi_raw = doi_raw.strip() if isinstance(doi_raw, str) else ""
    doi_norm = normalize_doi(doi_raw.replace("https://doi.org/", "")) if doi_raw else None

    journal = _extract_journal_from_openalex_work(work)
    vol_s, issue_s, pages_s = _extract_biblio_pages_from_openalex_work(work)

    pdf_url = _extract_pdf_url_from_openalex_work(work)
    source_url = _extract_source_url_from_openalex_work(work)

    rel = max(0.0, min(1.0, float(relevance)))
    ts = max(0.0, min(1.0, float(title_sim)))
    final_score = ts * 0.55 + rel * 0.35 + 0.05

    meta = PaperMetadata(
        title=title,
        authors=_extract_authors_from_openalex_work(work),
        year=_extract_year_from_openalex_work(work),
        journal=journal,
        volume=vol_s,
        issue=issue_s,
        pages=pages_s,
        doi=doi_norm,
        pdf_url=pdf_url,
        source_url=source_url,
        enriched_by=enriched_by,
        confidence={},
        search_score=round(min(1.0, max(0.0, final_score)), 4),
        matched_query=matched_query,
    )

    # If we have DOI — prefer CrossRef canonical metadata for bibliographic fields
    if meta.doi:
        enriched = crossref_enrich(meta.doi)
        if enriched:
            enriched.source_url = meta.source_url or enriched.source_url
            enriched.pdf_url = enriched.pdf_url or meta.pdf_url
            enriched.search_score = meta.search_score
            enriched.matched_query = matched_query
            enriched.enriched_by = enriched_by
            # Для русскоязычных запросов не теряем русскоязычный заголовок из OpenAlex,
            # если CrossRef вернул только латиницу/англоязычный вариант.
            if meta.title and query_has_cyrillic(matched_query):
                mt = meta.title or ""
                et = enriched.title or ""
                if CYRILLIC_RE.search(mt) and not CYRILLIC_RE.search(et):
                    enriched.title = mt
            return enriched

    return meta


def query_has_cyrillic(text: str) -> bool:
    return bool(CYRILLIC_RE.search(text or ""))


def search_openalex_candidates_by_title(title: str, per_page: int = 25) -> List[PaperMetadata]:
    """
    Supplemental catalog search via OpenAlex (helps RU/non-CrossRef coverage).
    """
    query_variants = _build_title_query_variants(title)
    if not query_variants:
        return []

    original_norm = _normalize_text_for_compare(title)
    started = time.monotonic()

    best_by_doi: Dict[str, PaperMetadata] = {}
    best_without_doi: List[PaperMetadata] = []
    seen_queries = set()

    def process_query(q: str) -> None:
        if (time.monotonic() - started) > SEARCH_TIME_BUDGET_SEC:
            return
        if q in seen_queries:
            return
        seen_queries.add(q)

        works = _openalex_works_search_cached(q, min(per_page, 25))
        if not works:
            return

        rel_scores: List[float] = []
        raw_scores: List[float] = []
        for w in works:
            rs = w.get("relevance_score")
            try:
                if rs is None:
                    raw_scores.append(0.0)
                else:
                    raw_scores.append(float(rs))
            except Exception:
                raw_scores.append(0.0)

        mx = max(raw_scores) if raw_scores else 1.0
        if mx <= 0:
            mx = 1.0

        q_norm = _normalize_text_for_compare(q)

        for idx, work in enumerate(works):
            try:
                rs_raw = raw_scores[idx] if idx < len(raw_scores) else 0.0
                relevance = float(rs_raw) / mx if mx > 0 else 0.0
            except Exception:
                relevance = 0.0

            if not isinstance(work, dict):
                continue
            w_title = (work.get("display_name") or "").strip()
            if len(w_title) < 8:
                continue

            title_norm = _normalize_text_for_compare(w_title)
            sim_o = SequenceMatcher(None, original_norm, title_norm).ratio()
            sim_q = SequenceMatcher(None, q_norm, title_norm).ratio()
            sim = max(sim_o, sim_q)
            if sim < 0.35:
                continue

            paper = _paper_from_openalex_work(
                work,
                enriched_by="openalex_search",
                matched_query=q,
                relevance=relevance,
                title_sim=sim,
            )

            if paper.doi:
                existing = best_by_doi.get(paper.doi)
                if existing is None or (paper.search_score or 0.0) > (existing.search_score or 0.0):
                    best_by_doi[paper.doi] = paper
            else:
                best_without_doi.append(paper)

    for q in query_variants[:4]:
        process_query(q)

    ranked = list(best_by_doi.values()) + best_without_doi
    ranked.sort(key=lambda x: x.search_score or 0.0, reverse=True)
    return ranked[:25]


def _cyberleninka_pdf_url_from_article_page(source_url: str) -> Optional[str]:
    """
    У статей CyberLeninka PDF отдаётся по тому же пути, что и страница, с суффиксом /pdf.
    Пример: https://cyberleninka.ru/article/n/slug -> .../slug/pdf
    """
    u = (source_url or "").strip()
    if not u or "cyberleninka.ru" not in u.lower():
        return None
    if "/article/n/" not in u:
        return None
    base = u.split("#", 1)[0].rstrip("/")
    if base.lower().endswith("/pdf"):
        cand = base
    else:
        cand = base + "/pdf"
    return cand if is_public_http_url(cand) else None


def _strip_light_html_to_text(raw: str) -> str:
    s = (raw or "").replace("\ufeff", "")
    s = _HTML_TAGS_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _cyberleninka_prime_requests_session(sess: requests.Session) -> None:
    """
    CyberLeninka отдаёт /api/search 400 без «прогрева» сессии так же, как в браузере.
    """
    try:
        sess.get("https://cyberleninka.ru/", timeout=min(10, API_TIMEOUTS[1]))
    except requests.RequestException:
        pass


def _cyberleninka_search_page_url(query: str) -> str:
    q = (query or "").strip()
    return "https://cyberleninka.ru/search?q=" + quote(q)


@lru_cache(maxsize=256)
def _cyberleninka_search_api_cached(payload_key: str) -> Tuple[dict, ...]:
    """
    POST https://cyberleninka.ru/api/search
    Тело — JSON: {mode, q, size, from}. Возвращает кортеж «статей» (как dict) из поля articles.
    """
    try:
        payload = json.loads(payload_key)
    except Exception:
        return tuple()

    if not isinstance(payload, dict):
        return tuple()

    sess = requests.Session()
    sess.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
        }
    )

    q = str(payload.get("q") or "").strip()
    if not q:
        return tuple()

    ref = _cyberleninka_search_page_url(q)
    _cyberleninka_prime_requests_session(sess)

    try:
        sess.get(ref, timeout=min(10, API_TIMEOUTS[1]))
    except requests.RequestException:
        pass

    headers = {
        "Content-Type": "application/json",
        "Origin": "https://cyberleninka.ru",
        "Referer": ref,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    for timeout in API_TIMEOUTS:
        try:
            r = sess.post(
                "https://cyberleninka.ru/api/search",
                data=body,
                headers=headers,
                timeout=timeout,
            )
            if r.status_code != 200:
                continue
            data = r.json() or {}
            arts = data.get("articles") if isinstance(data, dict) else None
            if isinstance(arts, list):
                out: List[dict] = []
                for a in arts:
                    if isinstance(a, dict):
                        out.append(a)
                return tuple(out)
        except (requests.RequestException, ValueError) as e:
            LOGGER.warning("CyberLeninka search API failure for query '%s': %s", q, e)
            time.sleep(0.15)
    return tuple()


def _paper_from_cyberleninka_hit(
    item: dict,
    *,
    matched_query: str,
    title_sim: float,
) -> Optional[PaperMetadata]:
    link = (item.get("link") or "").strip()
    if not link.startswith("/"):
        return None

    source_url = urljoin("https://cyberleninka.ru", link)
    if not is_public_http_url(source_url):
        return None

    raw_title = (item.get("name") or "").strip()
    title = _strip_light_html_to_text(raw_title) or raw_title or None

    authors = item.get("authors")
    authors_list: Optional[List[str]] = None
    if isinstance(authors, list):
        cleaned = []
        for a in authors:
            if isinstance(a, str):
                s = a.strip()
                if s:
                    cleaned.append(s)
        authors_list = cleaned[:40] or None

    year_s: Optional[str] = None
    y = item.get("year")
    if isinstance(y, int):
        year_s = str(y)
    elif isinstance(y, str) and y.strip():
        year_s = y.strip()

    journal = (item.get("journal") or "").strip() or None

    ts = max(0.0, min(1.0, float(title_sim)))
    final_score = ts * 0.75 + 0.08

    pdf_guess = _cyberleninka_pdf_url_from_article_page(source_url)

    return PaperMetadata(
        title=title,
        authors=authors_list,
        year=year_s,
        journal=journal,
        doi=None,
        pdf_url=pdf_guess,
        source_url=source_url,
        enriched_by="cyberleninka_search",
        confidence={},
        search_score=round(min(1.0, max(0.0, final_score)), 4),
        matched_query=matched_query,
    )


def search_cyberleninka_candidates_by_title(title: str, per_page: int = 25) -> List[PaperMetadata]:
    """
    Прямой поиск по каталогу CyberLeninka через их JSON API (/api/search).
    Нужен для русскоязычных записей, которые плохо покрыты CrossRef/OpenAlex.
    """
    query_variants = _build_title_query_variants(title)
    if not query_variants:
        return []

    original_norm = _normalize_text_for_compare(title)
    started = time.monotonic()

    best_by_url: Dict[str, PaperMetadata] = {}
    best_list: List[PaperMetadata] = []

    seen_queries = set()

    size = max(5, min(int(per_page or 10), 25))

    def process_query(q: str) -> None:
        if (time.monotonic() - started) > SEARCH_TIME_BUDGET_SEC:
            return
        if q in seen_queries:
            return
        seen_queries.add(q)

        payload = {"mode": "articles", "q": q, "size": size, "from": 0}
        items = _cyberleninka_search_api_cached(json.dumps(payload, sort_keys=True, ensure_ascii=False))
        if not items:
            return

        q_norm = _normalize_text_for_compare(q)

        for it in items:
            if not isinstance(it, dict):
                continue
            raw_title = (it.get("name") or "").strip()
            plain = _strip_light_html_to_text(raw_title)
            if len(plain) < 8:
                continue

            title_norm = _normalize_text_for_compare(plain)
            sim_o = SequenceMatcher(None, original_norm, title_norm).ratio()
            sim_q = SequenceMatcher(None, q_norm, title_norm).ratio()
            sim = max(sim_o, sim_q)
            if sim < 0.34:
                continue

            paper = _paper_from_cyberleninka_hit(it, matched_query=q, title_sim=sim)
            if paper is None or not paper.source_url:
                continue

            key = (paper.source_url or "").split("?", 1)[0].lower().rstrip("/")
            existing = best_by_url.get(key)
            if existing is None or (paper.search_score or 0.0) > (existing.search_score or 0.0):
                best_by_url[key] = paper

    for q in query_variants[:3]:
        process_query(q)

    best_list = list(best_by_url.values())
    best_list.sort(key=lambda x: x.search_score or 0.0, reverse=True)
    return best_list[:25]


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


def doi_syntax_plausible(doi: str) -> bool:
    """Проверка формы DOI (частые ошибки OCR / опечатки в PDF)."""
    d = normalize_doi(doi or "").strip()
    return bool(DOI_STRICT_INPUT_RE.match(d))


def openalex_retraction_hint(doi: str) -> Optional[bool]:
    """OpenAlex помечает retracted; офлайн/ошибка API → None."""
    nd = normalize_doi(doi or "").strip()
    if not nd:
        return None
    url = f"https://api.openalex.org/works/https://doi.org/{quote(nd, safe='')}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json() or {}
        return bool(data.get("is_retracted"))
    except Exception:
        return None


def _attach_quality_hints(meta: PaperMetadata) -> None:
    """Подмешивает в meta.confidence метку рестракции (float 1.0) при наличии DOI."""
    if not meta or not meta.doi:
        return
    retracted = openalex_retraction_hint(meta.doi)
    if retracted is True:
        meta.confidence = meta.confidence or {}
        meta.confidence["retracted_openalex"] = 1.0


def _normalize_text_for_compare(text: str) -> str:
    text = (text or "").strip().lower().replace("ё", "е")
    text = NON_WORD_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def similarity_normalized_titles(user_title: str, meta_title: Optional[str]) -> float:
    """Сходство строк заголовков после нормализации (для сопоставления с вводом пользователя)."""
    a = _normalize_text_for_compare(user_title or "")
    b = _normalize_text_for_compare(meta_title or "")
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def explain_candidate_ranking(
    user_title: Optional[str],
    user_doi: Optional[str],
    item: PaperMetadata,
) -> str:
    """
    Пояснение ранжирования варианта: проценты по доступным сигналам (заголовок, авторы, год, DOI).
    """
    signals: List[str] = []
    blend: List[float] = []

    ut = (user_title or "").strip()
    ud = normalize_doi(user_doi) if user_doi else None
    if not ud:
        ud = None

    if ut and (item.title or "").strip():
        sim = similarity_normalized_titles(ut, item.title)
        pct = sim * 100.0
        signals.append(f"совпадение по названию ≈ {pct:.0f}%")
        blend.append(pct)

    if ut and item.authors:
        toks = re.findall(r"[A-Za-zА-Яа-яЁё]{3,}", ut)
        auth_join = " ".join(item.authors).lower()
        hits = sum(1 for t in toks[:8] if t.lower() in auth_join)
        if toks:
            ap = min(100.0, (hits / max(min(len(toks), 8), 1)) * 100.0)
            if hits:
                signals.append(f"совпадение по авторам ≈ {ap:.0f}%")
                blend.append(ap)

    if ut and item.year:
        ym = re.search(r"\b(19\d{2}|20\d{2})\b", ut)
        if ym:
            ok = ym.group(1) == str(item.year).strip()
            yp = 100.0 if ok else 0.0
            signals.append(f"совпадение по году ≈ {yp:.0f}%")
            blend.append(yp)

    if ud and item.doi and normalize_doi(item.doi) == ud:
        mconf = item.confidence or {}
        if float(mconf.get("doi_invalid") or 0) >= 1.0:
            signals.append(
                "DOI: строка совпадает с единственной записью, но формат не прошёл проверку — "
                "метаданные из Crossref, скорее всего, не получены; «100%» здесь не про валидность идентификатора"
            )
            blend.append(0.0)
        else:
            signals.append("совпадение по DOI 100%")
            blend.append(100.0)

    if item.search_score is not None:
        signals.append(f"внутренняя оценка каталога search_score={item.search_score:.3f}")

    if item.matched_query and item.matched_query != ut:
        mq = item.matched_query if len(item.matched_query) <= 90 else item.matched_query[:87] + "…"
        signals.append(f"использован вариант запроса «{mq}»")

    summary = ""
    if blend:
        summary = f" Сводная оценка по числовым признакам ≈ {sum(blend) / len(blend):.0f}%."

    if not signals:
        return "Ранжирование по скору каталога (CrossRef и др.) и правилам слияния дубликатов."

    return "Объяснение (explain): " + "; ".join(signals) + "." + summary


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

def is_journal_rubric_list_url(url: str) -> bool:
    """
    Эвристика: страница списка статей (рубрика/выпуск), а не страница одной статьи.
    Для таких URL поиск по полю «название» должен обрабатываться отдельно (см. resolve_article_from_journal_listing).
    """
    try:
        p = urlparse(url or "")
        host = (p.netloc or "").lower()
        path = (p.path or "").lower()
    except Exception:
        return False
    if "mmi.sgu.ru" in host and "rubrika" in path:
        return True
    if "mmi.sgu.ru" in host and "/ru/issue" in path:
        return True
    return False


def is_cyberleninka_non_article_page(url: str) -> bool:
    """
    True для главной CyberLeninka, журналов, поиска и т.п. — всё, что не карточка статьи /article/n/....

    Если пользователь указывает такой URL вместе с названием, парсить HTML страницы бессмысленно:
    на главной нет метаданных статьи (попадает рекламный текст из <title>).
    """
    try:
        p = urlparse((url or "").strip())
        host = (p.netloc or "").lower()
    except Exception:
        return False
    if "cyberleninka.ru" not in host:
        return False
    path = (p.path or "").strip().lower()
    if path.startswith("/article/n/"):
        return False
    return True


_JUNK_ANCHOR_RE = re.compile(
    r"^(read\s*more|далее|подробнее|читать(\s+далее)?|ещё|еще|more|next|prev|previous|назад|войти|регистрац|"
    r"все\s+новости|архив|cookie|политика\s+конфиденциальности)\b",
    re.IGNORECASE,
)

_BAD_LISTING_PATH_PREFIXES = (
    "/wp-admin",
    "/wp-login",
    "/login",
    "/register",
    "/cart",
    "/checkout",
    "/search",
    "/tag/",
    "/tags/",
    "/category/",
    "/author/",
    "/static/",
    "/assets/",
    "/images/",
    "/img/",
    "/fonts/",
    "/ajax/",
    "/api/",
)


def _netloc_key(netloc: str) -> str:
    n = (netloc or "").lower()
    if n.startswith("www."):
        n = n[4:]
    return n


def _link_in_boilerplate_nav(a) -> bool:
    for parent in a.parents:
        name = getattr(parent, "name", None)
        if name in {"nav", "footer", "header", "aside"}:
            return True
        try:
            if parent.get("role") == "navigation":
                return True
        except Exception:
            pass
    return False


def _is_junk_anchor_label(label: str, sim: float) -> bool:
    t = (label or "").strip()
    if len(t) < 2:
        return True
    low = t.lower()
    if len(t) < 12 and sim < 0.72 and low in {"далее", "ещё", "еще", "more", "next", "prev", "назад"}:
        return True
    return _JUNK_ANCHOR_RE.match(low) is not None


def _path_looks_like_article_url(path: str, netloc: str) -> bool:
    pl = (path or "").lower()
    nl = (netloc or "").lower()
    if nl.endswith("doi.org") and "/10." in pl:
        return True
    markers = (
        "/article/",
        "/articles/",
        "/paper/",
        "/papers/",
        "/doi/",
        "/publication",
        "/publikats",
        "/fulltext",
        "/document",
        "/docs/",
        "/content/",
        "/science/article",
        "/chapter/",
        "/abs/",
        "/html/",
        "/meta/",
        "/reader/",
        "/view/",
    )
    return any(m in pl for m in markers)


def _path_allowed_for_general_link(path: str) -> bool:
    pl = (path or "").lower()
    if not pl or pl == "/":
        return False
    for pfx in _BAD_LISTING_PATH_PREFIXES:
        if pl.startswith(pfx):
            return False
    segments = [s for s in pl.split("/") if s]
    return len(segments) >= 2


def find_article_url_on_listing_page(
    html: str,
    page_url: str,
    title_query: str,
    *,
    min_similarity: float = 0.52,
) -> Optional[str]:
    """
    Ищет на HTML-странице ссылку, текст/заголовок которой ближе всего к title_query.

    Два уровня: «строгий» (типичные пути статей /doi/, /article/, …) и «общий» (любая
    содержательная ссылка на том же сайте с длинным якорем) — чтобы работать не только
    с заранее известными издателями.
    """
    q = (title_query or "").strip()
    if len(q) < 6:
        return None
    q_norm = _normalize_text_for_compare(q)
    try:
        page_host_key = _netloc_key(urlparse(page_url).netloc)
    except Exception:
        page_host_key = ""

    soup = BeautifulSoup(html, "html.parser")
    strict_min = max(0.48, float(min_similarity) - 0.02)
    general_min = max(0.54, float(min_similarity) + 0.02)

    best_strict = {"sim": -1.0, "len": -1, "url": ""}
    best_general = {"sim": -1.0, "len": -1, "url": ""}

    def better(rec: dict, sim: float, alen: int) -> bool:
        if sim > rec["sim"]:
            return True
        if sim == rec["sim"] and alen > rec["len"]:
            return True
        return False

    for a in soup.find_all("a", href=True):
        if _link_in_boilerplate_nav(a):
            continue
        href = (a.get("href") or "").strip()
        if not href or href.startswith("#") or href.lower().startswith("javascript:"):
            continue
        abs_url = urljoin(page_url, href)
        if not is_public_http_url(abs_url):
            continue
        try:
            pu = urlparse(abs_url)
            cand_host_key = _netloc_key(pu.netloc)
            path = pu.path or ""
        except Exception:
            continue

        if cand_host_key != page_host_key:
            nl = (pu.netloc or "").lower()
            if nl not in ("doi.org", "www.doi.org"):
                continue

        text = a.get_text(" ", strip=True) or ""
        title_attr = (a.get("title") or "").strip()
        t_norm = _normalize_text_for_compare(text)
        ta_norm = _normalize_text_for_compare(title_attr)
        sim_text = SequenceMatcher(None, q_norm, t_norm).ratio() if t_norm else 0.0
        sim_attr = SequenceMatcher(None, q_norm, ta_norm).ratio() if ta_norm else 0.0
        sim = max(sim_text, sim_attr)
        label = text if len(text) >= len(title_attr) else title_attr
        if _is_junk_anchor_label(label, sim):
            continue
        if len(text) < 6 and len(title_attr) < 6:
            continue

        anchor_len = max(len(text), len(title_attr))
        path_article = _path_looks_like_article_url(path, pu.netloc or "")

        if path_article and sim >= strict_min:
            if better(best_strict, sim, anchor_len):
                best_strict = {"sim": sim, "len": anchor_len, "url": abs_url}
        if anchor_len >= 10 and sim >= general_min and _path_allowed_for_general_link(path):
            if better(best_general, sim, anchor_len):
                best_general = {"sim": sim, "len": anchor_len, "url": abs_url}

    chosen = ""
    if best_strict["url"]:
        chosen = best_strict["url"]
    elif best_general["url"]:
        chosen = best_general["url"]

    if not chosen:
        return None
    base_listing = (page_url or "").split("#")[0].rstrip("/").lower()
    base_pick = chosen.split("#")[0].rstrip("/").lower()
    if base_pick == base_listing:
        return None
    return chosen


def resolve_article_from_listing_by_title(listing_url: str, title_query: str) -> Optional[PaperMetadata]:
    """
    Универсально: загружает страницу (рубрика, выпуск, главная раздела и т.д.), ищет ссылку
    по близости текста к title_query, затем извлекает метаданные как для обычного URL статьи.
    """
    listing_url = (listing_url or "").strip()
    title_query = (title_query or "").strip()
    if not listing_url or not title_query or len(title_query) < 6:
        return None
    try:
        html = fetch_html(listing_url)
    except Exception:
        return None
    article_url = find_article_url_on_listing_page(html, listing_url, title_query)
    if not article_url:
        return None
    if not is_public_http_url(article_url):
        return None
    return extract_metadata_from_url(article_url)


def resolve_article_from_journal_listing(listing_url: str, title_query: str) -> Optional[PaperMetadata]:
    """
    Загружает страницу рубрики/списка, находит ссылку на статью по названию и извлекает метаданные
    как при обычном URL статьи (в т.ч. DOI → Crossref).
    """
    return resolve_article_from_listing_by_title(listing_url, title_query)


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


def _openlibrary_doc_title(doc: Dict[str, Any]) -> str:
    t = doc.get("title")
    if isinstance(t, list) and t:
        return str(t[0] or "").strip()
    return str(t or "").strip()


def _openlibrary_publish_year(doc: Dict[str, Any]) -> Optional[int]:
    for key in ("first_publish_year", "publish_year"):
        y = doc.get(key)
        if y is None:
            continue
        try:
            return int(y)
        except (TypeError, ValueError):
            continue
    return None


_SKIP_TITLE_WORDS = frozenset(
    {"the", "a", "an", "on", "in", "of", "and", "for", "to", "at", "by", "key", "from"}
)


def _first_significant_title_keyword(title: str) -> Optional[str]:
    """Первое значимое слово заголовка (для короткого запроса author+title в Open Library)."""
    for raw in re.split(r"[^\w]+", title):
        w = raw.strip()
        if len(w) >= 4 and w.lower() not in _SKIP_TITLE_WORDS:
            return w[:80]
    return None


def _extract_publisher_hint_from_reference_line(cite_line: str) -> Optional[str]:
    """Хвост строки цитаты «... Title. Publisher, City.» — издательство до первой запятой."""
    s = (cite_line or "").strip().rstrip(".")
    if not s:
        return None
    chunks = re.split(r"\.\s+", s)
    if len(chunks) < 2:
        return None
    tail = chunks[-1].strip()
    if "," not in tail:
        return None
    left = tail.split(",")[0].strip()
    if len(left) < 3:
        return None
    if re.fullmatch(r"\d{4}", left):
        return None
    return left[:220]


def _edition_publishers_flat(ed: Dict[str, Any]) -> List[str]:
    p = ed.get("publishers")
    if isinstance(p, list):
        return [str(x).strip() for x in p if str(x).strip()]
    if isinstance(p, str) and p.strip():
        return [p.strip()]
    return []


def _normalize_publisher_for_match(s: str) -> str:
    s = re.sub(r"[-_/]+", " ", (s or "").lower())
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def _publisher_hint_fit_score(pub_line: str, hint: str) -> float:
    """Сходство строки издателя из каталога с подсказкой из цитаты (подстрока / токены / fuzzy)."""
    h = _normalize_publisher_for_match(hint)
    p = _normalize_publisher_for_match(pub_line)
    if not h or not p:
        return 0.0
    if h in p or p in h:
        return 1.0
    th = [t for t in h.split() if len(t) >= 3]
    pw = p.split()
    set_p = set(t for t in pw if len(t) >= 3)
    if th and set_p:
        inter = len(set(th) & set_p)
        uni = len(set(th) | set_p) or 1
        jacc = inter / uni
    else:
        jacc = 0.0
    return max(jacc, SequenceMatcher(None, h, p).ratio())


def _openlibrary_edition_supplement(
    work_key: str,
    year_hint: Optional[str],
    publisher_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """Подтягивает издательство, страницы и ISBN из /works/.../editions (индекс search часто их не содержит)."""
    out: Dict[str, Any] = {}
    if not isinstance(work_key, str) or not work_key.startswith("/works/"):
        return out
    try:
        r = requests.get(
            f"https://openlibrary.org{work_key}/editions.json",
            timeout=API_TIMEOUTS,
            headers={"User-Agent": "BiblioParser/1.0 (openlibrary.org editions)"},
        )
        if not r.ok:
            return out
        entries = r.json().get("entries") or []
    except Exception as ex:
        LOGGER.debug("openlibrary editions: %s", ex)
        return out

    if not entries:
        return out

    def _edition_year(ed: Dict[str, Any]) -> Optional[int]:
        pd = ed.get("publish_date")
        if pd:
            m = re.search(r"(19|20)\d{2}", str(pd))
            if m:
                return int(m.group(0))
        return None

    ywant = int(year_hint) if year_hint and year_hint.isdigit() else None
    ph = (publisher_hint or "").strip()

    def _edition_rank_score(ed: Dict[str, Any]) -> float:
        ey = _edition_year(ed)
        y_pri = 0.0
        if ywant is not None and ey is not None:
            if ey == ywant:
                y_pri = 100.0
            elif abs(ey - ywant) <= 1:
                y_pri = 72.0
            elif abs(ey - ywant) <= 2:
                y_pri = 45.0
            else:
                y_pri = max(0.0, 20.0 - float(abs(ey - ywant)))
        elif ey is not None:
            y_pri = 8.0
        pub_sc = 0.0
        if ph:
            for pl in _edition_publishers_flat(ed):
                pub_sc = max(pub_sc, _publisher_hint_fit_score(pl, ph))
        return y_pri + pub_sc * 55.0

    chosen = max(entries, key=_edition_rank_score)
    best_pub_fit = 0.0
    if ph:
        for pl in _edition_publishers_flat(chosen):
            best_pub_fit = max(best_pub_fit, _publisher_hint_fit_score(pl, ph))

    pubs = chosen.get("publishers")
    if isinstance(pubs, list) and pubs:
        out["publisher"] = str(pubs[0]).strip()
    elif isinstance(pubs, str) and pubs.strip():
        out["publisher"] = pubs.strip()

    np = chosen.get("number_of_pages")
    if np is not None:
        try:
            out["pages"] = str(int(np))
        except (TypeError, ValueError):
            out["pages"] = str(np)

    i13, i10 = chosen.get("isbn_13"), chosen.get("isbn_10")
    if isinstance(i13, list) and i13:
        out["isbn"] = str(i13[0])
    elif isinstance(i13, str) and i13.strip():
        out["isbn"] = i13.strip()
    elif isinstance(i10, list) and i10:
        out["isbn"] = str(i10[0])
    elif isinstance(i10, str) and i10.strip():
        out["isbn"] = i10.strip()

    if ph:
        out["publisher_match_score"] = round(best_pub_fit, 4)
        out["publisher_hint_used"] = ph[:220]

    return out


def search_openlibrary_docs(title: str, author: Optional[str], year: Optional[str]) -> List[Dict[str, Any]]:
    """Поиск в каталоге Open Library (search.json), без API-ключа."""
    title = (title or "").strip()
    author = (author or "").strip()
    if not title and not author:
        return []

    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "BiblioParser/1.0 (metadata enrichment; openlibrary.org)",
            "Accept": "application/json",
        }
    )
    docs: List[Dict[str, Any]] = []

    def _merge(new_docs: List[Dict[str, Any]]) -> None:
        seen = {str(d.get("key") or id(d)) for d in docs}
        for d in new_docs:
            sig = str(d.get("key") or id(d))
            if sig not in seen:
                seen.add(sig)
                docs.append(d)

    try:
        # Короткий запрос (фамилия + первое значимое слово заголовка) часто лучше длинной строки с агрегатора.
        if title and author:
            kw = _first_significant_title_keyword(title)
            if kw:
                r = s.get(
                    "https://openlibrary.org/search.json",
                    params={"limit": "22", "title": kw, "author": author[:120]},
                    timeout=API_TIMEOUTS,
                )
                if r.ok:
                    _merge(r.json().get("docs") or [])
        if title:
            params: Dict[str, str] = {"limit": "22", "title": title[:400]}
            if author:
                params["author"] = author[:120]
            r = s.get("https://openlibrary.org/search.json", params=params, timeout=API_TIMEOUTS)
            if r.ok:
                _merge(r.json().get("docs") or [])
    except Exception as ex:
        LOGGER.debug("openlibrary title/author: %s", ex)

    try:
        q = " ".join(x for x in (title, author, year or "") if x).strip()
        if q:
            r = s.get(
                "https://openlibrary.org/search.json",
                params={"q": q[:500], "limit": "22"},
                timeout=API_TIMEOUTS,
            )
            if r.ok:
                _merge(r.json().get("docs") or [])
    except Exception as ex:
        LOGGER.debug("openlibrary q=: %s", ex)

    return docs


def pick_best_openlibrary_doc(
    docs: List[Dict[str, Any]],
    title_hint: str,
    year_hint: Optional[str],
    author_surname: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not docs or not (title_hint or "").strip():
        return None
    nt = _normalize_text_for_compare(title_hint)
    best_doc: Optional[Dict[str, Any]] = None
    best_score = 0.0
    ywant: Optional[int] = None
    if year_hint and year_hint.isdigit():
        ywant = int(year_hint)

    for doc in docs:
        ol_title = _openlibrary_doc_title(doc)
        if not ol_title:
            continue
        sim = SequenceMatcher(None, nt, _normalize_text_for_compare(ol_title)).ratio()
        score = sim
        py = _openlibrary_publish_year(doc)
        if ywant is not None and py is not None:
            if py == ywant:
                score += 0.48
            elif abs(py - ywant) <= 1:
                score += 0.12
        if author_surname:
            blob = " ".join(doc.get("author_name") or []).lower()
            if author_surname.lower() in blob:
                score += 0.18
        if score > best_score:
            best_score = score
            best_doc = doc

    if best_doc is None:
        return None
    if best_score < 0.52:
        return None
    ol_title = _openlibrary_doc_title(best_doc)
    sim_only = SequenceMatcher(None, nt, _normalize_text_for_compare(ol_title)).ratio()
    py = _openlibrary_publish_year(best_doc)
    year_ok = ywant is not None and py is not None and py == ywant
    if sim_only < 0.36 and not year_ok:
        return None
    return best_doc


def merge_openlibrary_into_metadata(
    doc: Dict[str, Any],
    meta: PaperMetadata,
    publisher_hint: Optional[str] = None,
) -> PaperMetadata:
    """Подставляет издательство, объём, при необходимости уточняет заголовок и автора."""
    ol_title = _openlibrary_doc_title(doc)
    if ol_title:
        meta.title = ol_title

    pub = doc.get("publisher")
    if isinstance(pub, list) and pub:
        meta.journal = str(pub[0]).strip()
    elif isinstance(pub, str) and pub.strip():
        meta.journal = pub.strip()

    npages = doc.get("number_of_pages_median")
    if npages is None:
        npages = doc.get("number_of_pages")
    if isinstance(npages, list) and npages:
        npages = npages[0]
    if npages is not None:
        try:
            meta.pages = str(int(npages))
        except (TypeError, ValueError):
            meta.pages = str(npages)

    an = doc.get("author_name")
    if isinstance(an, list) and an:
        meta.authors = [str(x).strip() for x in an[:12] if str(x).strip()]

    meta.confidence = meta.confidence or {}
    isbn_l = doc.get("isbn")
    if isinstance(isbn_l, list) and isbn_l:
        meta.confidence["isbn"] = str(isbn_l[0])
    wkey = doc.get("key")
    if isinstance(wkey, str) and wkey.startswith("/"):
        meta.confidence["openlibrary_url"] = f"https://openlibrary.org{wkey}"
        sup = _openlibrary_edition_supplement(wkey, meta.year, publisher_hint)
        if sup.get("publisher"):
            meta.journal = sup["publisher"]
        if sup.get("pages"):
            meta.pages = sup["pages"]
        if sup.get("isbn"):
            meta.confidence["isbn"] = sup["isbn"]
        if sup.get("publisher_match_score") is not None:
            meta.confidence["openlibrary_publisher_match_score"] = float(sup["publisher_match_score"])
        if sup.get("publisher_hint_used"):
            meta.confidence["reference_publisher_hint"] = str(sup["publisher_hint_used"])[:220]

    eb = meta.enriched_by or ""
    if "openlibrary" not in eb:
        meta.enriched_by = f"{eb}+openlibrary" if eb else "openlibrary"
    return meta


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

    if style == "journal_auto":
        jl = (m.journal or "").lower()
        if "nature" in jl:
            return format_citation(m, "nature")
        if any(x in jl for x in ("springer", "bmc", "biomed central", "frontiers")):
            return format_citation(m, "springer")
        return format_citation(m, "gost")

    if style == "springer":
        au = _format_authors(m.authors, "apa")
        title = m.title or "Untitled"
        j = m.journal or ""
        y = m.year or ""
        vol = m.volume or ""
        iss = m.issue or ""
        pg = m.pages or ""
        voliss = ""
        if vol or iss or pg:
            voliss = (vol or "") + (f"({iss})" if iss else "") + (f":{pg}" if pg else "")
        doi = f"https://doi.org/{m.doi}" if m.doi else ""
        left = f"{au} {title}." if au else f"{title}."
        mid = f" {j} " if j else " "
        tail = f" {y} {voliss} {doi}".replace("  ", " ").strip()
        return (left + mid + tail).strip()

    if style == "nature":
        au = _format_authors(m.authors, "ieee") or "Authors"
        title = m.title or "Untitled"
        j = m.journal or ""
        vol = m.volume or ""
        pg = m.pages or ""
        yr = m.year or ""
        bits = [f"{au}.", title + "."]
        if j:
            bits.append(j + ".")
        tail_parts = []
        if vol:
            tail_parts.append(vol)
        if pg:
            tail_parts.append(pg)
        if yr:
            tail_parts.append(f"({yr})")
        if tail_parts:
            bits.append(" ".join(tail_parts) + ".")
        doi = f"doi:{m.doi}" if m.doi else ""
        if doi:
            bits.append(doi)
        return " ".join(x for x in bits if x)

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


def _extract_reference_listing_primary_citation(html: str) -> Optional[str]:
    """Строка вида «Author (YYYY) Title…» на страницах списков цитирования (SCIRP и др.)."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for meta in soup.find_all("meta"):
        prop = (meta.get("property") or meta.get("name") or "").lower()
        if prop in ("og:description", "description"):
            c = (meta.get("content") or "").strip()
            if c and _BOOKISH_BANNER_YEAR_RE.search(c) and len(c) >= 25:
                return c[:900]
    text = soup.get_text("\n", strip=True)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    skip_frag = (
        "copyright",
        "follow scirp",
        "contact us",
        "login",
        "journals a-z",
        "select journal",
        "scientific research publishing",
    )
    for ln in lines:
        low = ln.lower()
        if any(f in low for f in skip_frag) and len(ln) < 100:
            continue
        if _BOOKISH_BANNER_YEAR_RE.search(ln) and len(ln) >= 28:
            if re.match(r"^[A-Za-zА-Яа-яЁё0-9]", ln):
                return ln[:900]
    return None


def _strip_publisher_tail_from_citation_line(line: str) -> str:
    """Убирает хвост «. Издательство, город» — он часто сбивает поиск в Crossref."""
    s = (line or "").strip()
    if not s:
        return s
    cut_patterns = (
        r"\.\s*McGraw-Hill\b",
        r",\s*McGraw-Hill\b",
        r"\.\s*Springer\b",
        r",\s*Springer\b",
        r"\.\s*John Wiley\b",
        r",\s*John Wiley\b",
    )
    for pat in cut_patterns:
        m = re.search(pat, s, re.I)
        if m:
            s = s[: m.start()].rstrip("., ")
            break
    return s


def _parse_paren_year_book_citation(line: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Разбор строки «Author, I. (YYYY) Book title…» → фамилия, год, заголовок."""
    s = (line or "").strip()
    if not s:
        return None, None, None
    m = re.match(
        r"^\s*([^(\n]+?)\s*\(\s*((?:19|20)\d{2})\s*\)\s*(.+)$",
        s,
    )
    if not m:
        return None, None, None
    author_chunk, year, title_rest = m.group(1).strip(), m.group(2), m.group(3).strip()
    surname = author_chunk.split(",")[0].strip() or None
    title_rest = _strip_publisher_tail_from_citation_line(title_rest)
    title_rest = title_rest.rstrip(". ")
    return surname, year, title_rest or None


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

    # DOI в разметке страницы часто есть даже когда эвристика «не статья» (Drupal/журналы вроде mmi.sgu.ru).
    doi_in_html = extract_doi_rule_based(html)
    if doi_in_html:
        doi_n = normalize_doi(doi_in_html)
        enriched_early = crossref_enrich(doi_n)
        if enriched_early:
            enriched_early.source_url = url
            enriched_early.confidence = enriched_early.confidence or {}
            enriched_early.confidence.update(meta.confidence or {})
            enriched_early.pdf_url = enriched_early.pdf_url or extract_pdf_url_from_html(html, source_url=url)
            return enriched_early

    # Карточка вторичной ссылки без DOI (напр. SCIRP «references»): видимая цитата → Crossref bibliographic search.
    path_low = (urlparse(url).path or "").lower()
    if "scirp.org" in host and ("reference" in path_low or "referencespapers" in path_low):
        cite_line = _extract_reference_listing_primary_citation(html)
        if cite_line:
            surname, year_hint, title_only = _parse_paren_year_book_citation(cite_line)
            if title_only:
                cite_query = f"{surname} {title_only}".strip() if surname else title_only
            else:
                cite_query = _strip_publisher_tail_from_citation_line(cite_line)
            if not year_hint:
                ym = re.search(r"\(\s*((?:19|20)\d{2})\s*\)", cite_line)
                year_hint = ym.group(1) if ym else None
            cands = search_crossref_candidates_relaxed(cite_query, rows=18)
            if year_hint:
                same_year = [c for c in cands if (c.year or "").strip() == year_hint]
                if same_year:
                    cands = same_year + [c for c in cands if c not in same_year]
                else:
                    # Нет совпадения по году (типично для старых монографий) — чужой DOI хуже, чем «только разбор строки».
                    cands = []
            if cands:
                best = cands[0]
                best.source_url = url
                best.confidence = best.confidence or {}
                best.confidence.update(meta.confidence or {})
                best.confidence["enriched_by"] = "crossref_from_reference_listing"
                best.confidence["reference_listing_query"] = cite_query[:240]
                best.pdf_url = best.pdf_url or extract_pdf_url_from_html(html, source_url=url)
                return best
            # Запись без DOI: строка цитаты + при возможности обогащение из Open Library (издательство, стр., ISBN).
            if title_only or cite_line:
                fallback = PaperMetadata(
                    title=title_only or cite_query[:500],
                    authors=[surname] if surname else None,
                    year=year_hint,
                    source_url=url,
                    confidence=dict(meta.confidence or {}),
                    enriched_by="reference_page_parse_only",
                )
                fb = fallback.confidence or {}
                fb["reference_banner_raw"] = cite_line[:900] if cite_line else ""
                ol_doc = None
                if title_only:
                    ol_docs = search_openlibrary_docs(title_only, surname or "", year_hint or "")
                    ol_doc = pick_best_openlibrary_doc(
                        ol_docs, title_only, year_hint, surname
                    )
                if ol_doc:
                    pub_hint = (
                        _extract_publisher_hint_from_reference_line(cite_line) if cite_line else None
                    )
                    merge_openlibrary_into_metadata(ol_doc, fallback, publisher_hint=pub_hint)
                    fb = fallback.confidence or {}
                    note_ol = (
                        "В Crossref нет однозначной записи с этим годом; издательство, объём и ISBN уточнены по Open Library. "
                        "Сверьте с нужным изданием (год, город, переплёт)."
                    )
                    mscore = fb.get("openlibrary_publisher_match_score")
                    if (
                        pub_hint
                        and mscore is not None
                        and float(mscore) < 0.5
                    ):
                        note_ol += (
                            " В Open Library для этой работы нет издания с близким к цитате издателем — "
                            "показана запись с наилучшим сочетанием года и издателя в каталоге."
                        )
                    fb["reference_note"] = note_ol
                else:
                    fb["reference_note"] = (
                        "Нет однозначного совпадения в Crossref по году/названию; "
                        "проверьте издательство и число страниц (Open Library, WorldCat, каталог издателя)."
                    )
                fallback.confidence = fb
                return fallback

    # Предклассификатор страницы: если это явно не статья и домен не из приоритетных, не запускаем тяжёлый ML.
    priority_hosts = (
        "arxiv.org",
        "ieeexplore.ieee.org",
        "sciencedirect.com",
        "link.springer.com",
        "cyberleninka.ru",
        "scirp.org",
    )
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
    base: List[PaperMetadata] = list(candidates or [])
    if not base:
        relaxed = search_crossref_candidates_relaxed(title, rows=10)
        base = list(relaxed or [])

    # Russian / Cyrillic titles: CyberLeninka JSON search + OpenAlex (CrossRef часто бессилен)
    if query_has_cyrillic(title):
        cl = search_cyberleninka_candidates_by_title(title, per_page=25)
        oa = search_openalex_candidates_by_title(title, per_page=25)

        seen = set()
        merged: List[PaperMetadata] = []

        def key_for(p: PaperMetadata) -> str:
            if p.doi:
                return f"doi::{p.doi}"
            su = (p.source_url or "").strip().lower().rstrip("/")
            if su:
                host = (urlparse(su).netloc or "").lower()
                if host.endswith("cyberleninka.ru"):
                    return f"url::{su}"
            return f"title::{_normalize_text_for_compare(p.title or '')}"

        for item in base:
            k = key_for(item)
            if k in seen:
                continue
            seen.add(k)
            merged.append(item)

        original_norm = _normalize_text_for_compare(title)

        for item in cl:
            k = key_for(item)
            if k in seen:
                continue

            title_norm = _normalize_text_for_compare(item.title or "")
            sim = SequenceMatcher(None, original_norm, title_norm).ratio()

            landing = (item.source_url or "").lower()
            prefer = ("cyberleninka.ru" in landing) and (sim >= 0.58)
            if not prefer:
                continue

            seen.add(k)
            merged.append(item)

        for item in oa:
            k = key_for(item)
            if k in seen:
                continue

            title_norm = _normalize_text_for_compare(item.title or "")
            sim = SequenceMatcher(None, original_norm, title_norm).ratio()

            landing = (item.source_url or "").lower()
            prefer = ("cyberleninka.ru" in landing) or (sim >= 0.72)
            if not prefer:
                continue

            seen.add(k)
            merged.append(item)

        merged.sort(key=lambda x: x.search_score or 0.0, reverse=True)

        out = merged[:max_results]
        if out:
            return out

    if base:
        return base[:max_results]
    return [PaperMetadata(title=title, source_url=None, confidence={})]


def extract_metadata_from_title(title: str) -> PaperMetadata:
    candidates = search_metadata_candidates_from_title(title, max_results=1)
    if candidates:
        return candidates[0]
    return PaperMetadata(title=(title or "").strip(), source_url=None, confidence={})

def extract_metadata_from_doi(doi: str) -> PaperMetadata:
    nd = normalize_doi(doi or "")
    if not nd.strip():
        return PaperMetadata(
            doi=(doi or "").strip() or None,
            source_url=None,
            enriched_by=None,
            confidence={"doi_invalid": 1.0},
        )

    if not doi_syntax_plausible(nd):
        return PaperMetadata(
            doi=nd,
            source_url=f"https://doi.org/{nd}",
            enriched_by="doi_validation",
            confidence={"doi_invalid": 1.0},
        )

    meta = crossref_enrich(nd)
    if meta:
        meta.source_url = f"https://doi.org/{nd}"
        meta.confidence = meta.confidence or {}
        _attach_quality_hints(meta)
        return meta

    meta2 = datacite_enrich(nd)
    if meta2:
        meta2.source_url = f"https://doi.org/{nd}"
        meta2.confidence = meta2.confidence or {}
        _attach_quality_hints(meta2)
        return meta2

    out = PaperMetadata(doi=nd, source_url=f"https://doi.org/{nd}", enriched_by=None, confidence={})
    _attach_quality_hints(out)
    return out

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

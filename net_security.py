import ipaddress
import os
import socket
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests


BLOCKED_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _is_ip_private_or_local(ip_str: str) -> bool:
    try:
        ip_obj = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    return bool(
        ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_multicast
        or ip_obj.is_reserved
        or ip_obj.is_unspecified
    )


def is_public_http_url(url: str) -> bool:
    """
    Basic SSRF guard:
    - only http/https
    - no credentials in URL
    - host must resolve and all resolved IPs must be public
    """
    try:
        parsed = urlparse((url or "").strip())
    except Exception:
        return False

    if parsed.scheme not in {"http", "https"}:
        return False
    if not parsed.netloc:
        return False
    if parsed.username or parsed.password:
        return False

    host = (parsed.hostname or "").strip().lower()
    if not host or host in BLOCKED_HOSTS:
        return False
    if host.endswith(".local"):
        return False

    # Onion-адреса: DNS вне Tor не резолвится. Разрешаем только явно (исследовательские сценарии через Tor).
    if host.endswith(".onion"):
        return os.getenv("ALLOW_ONION", "").strip().lower() in ("1", "true", "yes", "on")

    try:
        infos = socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False

    if not infos:
        return False

    resolved_ips = {info[4][0] for info in infos if info and info[4]}
    if not resolved_ips:
        return False

    for ip_str in resolved_ips:
        if _is_ip_private_or_local(ip_str):
            return False
    return True


def explain_public_http_url(url: str) -> Dict[str, Any]:
    """
    Детальный разбор проверки SSRF для отображения в UI и API /api/url_check.
    Поля совместимы с логикой is_public_http_url (без сетевых HEAD-запросов).
    """
    raw = (url or "").strip()
    out: Dict[str, Any] = {
        "ok": False,
        "scheme_ok": False,
        "host_present": False,
        "no_credentials": False,
        "host_allowed": False,
        "dns_ok": False,
        "ips_public": False,
        "resolved_ips": [],
        "reason": "",
    }
    try:
        parsed = urlparse(raw)
    except Exception:
        out["reason"] = "parse_error"
        return out

    if parsed.scheme not in {"http", "https"}:
        out["reason"] = "bad_scheme"
        return out
    out["scheme_ok"] = True

    if not parsed.netloc:
        out["reason"] = "no_netloc"
        return out
    out["host_present"] = True

    if parsed.username or parsed.password:
        out["reason"] = "credentials_in_url"
        return out
    out["no_credentials"] = True

    host = (parsed.hostname or "").strip().lower()
    if not host:
        out["reason"] = "empty_host"
        return out
    if host in BLOCKED_HOSTS:
        out["reason"] = "blocked_host_literal"
        return out
    if host.endswith(".local"):
        out["reason"] = "mdns_local_domain"
        return out
    if host.endswith(".onion"):
        allowed = os.getenv("ALLOW_ONION", "").strip().lower() in ("1", "true", "yes", "on")
        if not allowed:
            out["reason"] = "onion_requires_allow_onion"
            return out
    out["host_allowed"] = True

    try:
        infos = socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)
    except socket.gaierror:
        out["reason"] = "dns_resolution_failed"
        return out

    if not infos:
        out["reason"] = "dns_empty"
        return out
    out["dns_ok"] = True

    resolved_ips = sorted({info[4][0] for info in infos if info and info[4]})
    out["resolved_ips"] = resolved_ips
    if not resolved_ips:
        out["reason"] = "no_resolved_ip"
        return out

    for ip_str in resolved_ips:
        if _is_ip_private_or_local(ip_str):
            out["reason"] = "private_or_local_ip"
            out["bad_ip"] = ip_str
            return out

    out["ips_public"] = True
    out["ok"] = True
    out["reason"] = "ok"
    return out


def probe_http_redirect_chain(
    url: str,
    *,
    max_hops: int = 10,
    timeout: tuple = (5, 12),
) -> Dict[str, Any]:
    """
    Лёгкая проверка цепочки редиректов (HEAD, без тела). Каждый следующий URL
    снова проверяется is_public_http_url. При ошибке возвращает hops + сообщение.
    """
    expl = explain_public_http_url(url)
    if not expl.get("ok"):
        return {"ok": False, "hops": [(url or "").strip()], "error": expl.get("reason") or "blocked"}

    hops: List[str] = [(url or "").strip()]
    current = (url or "").strip()
    last_status: Optional[int] = None
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; BiblioParserUrlProbe/1.0)",
        "Accept": "*/*",
    }

    for _ in range(max_hops):
        try:
            r = requests.head(current, allow_redirects=False, timeout=timeout, headers=headers)
        except requests.RequestException as e:
            return {
                "ok": True,
                "hops": hops,
                "last_status": last_status,
                "probe_error": str(e)[:300],
            }

        last_status = r.status_code
        loc = (r.headers.get("Location") or "").strip()
        if r.status_code in (301, 302, 303, 307, 308) and loc:
            next_u = urljoin(current, loc)
            if not is_public_http_url(next_u):
                hops.append(f"[blocked redirect → {next_u}]")
                return {"ok": True, "hops": hops, "last_status": last_status, "blocked_redirect": True}
            if next_u == current:
                break
            hops.append(next_u)
            current = next_u
            continue
        break

    return {"ok": True, "hops": hops, "last_status": last_status}

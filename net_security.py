import ipaddress
import os
import socket
from urllib.parse import urlparse


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

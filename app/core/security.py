"""SSRF protection and client playback header sanitization."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import socket
from typing import Dict, Iterable, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit

from app.core.errors import StreamURLRejected

logger = logging.getLogger(__name__)

ALLOWED_HEADER_NAMES = frozenset(
    {
        "user-agent",
        "referer",
        "origin",
        "accept",
        "accept-language",
        "cookie",
        "authorization",
    }
)
_HEADER_NAME_RE = re.compile(r"^[A-Za-z0-9-]{1,64}$")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")
_BLOCKED_HOSTNAMES = frozenset(
    {"localhost", "localhost.localdomain", "broadcasthost", "ip6-localhost", "ip6-loopback", "metadata.google.internal"}
)
MAX_URL_LENGTH = 4096
MAX_HEADER_VALUE_LENGTH = 2048


def _is_forbidden_ip(ip: ipaddress._BaseAddress) -> Optional[str]:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local:
        return "link-local / cloud metadata"
    if ip.is_private:
        return "private network"
    if ip.is_multicast:
        return "multicast"
    if ip.is_reserved or ip.is_unspecified:
        return "reserved"
    if not ip.is_global:
        return "non-public"
    return None


def _parse_and_check_syntax(url: str) -> tuple[str, str, int]:
    if not url or not isinstance(url, str):
        raise StreamURLRejected("Stream URL must be a non-empty string.")
    cleaned = url.strip()
    if len(cleaned) > MAX_URL_LENGTH:
        raise StreamURLRejected("Stream URL is too long.")
    if _CONTROL_CHARS_RE.search(cleaned):
        raise StreamURLRejected("Stream URL contains control characters.")
    parts = urlsplit(cleaned)
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:
        raise StreamURLRejected(f"Invalid URL scheme '{parts.scheme}'. Only http and https are allowed.")
    host = (parts.hostname or "").strip("[]").lower()
    if not host:
        raise StreamURLRejected("Stream URL lacks a valid hostname.")
    try:
        port = parts.port or (443 if scheme == "https" else 80)
    except ValueError as exc:
        raise StreamURLRejected("Stream URL has an invalid port.") from exc
    return cleaned, host, port


async def validate_stream_url(url: str, *, allow_private: bool = False) -> str:
    """Validate that ``url`` targets a public HTTP(S) host. Returns the cleaned URL.

    DNS resolution runs off the event loop. Every resolved address must be public.
    """
    cleaned, host, port = _parse_and_check_syntax(url)
    if allow_private:
        return cleaned
    if host in _BLOCKED_HOSTNAMES or host.endswith(".localhost") or host.endswith(".internal"):
        raise StreamURLRejected(f"Access to host '{host}' is prohibited.")

    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        reason = _is_forbidden_ip(literal)
        if reason:
            raise StreamURLRejected(f"Address {host} is a {reason} address.")
        return cleaned

    loop = asyncio.get_running_loop()
    try:
        infos = await asyncio.wait_for(
            loop.getaddrinfo(host, port, type=socket.SOCK_STREAM), timeout=5.0
        )
    except (socket.gaierror, asyncio.TimeoutError, OSError) as exc:
        raise StreamURLRejected(f"Could not resolve hostname '{host}'.") from exc
    if not infos:
        raise StreamURLRejected(f"No addresses resolved for hostname '{host}'.")
    for info in infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str.split("%", 1)[0])
        except ValueError as exc:
            raise StreamURLRejected(f"Invalid resolved address for '{host}'.") from exc
        reason = _is_forbidden_ip(ip)
        if reason:
            raise StreamURLRejected(f"Host '{host}' resolves to a {reason} address.")
    return cleaned


def sanitize_playback_headers(headers: Optional[Mapping[str, object]]) -> Dict[str, str]:
    """Whitelist playback headers and strip anything that could inject CRLF."""
    if not headers or not isinstance(headers, Mapping):
        return {}
    clean: Dict[str, str] = {}
    for key, value in headers.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        name = key.strip()
        if not _HEADER_NAME_RE.match(name) or name.lower() not in ALLOWED_HEADER_NAMES:
            continue
        val = _CONTROL_CHARS_RE.sub("", value).strip()[:MAX_HEADER_VALUE_LENGTH]
        if val:
            clean[name.lower()] = val
    return clean


def mask_headers(headers: Mapping[str, str], sensitive: Iterable[str] = ("cookie", "authorization")) -> Dict[str, str]:
    sens = {s.lower() for s in sensitive}
    return {k: ("***" if k.lower() in sens else v) for k, v in headers.items()}


def redact_url(url: str) -> str:
    """Strip query string and credentials for safe logging."""
    try:
        p = urlsplit(url)
        netloc = p.hostname or ""
        if p.port:
            netloc += f":{p.port}"
        return urlunsplit((p.scheme, netloc, p.path, "", ""))
    except Exception:
        return "<invalid-url>"


_VOLATILE_QUERY_KEYS = re.compile(
    r"^(token|auth|sig|signature|expires?|exp|e|st|policy|key-pair-id|hdnts|hdntl|"
    r"x-amz-.*|x-goog-.*|validfrom|validto|hash|h|t|ts|timestamp|nonce|session|sid|ip)$",
    re.IGNORECASE,
)


def stream_fingerprint(url: str) -> str:
    """Stable identity of a media stream, ignoring volatile auth/expiry query tokens.

    Non-volatile parameters (e.g. ``?id=123``) are kept because they may select content.
    """
    p = urlsplit(url.strip())
    host = (p.hostname or "").lower()
    stable = sorted(
        (k, v) for k, _, v in (pair.partition("=") for pair in p.query.split("&") if pair)
        if not _VOLATILE_QUERY_KEYS.match(k)
    )
    query = "&".join(f"{k}={v}" for k, v in stable)
    base = f"{p.scheme.lower()}://{host}{(':' + str(p.port)) if p.port else ''}{p.path}"
    return f"{base}?{query}" if query else base


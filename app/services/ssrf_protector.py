"""Server-Side Request Forgery (SSRF) and Header Security Validation.

Ensures that external media stream URLs and HTTP headers provided by clients
cannot be abused to probe internal network infrastructure, cloud metadata endpoints,
or localhost services.
"""

import ipaddress
import logging
import re
import socket
from typing import Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class SSRFValidationError(Exception):
    """Raised when an external URL or header fails security validation."""


# Whitelist of client playback headers permitted when accessing media sources
ALLOWED_HEADER_NAMES = {
    "user-agent",
    "referer",
    "origin",
    "accept",
    "range",
    "accept-language",
    "accept-encoding",
    "cookie",
}


def validate_stream_url(url: str) -> str:
    """Validate that the given stream URL is safe from SSRF attacks.

    Requirements:
    - Scheme must be strictly 'http' or 'https'.
    - Hostname must resolve to valid public IP addresses only.
    - Loopback, link-local (cloud metadata 169.254.169.254), private networks (RFC 1918),
      multicast, and reserved ranges are strictly rejected.

    Returns:
        The validated URL string if safe.

    Raises:
        SSRFValidationError: If the URL is invalid or targets a non-public network.
    """
    if not url or not isinstance(url, str):
        raise SSRFValidationError("Stream URL must be a non-empty string.")

    cleaned_url = url.strip()
    parsed = urlparse(cleaned_url)

    # 1. Enforce HTTP/HTTPS scheme
    if parsed.scheme.lower() not in {"http", "https"}:
        raise SSRFValidationError(
            f"Invalid URL scheme '{parsed.scheme}'. Only 'http' and 'https' are allowed."
        )

    hostname = parsed.hostname
    if not hostname:
        raise SSRFValidationError("Stream URL lacks a valid hostname.")

    # 2. Check for explicit localhost strings
    lowered_host = hostname.lower().strip("[]")
    if lowered_host in {"localhost", "localhost.localdomain", "broadcasthost", "ip6-localhost"}:
        raise SSRFValidationError(f"Access to '{hostname}' is prohibited.")

    # 3. Resolve DNS and check all resolved IP addresses
    try:
        # Resolve both IPv4 and IPv6
        addr_info = socket.getaddrinfo(lowered_host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise SSRFValidationError(f"Could not resolve hostname '{hostname}': {exc}") from exc
    except Exception as exc:
        raise SSRFValidationError(f"DNS resolution failure for '{hostname}': {exc}") from exc

    if not addr_info:
        raise SSRFValidationError(f"No IP addresses resolved for hostname '{hostname}'.")

    resolved_ips: List[str] = []
    for entry in addr_info:
        sockaddr = entry[4]
        ip_str = sockaddr[0]
        resolved_ips.append(ip_str)

        try:
            ip_obj = ipaddress.ip_address(ip_str)
        except ValueError:
            raise SSRFValidationError(f"Invalid resolved IP address: {ip_str}")

        # Check for non-public IP ranges
        if str(ip_obj) == "169.254.169.254" or ip_obj.is_link_local:
            raise SSRFValidationError(f"Resolved IP {ip_str} is a link-local / cloud metadata address.")
        if ip_obj.is_loopback:
            raise SSRFValidationError(f"Resolved IP {ip_str} is a loopback address.")
        if ip_obj.is_private:
            raise SSRFValidationError(f"Resolved IP {ip_str} is within a private network.")
        if ip_obj.is_multicast:
            raise SSRFValidationError(f"Resolved IP {ip_str} is a multicast address.")
        if ip_obj.is_reserved:
            raise SSRFValidationError(f"Resolved IP {ip_str} is a reserved address.")
        if ip_obj.is_unspecified:
            raise SSRFValidationError(f"Resolved IP {ip_str} is an unspecified address.")

        # Specific check for AWS / GCP / Azure metadata endpoint: 169.254.169.254
        if str(ip_obj) == "169.254.169.254":
            raise SSRFValidationError("Access to cloud instance metadata service is prohibited.")

    logger.debug("Stream URL '%s' validated successfully. Resolved IPs: %s", hostname, resolved_ips)
    return cleaned_url


def sanitize_playback_headers(headers: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Sanitize and filter playback request headers.

    - Whitelists only permitted HTTP headers.
    - Strips CRLF (\r, \n) and null bytes to prevent injection.
    - Limits header count and maximum length.
    """
    if not headers or not isinstance(headers, dict):
        return {}

    sanitized: Dict[str, str] = {}
    for key, val in headers.items():
        if not key or not isinstance(key, str) or not isinstance(val, str):
            continue

        clean_key = key.strip().lower()
        if clean_key not in ALLOWED_HEADER_NAMES:
            logger.debug("Discarding unapproved playback header '%s'.", key)
            continue

        # Prevent CRLF and control characters injection
        clean_val = re.sub(r"[\r\n\x00]", "", val).strip()

        # Enforce maximum header length of 1024 characters
        if len(clean_val) > 1024:
            clean_val = clean_val[:1024]

        sanitized[clean_key] = clean_val

    return sanitized


def mask_headers_for_logging(headers: Dict[str, str]) -> Dict[str, str]:
    """Return a copy of headers with sensitive authorization or cookie tokens masked."""
    masked: Dict[str, str] = {}
    for k, v in headers.items():
        if k.lower() in {"cookie", "authorization", "token"}:
            masked[k] = "***MASKED***"
        else:
            masked[k] = v
    return masked

"""Tests for SSRF validation and playback header sanitization."""

import pytest

from app.services.ssrf_protector import (
    SSRFValidationError,
    mask_headers_for_logging,
    sanitize_playback_headers,
    validate_stream_url,
)


def test_validate_stream_url_allowed_public():
    """Verify that valid public HTTPS and HTTP URLs pass validation."""
    valid_urls = [
        "https://example.com/stream.m3u8",
        "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
        "http://example.org/audio.mp3",
    ]
    for url in valid_urls:
        validated = validate_stream_url(url)
        assert validated == url


def test_validate_stream_url_rejects_non_http_schemes():
    """Verify non-HTTP/HTTPS schemes (file, ftp, gopher) are rejected."""
    bad_schemes = [
        "file:///etc/passwd",
        "file:///c:/Windows/System32/cmd.exe",
        "ftp://example.com/media.mp4",
        "gopher://example.com/",
        "data:text/html;base64,PHNjcmlwdD4=",
    ]
    for url in bad_schemes:
        with pytest.raises(SSRFValidationError, match="scheme"):
            validate_stream_url(url)


def test_validate_stream_url_rejects_localhost():
    """Verify localhost, 127.0.0.1, and loopback are rejected."""
    loopbacks = [
        "http://localhost/video.mp4",
        "http://localhost:8000/stream.m3u8",
        "http://127.0.0.1/audio.mp3",
        "http://127.0.0.2:8080/manifest.mpd",
        "http://[::1]/video.mp4",
    ]
    for url in loopbacks:
        with pytest.raises(SSRFValidationError, match="prohibited|loopback"):
            validate_stream_url(url)


def test_validate_stream_url_rejects_private_networks():
    """Verify private RFC 1918 addresses are rejected."""
    privates = [
        "http://10.0.0.1/stream.m3u8",
        "http://172.16.0.5/video.mp4",
        "http://192.168.1.100/media.mpd",
    ]
    for url in privates:
        with pytest.raises(SSRFValidationError, match="private network"):
            validate_stream_url(url)


def test_validate_stream_url_rejects_cloud_metadata():
    """Verify AWS / GCP / Azure 169.254.169.254 metadata service is rejected."""
    metadata_urls = [
        "http://169.254.169.254/latest/meta-data/",
        "http://169.254.169.254:80/stream.mp4",
        "http://169.254.1.1/video.mp4",
    ]
    for url in metadata_urls:
        with pytest.raises(SSRFValidationError, match="link-local|metadata"):
            validate_stream_url(url)


def test_sanitize_playback_headers():
    """Verify playback header filtering, CRLF removal, and length truncation."""
    raw_headers = {
        "User-Agent": "CinemaHubApp/1.5.1",
        "Referer": "https://cinema-hub.org/watch\r\nInjected-Header: evil",
        "X-Admin-Password": "secret_password",  # not in whitelist -> dropped
        "Accept": "*/*\n",
        "Cookie": "sessionid=xyz123\x00malicious",
    }

    sanitized = sanitize_playback_headers(raw_headers)

    assert "user-agent" in sanitized
    assert sanitized["user-agent"] == "CinemaHubApp/1.5.1"

    assert "referer" in sanitized
    assert "\r" not in sanitized["referer"]
    assert "\n" not in sanitized["referer"]

    assert "x-admin-password" not in sanitized

    assert "cookie" in sanitized
    assert "\x00" not in sanitized["cookie"]


def test_mask_headers_for_logging():
    """Verify sensitive tokens and cookies are masked for log safety."""
    headers = {
        "user-agent": "CinemaHub/1.0",
        "cookie": "token=secret_12345",
        "referer": "https://cinema-hub.org",
    }
    masked = mask_headers_for_logging(headers)
    assert masked["cookie"] == "***MASKED***"
    assert masked["user-agent"] == "CinemaHub/1.0"


"""Tests for SSRF validation and playback header sanitization."""

from unittest.mock import patch

import pytest

from app.core.errors import StreamURLRejected
from app.core.security import (
    mask_headers,
    redact_url,
    sanitize_playback_headers,
    stream_fingerprint,
    validate_stream_url,
)


@pytest.mark.asyncio
async def test_validate_stream_url_allowed_public():
    valid_urls = [
        "https://example.com/stream.m3u8",
        "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
        "http://8.8.8.8/audio.mp3",
    ]

    async def fake_getaddrinfo(host, port, type=0):
        return [(2, 1, 6, "", ("93.184.216.34", port))]

    with patch("asyncio.base_events.BaseEventLoop.getaddrinfo", side_effect=fake_getaddrinfo):
        for url in valid_urls:
            validated = await validate_stream_url(url)
            assert validated == url


@pytest.mark.asyncio
async def test_validate_stream_url_rejects_non_http_schemes():
    bad_schemes = [
        "file:///etc/passwd",
        "file:///c:/Windows/System32/cmd.exe",
        "ftp://example.com/media.mp4",
        "gopher://example.com/",
        "data:text/html;base64,PHNjcmlwdD4=",
    ]
    for url in bad_schemes:
        with pytest.raises(StreamURLRejected, match="scheme"):
            await validate_stream_url(url)


@pytest.mark.asyncio
async def test_validate_stream_url_rejects_localhost():
    loopbacks = [
        "http://localhost/video.mp4",
        "http://localhost:8000/stream.m3u8",
        "http://127.0.0.1/audio.mp3",
        "http://127.0.0.2:8080/manifest.mpd",
        "http://[::1]/video.mp4",
    ]
    for url in loopbacks:
        with pytest.raises(StreamURLRejected, match="prohibited|loopback"):
            await validate_stream_url(url)


@pytest.mark.asyncio
async def test_validate_stream_url_rejects_private_networks():
    privates = [
        "http://10.0.0.1/stream.m3u8",
        "http://172.16.0.5/video.mp4",
        "http://192.168.1.100/media.mpd",
    ]
    for url in privates:
        with pytest.raises(StreamURLRejected, match="private network"):
            await validate_stream_url(url)


@pytest.mark.asyncio
async def test_validate_stream_url_rejects_cloud_metadata():
    metadata_urls = [
        "http://169.254.169.254/latest/meta-data/",
        "http://169.254.169.254:80/stream.mp4",
        "http://169.254.1.1/video.mp4",
    ]
    for url in metadata_urls:
        with pytest.raises(StreamURLRejected, match="link-local|metadata"):
            await validate_stream_url(url)


def test_sanitize_playback_headers():
    raw_headers = {
        "User-Agent": "CinemaHubApp/1.5.1",
        "Referer": "https://cinema-hub.org/watch\r\nInjected-Header: evil",
        "X-Admin-Password": "secret_password",
        "Accept": "*/*\n",
        "Cookie": "sessionid=xyz123\x00malicious",
    }
    sanitized = sanitize_playback_headers(raw_headers)

    assert sanitized["user-agent"] == "CinemaHubApp/1.5.1"
    assert "\r" not in sanitized["referer"]
    assert "\n" not in sanitized["referer"]
    assert "x-admin-password" not in sanitized
    assert "\x00" not in sanitized["cookie"]


def test_mask_and_redact_and_fingerprint():
    masked = mask_headers({"user-agent": "CinemaHub/1.0", "cookie": "secret", "authorization": "Bearer x"})
    assert masked["cookie"] == "***"
    assert masked["authorization"] == "***"
    assert masked["user-agent"] == "CinemaHub/1.0"

    assert redact_url("https://user:pass@cdn.example.com/v/1.m3u8?token=abc") == "https://cdn.example.com/v/1.m3u8"
    fp1 = stream_fingerprint("https://cdn.example.com/v/1.m3u8?id=42&token=abc&expires=999")
    fp2 = stream_fingerprint("https://cdn.example.com/v/1.m3u8?token=xyz&id=42&exp=111")
    assert fp1 == fp2 == "https://cdn.example.com/v/1.m3u8?id=42"


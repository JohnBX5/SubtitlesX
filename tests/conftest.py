"""Pytest configuration, sample subtitle fixtures, and isolated app/service fixtures."""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.cache.store import CacheStore
from app.core.config import Settings, get_settings
from app.main import create_app
from app.opensubtitles.client import OpenSubtitlesClient
from app.routers.subtitles import get_service
from app.service import SubtitleBackendService
from app.sync.engine import SyncEngine

SAMPLE_VALID_SRT = """1
00:00:01,000 --> 00:00:04,000
Welcome to Cinema Hub!

2
00:00:05,500 --> 00:00:08,200
<i>Enjoy high quality streaming with subtitles.</i>

3
00:00:09,000 --> 00:00:12,000
Multi-line subtitle text
Line two of dialogue.
"""

SAMPLE_ARABIC_SRT = """1
00:00:01,000 --> 00:00:04,500
مرحبا بكم في سينما هاب!

2
00:00:05,000 --> 00:00:09,000
هذا ملف ترجمة باللغة العربية مع علامات ترقيم كاملة.

3
00:00:10,000 --> 00:00:14,000
مشاهدة ممتعة للجميع.
"""

SAMPLE_VALID_VTT = """WEBVTT

1
00:00:01.000 --> 00:00:04.000
Welcome to Cinema Hub WebVTT!

2
00:00:05.500 --> 00:00:08.200
<b>Bold dialogue cue</b>
"""

SAMPLE_VALID_ASS = """[Script Info]
Title: Cinema Hub Test
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,50,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:04.00,Default,,0,0,0,,Hello world from ASS!
Dialogue: 0,0:00:05.00,0:00:08.00,Default,,0,0,0,,{\\b1}Stylized line{\\b0}
"""

SAMPLE_VALID_TTML = """<?xml version="1.0" encoding="utf-8"?>
<tt xmlns="http://www.w3.org/ns/ttml" xmlns:ttp="http://www.w3.org/ns/ttml#parameter" ttp:timeBase="media">
  <body>
    <div>
      <p begin="00:00:01.000" end="00:00:04.000">TTML Subtitle line 1</p>
      <p begin="00:00:05.000" end="00:00:08.000">TTML Subtitle line 2</p>
    </div>
  </body>
</tt>
"""

SAMPLE_CLOUDFLARE_HTML = """<!DOCTYPE html>
<html lang="en-US">
<head>
<title>Attention Required! | Cloudflare</title>
<meta name="captcha-bypass" content="no" />
</head>
<body>
<h1>Please complete the security check to access opensubtitles.com</h1>
<div id="cf-browser-verification">Verifying your browser...</div>
</body>
</html>
"""

SAMPLE_HTML_ERROR = """<!DOCTYPE html>
<html>
<head><title>502 Bad Gateway</title></head>
<body><center><h1>502 Bad Gateway</h1></center></body>
</html>
"""


@pytest.fixture(scope="session")
def test_temp_dir() -> Generator[str, None, None]:
    temp_dir = tempfile.mkdtemp(prefix="test_subtitles_v2_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_settings(test_temp_dir: str, monkeypatch: pytest.MonkeyPatch) -> Settings:
    cache_dir = os.path.join(test_temp_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    db_file = os.path.join(test_temp_dir, "test_cache.db")

    monkeypatch.setenv("CACHE_DIR", cache_dir)
    monkeypatch.setenv("CACHE_DB_PATH", db_file)
    monkeypatch.setenv("OPENSUBTITLES_API_KEY", "test_api_key_12345")
    monkeypatch.setenv("OPENSUBTITLES_USERNAME", "test_user")
    monkeypatch.setenv("OPENSUBTITLES_PASSWORD", "test_password")
    monkeypatch.setenv("HTTP_TIMEOUT_SECONDS", "5.0")
    monkeypatch.setenv("CACHE_TTL_HOURS", "24")
    monkeypatch.setenv("RATE_LIMIT_SYNC_PER_MINUTE", "0")
    monkeypatch.setenv("API_ACCESS_KEY", "")

    get_settings.cache_clear()
    return Settings(
        CACHE_DIR=cache_dir,
        CACHE_DB_PATH=":memory:",
        OPENSUBTITLES_API_KEY="test_api_key_12345",
        OPENSUBTITLES_USERNAME="test_user",
        OPENSUBTITLES_PASSWORD="test_password",
        HTTP_TIMEOUT_SECONDS=5.0,
        CACHE_TTL_HOURS=24,
        RATE_LIMIT_SYNC_PER_MINUTE=0,
        API_ACCESS_KEY="",
    )


@pytest_asyncio.fixture
async def cache_store(test_settings: Settings) -> AsyncGenerator[CacheStore, None]:
    store = CacheStore(":memory:", memory_max_mb=16, max_size_mb=64)
    await store.open()
    try:
        yield store
    finally:
        await store.close()


@pytest_asyncio.fixture
async def backend_service(
    test_settings: Settings, cache_store: CacheStore
) -> AsyncGenerator[SubtitleBackendService, None]:
    os_client = OpenSubtitlesClient(test_settings)
    sync_engine = SyncEngine(settings=test_settings, cache=cache_store)
    service = SubtitleBackendService(
        settings=test_settings,
        cache=cache_store,
        opensubtitles=os_client,
        sync_engine=sync_engine,
    )
    try:
        yield service
    finally:
        sync_engine.close()
        await os_client.close()


@pytest_asyncio.fixture
async def test_app(
    test_settings: Settings, backend_service: SubtitleBackendService
) -> AsyncGenerator[FastAPI, None]:
    application = create_app(test_settings)
    application.state.service = backend_service
    application.state.cache = backend_service.cache
    application.state.opensubtitles = backend_service.opensubtitles
    application.state.sync_engine = backend_service.sync_engine
    application.dependency_overrides[get_service] = lambda: backend_service
    yield application
    application.dependency_overrides.clear()


@pytest_asyncio.fixture
async def async_client(test_app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


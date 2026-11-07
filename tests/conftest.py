"""Pytest configuration, fixtures, and mock factories."""

import asyncio
import os
import shutil
import tempfile
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings, get_settings
from app.database import Base, get_db
from app.main import app
from app.schemas.subtitle import SubtitleCandidate, SubtitleFileInfo


# -----------------------------------------------------------------------------
# Sample Subtitle Data Fixtures
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Environment & Temporary Database Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for tests and clean up on teardown."""
    temp_dir = tempfile.mkdtemp(prefix="test_subtitles_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def override_settings(test_temp_dir: str, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Override application settings with isolated test configurations."""
    test_cache_dir = os.path.join(test_temp_dir, "cache")
    os.makedirs(test_cache_dir, exist_ok=True)
    db_file = os.path.join(test_temp_dir, "test_subtitles.db")

    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_file}")
    monkeypatch.setenv("CACHE_DIR", test_cache_dir)
    monkeypatch.setenv("OPENSUBTITLES_API_KEY", "test_api_key_12345")
    monkeypatch.setenv("OPENSUBTITLES_USERNAME", "test_user")
    monkeypatch.setenv("OPENSUBTITLES_PASSWORD", "test_password")
    monkeypatch.setenv("HTTP_TIMEOUT_SECONDS", "5.0")
    monkeypatch.setenv("CACHE_TTL_HOURS", "24")

    # Invalidate cached settings
    from app.config import get_settings
    get_settings.cache_clear()
    return get_settings()


@pytest_asyncio.fixture
async def test_db_session(override_settings: Settings) -> AsyncGenerator[AsyncSession, None]:
    """Provide an isolated database session with freshly created tables."""
    engine = create_async_engine(
        override_settings.DATABASE_URL,
        echo=False,
        connect_args={"check_same_thread": False},
    )
    async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    async with async_session() as session:
        yield session

    await engine.dispose()


@pytest_asyncio.fixture
async def async_client(test_db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Provide an HTTPX AsyncClient bound to the FastAPI app with test db session dependency."""
    async def override_get_db():
        yield test_db_session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()


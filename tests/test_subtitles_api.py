"""Full API integration tests for /api/subtitles/languages and /api/subtitles/download."""

import os
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from app.schemas.subtitle import MovieInfo, SubtitleCandidate, SubtitleFileInfo
from tests.conftest import SAMPLE_VALID_SRT


@pytest.mark.asyncio
async def test_languages_endpoint_missing_parameters(async_client: AsyncClient):
    """Verify /api/subtitles/languages returns 400 when neither imdb_id nor query is provided."""
    response = await async_client.get("/api/subtitles/languages")
    assert response.status_code == 400
    data = response.json()
    assert "At least one valid identification method is required" in data["detail"]


@pytest.mark.asyncio
async def test_languages_endpoint_success(async_client: AsyncClient):
    """Verify /api/subtitles/languages returns 200 with structured language discovery data."""
    mock_movie = MovieInfo(imdb_id="tt0111161", title="The Shawshank Redemption")
    mock_candidates = [
        SubtitleCandidate(
            id="1",
            subtitle_id="101",
            language="ar",
            files=[SubtitleFileInfo(file_id=1001, file_name="shawshank_ar.srt")],
        ),
        SubtitleCandidate(
            id="2",
            subtitle_id="102",
            language="en",
            files=[
                SubtitleFileInfo(file_id=2001, file_name="shawshank_en_cd1.srt"),
                SubtitleFileInfo(file_id=2002, file_name="shawshank_en_cd2.srt"),
            ],
        ),
    ]

    with patch(
        "app.routers.subtitles.opensubtitles_client.search_subtitles",
        new_callable=AsyncMock,
    ) as mock_search:
        mock_search.return_value = (mock_movie, mock_candidates)

        response = await async_client.get("/api/subtitles/languages?imdb_id=tt0111161")
        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["movie"]["imdb_id"] == "tt0111161"
        assert data["movie"]["title"] == "The Shawshank Redemption"

        langs = data["languages"]
        assert len(langs) == 2

        # English has 2 files
        en_item = next(l for l in langs if l["language_code"] == "en")
        assert en_item["available_files"] == 2
        assert en_item["language_name"] == "English"

        # Arabic has 1 file
        ar_item = next(l for l in langs if l["language_code"] == "ar")
        assert ar_item["available_files"] == 1
        assert ar_item["language_name"] == "Arabic"


@pytest.mark.asyncio
async def test_download_endpoint_missing_language(async_client: AsyncClient):
    """Verify /api/subtitles/download returns 422 when required language query param is missing."""
    response = await async_client.get("/api/subtitles/download?imdb_id=tt0111161")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_download_endpoint_invalid_language_format(async_client: AsyncClient):
    """Verify /api/subtitles/download returns 400 for an invalid language code format."""
    response = await async_client.get("/api/subtitles/download?imdb_id=tt0111161&language=invalid_code_123")
    assert response.status_code == 400
    assert "Invalid language code" in response.json()["detail"]


@pytest.mark.asyncio
async def test_download_endpoint_not_found(async_client: AsyncClient):
    """Verify /api/subtitles/download returns 404 when no subtitles match the query."""
    with patch(
        "app.routers.subtitles.opensubtitles_client.search_subtitles",
        new_callable=AsyncMock,
    ) as mock_search:
        mock_search.return_value = (None, [])

        response = await async_client.get("/api/subtitles/download?imdb_id=tt0111161&language=ar")
        assert response.status_code == 404
        assert "No subtitles found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_download_cache_miss_then_hit(async_client: AsyncClient):
    """Verify complete download workflow: cache miss fetches & stores, second call is cache hit."""
    mock_movie = MovieInfo(imdb_id="tt0111161", title="The Shawshank Redemption")
    mock_candidates = [
        SubtitleCandidate(
            id="sub_999",
            subtitle_id="999",
            language="en",
            from_trusted=True,
            ratings=9.0,
            votes=50,
            download_count=10000,
            files=[SubtitleFileInfo(file_id=88888, file_name="shawshank.srt")],
        )
    ]

    with patch(
        "app.routers.subtitles.opensubtitles_client.search_subtitles",
        new_callable=AsyncMock,
    ) as mock_search, patch(
        "app.routers.subtitles.opensubtitles_client.request_download_link",
        new_callable=AsyncMock,
    ) as mock_req_link, patch(
        "app.routers.subtitles.opensubtitles_client.fetch_file_bytes",
        new_callable=AsyncMock,
    ) as mock_fetch:

        mock_search.return_value = (mock_movie, mock_candidates)
        mock_req_link.return_value = ("https://download.mock/sub.srt", "The.Shawshank.Redemption.1994.en.srt")
        mock_fetch.return_value = SAMPLE_VALID_SRT.encode("utf-8")

        # 1. First Request: Cache MISS
        resp1 = await async_client.get("/api/subtitles/download?imdb_id=tt0111161&language=en")
        assert resp1.status_code == 200
        assert resp1.headers["X-Cache"] == "MISS"
        assert resp1.headers["X-Subtitle-Format"] == "srt"
        assert resp1.headers["X-Subtitle-Cue-Count"] == "3"
        assert "attachment" in resp1.headers["Content-Disposition"]
        assert "Welcome to Cinema Hub!" in resp1.text
        assert mock_search.call_count == 1
        assert mock_fetch.call_count == 1

        # 2. Second Request: Cache HIT
        resp2 = await async_client.get("/api/subtitles/download?imdb_id=tt0111161&language=en")
        assert resp2.status_code == 200
        assert resp2.headers["X-Cache"] == "HIT"
        assert resp2.headers["X-Subtitle-Format"] == "srt"
        assert resp2.headers["X-Subtitle-Cue-Count"] == "3"
        assert "Welcome to Cinema Hub!" in resp2.text

        # Verify OpenSubtitles was NOT contacted on cache hit
        assert mock_search.call_count == 1
        assert mock_fetch.call_count == 1


@pytest.mark.asyncio
async def test_download_endpoint_vtt_conversion(async_client: AsyncClient):
    """Verify that requesting format=vtt converts SRT to WebVTT with appropriate headers."""
    mock_movie = MovieInfo(imdb_id="tt0111161", title="The Shawshank Redemption")
    mock_candidates = [
        SubtitleCandidate(
            id="sub_1",
            subtitle_id="1",
            language="en",
            from_trusted=True,
            files=[SubtitleFileInfo(file_id=111, file_name="sub.srt")],
        )
    ]

    with patch(
        "app.routers.subtitles.opensubtitles_client.search_subtitles",
        new_callable=AsyncMock,
    ) as mock_search, patch(
        "app.routers.subtitles.opensubtitles_client.request_download_link",
        new_callable=AsyncMock,
    ) as mock_req_link, patch(
        "app.routers.subtitles.opensubtitles_client.fetch_file_bytes",
        new_callable=AsyncMock,
    ) as mock_fetch:

        mock_search.return_value = (mock_movie, mock_candidates)
        mock_req_link.return_value = ("https://download.mock/sub.srt", "movie.srt")
        mock_fetch.return_value = SAMPLE_VALID_SRT.encode("utf-8")

        response = await async_client.get("/api/subtitles/download?imdb_id=tt0111161&language=en&format=vtt")
        assert response.status_code == 200
        assert response.headers["X-Subtitle-Format"] == "vtt"
        assert "text/vtt" in response.headers["Content-Type"]
        assert response.text.startswith("WEBVTT")
        assert "00:00:01.000 --> 00:00:04.000" in response.text


# -----------------------------------------------------------------------------
# Optional Live Integration Test (Opt-in via LIVE_OPENSUBTITLES_TEST=1)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("LIVE_OPENSUBTITLES_TEST") != "1",
    reason="Live OpenSubtitles test skipped by default. Set LIVE_OPENSUBTITLES_TEST=1 and configure .env to run.",
)
async def test_live_opensubtitles_integration(async_client: AsyncClient):
    """Opt-in live integration test against real OpenSubtitles API.

    Verifies authentication, discovery for Shawshank Redemption (tt0111161),
    downloading a real subtitle, and persistent caching.
    """
    # 1. Test live language discovery
    lang_resp = await async_client.get("/api/subtitles/languages?imdb_id=tt0111161")
    assert lang_resp.status_code == 200
    data = lang_resp.json()
    assert data["success"] is True
    assert len(data["languages"]) > 0

    # 2. Pick the first available language code
    first_lang = data["languages"][0]["language_code"]

    # 3. Download the subtitle
    dl_resp = await async_client.get(f"/api/subtitles/download?imdb_id=tt0111161&language={first_lang}")
    assert dl_resp.status_code == 200
    assert dl_resp.headers["X-Cache"] == "MISS"
    assert len(dl_resp.text) > 0

    # 4. Immediate second download must be a cache hit
    hit_resp = await async_client.get(f"/api/subtitles/download?imdb_id=tt0111161&language={first_lang}")
    assert hit_resp.status_code == 200
    assert hit_resp.headers["X-Cache"] == "HIT"


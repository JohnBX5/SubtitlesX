"""API integration tests for /api/subtitles/languages and /api/subtitles/download."""

import os
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from app.opensubtitles.models import LanguageInfo, MovieInfo, SubtitleCandidate, SubtitleFile
from app.service import SubtitleBackendService
from tests.conftest import SAMPLE_VALID_SRT


@pytest.mark.asyncio
async def test_languages_endpoint_missing_parameters(async_client: AsyncClient):
    response = await async_client.get("/api/subtitles/languages")
    assert response.status_code == 400
    assert "At least one valid identification method is required" in response.json()["detail"]


@pytest.mark.asyncio
async def test_languages_endpoint_success(async_client: AsyncClient, backend_service: SubtitleBackendService):
    mock_movie = MovieInfo(imdb_id="tt0111161", title="The Shawshank Redemption")
    mock_langs = [
        LanguageInfo(language_code="en", language_name="English", available_files=2),
        LanguageInfo(language_code="ar", language_name="Arabic", available_files=1),
    ]

    with patch.object(
        backend_service.opensubtitles,
        "languages",
        new_callable=AsyncMock,
        return_value=(mock_movie, mock_langs),
    ):
        response = await async_client.get("/api/subtitles/languages?imdb_id=tt0111161")
        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["movie"]["imdb_id"] == "tt0111161"
        assert data["movie"]["title"] == "The Shawshank Redemption"
        langs = data["languages"]
        assert len(langs) == 2
        assert next(l for l in langs if l["language_code"] == "en")["available_files"] == 2
        assert next(l for l in langs if l["language_code"] == "ar")["available_files"] == 1


@pytest.mark.asyncio
async def test_download_endpoint_missing_language(async_client: AsyncClient):
    response = await async_client.get("/api/subtitles/download?imdb_id=tt0111161")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_download_endpoint_invalid_language_format(async_client: AsyncClient):
    response = await async_client.get("/api/subtitles/download?imdb_id=tt0111161&language=invalid_code_123")
    assert response.status_code == 400
    assert "Invalid language code" in response.json()["detail"]


@pytest.mark.asyncio
async def test_download_endpoint_not_found(async_client: AsyncClient, backend_service: SubtitleBackendService):
    with patch.object(
        backend_service.opensubtitles,
        "search",
        new_callable=AsyncMock,
        return_value=(None, []),
    ):
        response = await async_client.get("/api/subtitles/download?imdb_id=tt0111161&language=ar")
        assert response.status_code == 404
        assert "No subtitles found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_download_cache_miss_then_hit(async_client: AsyncClient, backend_service: SubtitleBackendService):
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
            files=[SubtitleFile(file_id=88888, file_name="shawshank.srt")],
        )
    ]

    with patch.object(
        backend_service.opensubtitles,
        "search",
        new_callable=AsyncMock,
        return_value=(mock_movie, mock_candidates),
    ) as mock_search, patch.object(
        backend_service.opensubtitles,
        "download_file",
        new_callable=AsyncMock,
        return_value=(SAMPLE_VALID_SRT.encode("utf-8"), "The.Shawshank.Redemption.1994.en.srt"),
    ) as mock_dl:
        resp1 = await async_client.get("/api/subtitles/download?imdb_id=tt0111161&language=en")
        assert resp1.status_code == 200
        assert resp1.headers["X-Cache"] == "MISS"
        assert resp1.headers["X-Subtitle-Format"] == "srt"
        assert resp1.headers["X-Subtitle-Cue-Count"] == "3"
        assert "attachment" in resp1.headers["Content-Disposition"]
        assert "Welcome to Cinema Hub!" in resp1.text
        assert mock_search.call_count == 1
        assert mock_dl.call_count == 1

        resp2 = await async_client.get("/api/subtitles/download?imdb_id=tt0111161&language=en")
        assert resp2.status_code == 200
        assert resp2.headers["X-Cache"] == "HIT"
        assert resp2.headers["X-Subtitle-Format"] == "srt"
        assert resp2.headers["X-Subtitle-Cue-Count"] == "3"
        assert "Welcome to Cinema Hub!" in resp2.text
        assert mock_search.call_count == 1
        assert mock_dl.call_count == 1


@pytest.mark.asyncio
async def test_download_endpoint_vtt_conversion(async_client: AsyncClient, backend_service: SubtitleBackendService):
    mock_movie = MovieInfo(imdb_id="tt0111161", title="The Shawshank Redemption")
    mock_candidates = [
        SubtitleCandidate(
            id="sub_1",
            subtitle_id="1",
            language="en",
            from_trusted=True,
            files=[SubtitleFile(file_id=111, file_name="sub.srt")],
        )
    ]

    with patch.object(
        backend_service.opensubtitles,
        "search",
        new_callable=AsyncMock,
        return_value=(mock_movie, mock_candidates),
    ), patch.object(
        backend_service.opensubtitles,
        "download_file",
        new_callable=AsyncMock,
        return_value=(SAMPLE_VALID_SRT.encode("utf-8"), "movie.srt"),
    ):
        response = await async_client.get("/api/subtitles/download?imdb_id=tt0111161&language=en&format=vtt")
        assert response.status_code == 200
        assert response.headers["X-Subtitle-Format"] == "vtt"
        assert "text/vtt" in response.headers["Content-Type"]
        assert response.text.startswith("WEBVTT")
        assert "00:00:01.000 --> 00:00:04.000" in response.text


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("LIVE_OPENSUBTITLES_TEST") != "1",
    reason="Live OpenSubtitles test skipped by default.",
)
async def test_live_opensubtitles_integration(async_client: AsyncClient):
    lang_resp = await async_client.get("/api/subtitles/languages?imdb_id=tt0111161")
    assert lang_resp.status_code == 200
    data = lang_resp.json()
    assert data["success"] is True
    assert len(data["languages"]) > 0


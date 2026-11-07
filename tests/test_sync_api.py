"""Integration tests for the /api/subtitles/synchronize and synchronized /download endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from app.schemas.subtitle import MovieInfo, SubtitleCandidate, SubtitleFileInfo
from app.services.sync_service import SyncResult
from tests.conftest import SAMPLE_VALID_SRT


@pytest.mark.asyncio
async def test_synchronize_endpoint_missing_parameters(async_client: AsyncClient):
    """Verify /api/subtitles/synchronize returns 400 when no identification or subtitle is provided."""
    payload = {"language": "en"}
    response = await async_client.post("/api/subtitles/synchronize", json=payload)
    assert response.status_code == 400
    assert "At least one subtitle source is required" in response.json()["detail"]


@pytest.mark.asyncio
async def test_synchronize_endpoint_invalid_language(async_client: AsyncClient):
    """Verify /api/subtitles/synchronize returns 400 for an invalid language code format."""
    payload = {
        "language": "invalid_code_123",
        "imdb_id": "tt0111161",
    }
    response = await async_client.post("/api/subtitles/synchronize", json=payload)
    assert response.status_code == 400
    assert "Invalid language code" in response.json()["detail"]


@pytest.mark.asyncio
async def test_synchronize_endpoint_with_custom_subtitle_content(async_client: AsyncClient):
    """Verify synchronization works when the client provides direct subtitle_content."""
    mock_sync_result = SyncResult(
        success=True,
        status="synchronized",
        offset_seconds=3.25,
        confidence_score=0.85,
        audio_analyzed=True,
        processing_time_seconds=0.45,
        subtitle_content="1\n00:00:04,250 --> 00:00:07,250\nWelcome to Cinema Hub!\n",
        format="srt",
        cue_count=1,
        detail="Alignment status: synchronized",
    )

    with patch(
        "app.routers.subtitles.subtitle_sync_service.synchronize_subtitle",
        new_callable=AsyncMock,
        return_value=mock_sync_result,
    ):
        payload = {
            "language": "en",
            "subtitle_content": SAMPLE_VALID_SRT,
            "stream_url": "https://example.com/stream.m3u8",
        }
        response = await async_client.post("/api/subtitles/synchronize", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["sync_status"] == "synchronized"
        assert data["offset_seconds"] == 3.25
        assert data["audio_analyzed"] is True
        assert "00:00:04,250" in data["subtitle_content"]


@pytest.mark.asyncio
async def test_synchronize_endpoint_fallback_unavailable(async_client: AsyncClient):
    """Verify synchronization returns status='unavailable' when stream_url is omitted."""
    payload = {
        "language": "en",
        "subtitle_content": SAMPLE_VALID_SRT,
    }
    response = await async_client.post("/api/subtitles/synchronize", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert data["sync_status"] == "unavailable"
    assert data["audio_analyzed"] is False
    assert "Welcome to Cinema Hub!" in data["subtitle_content"]


@pytest.mark.asyncio
async def test_download_endpoint_with_sync_param(async_client: AsyncClient):
    """Verify /api/subtitles/download with sync=true performs synchronization and attaches X-Sync headers."""
    mock_movie = MovieInfo(imdb_id="tt0111161", title="The Shawshank Redemption")
    mock_candidates = [
        SubtitleCandidate(
            id="1",
            subtitle_id="101",
            language="en",
            files=[SubtitleFileInfo(file_id=1001, file_name="shawshank.srt")],
        )
    ]
    mock_sync_result = SyncResult(
        success=True,
        status="synchronized",
        offset_seconds=5.0,
        confidence_score=0.9,
        audio_analyzed=True,
        processing_time_seconds=0.3,
        subtitle_content="1\n00:00:06,000 --> 00:00:09,000\nWelcome to Cinema Hub!\n",
        format="srt",
        cue_count=1,
    )

    with patch(
        "app.routers.subtitles.opensubtitles_client.search_subtitles",
        new_callable=AsyncMock,
        return_value=(mock_movie, mock_candidates),
    ), patch(
        "app.routers.subtitles.opensubtitles_client.request_download_link",
        new_callable=AsyncMock,
        return_value=("https://download.mock/sub.srt", "sub.srt"),
    ), patch(
        "app.routers.subtitles.opensubtitles_client.fetch_file_bytes",
        new_callable=AsyncMock,
        return_value=SAMPLE_VALID_SRT.encode("utf-8"),
    ), patch(
        "app.routers.subtitles.subtitle_sync_service.synchronize_subtitle",
        new_callable=AsyncMock,
        return_value=mock_sync_result,
    ):
        url = "/api/subtitles/download?imdb_id=tt0111161&language=en&sync=true&stream_url=https://example.com/video.mp4"
        response = await async_client.get(url)

        assert response.status_code == 200
        assert response.headers["X-Sync-Status"] == "synchronized"
        assert response.headers["X-Sync-Offset"] == "5.000"
        assert response.headers["X-Audio-Analyzed"] == "true"
        assert "Welcome to Cinema Hub!" in response.text


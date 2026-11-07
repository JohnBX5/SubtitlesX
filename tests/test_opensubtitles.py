"""Unit and integration tests for OpenSubtitles API client, authentication, and ranking."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.config import get_settings
from app.schemas.subtitle import SubtitleCandidate, SubtitleFileInfo
from app.services.authentication import OpenSubtitlesAuthError, OpenSubtitlesAuthManager
from app.services.opensubtitles import (
    OpenSubtitlesAPIError,
    OpenSubtitlesClient,
    opensubtitles_client,
)
from app.services.ranking import (
    calculate_candidate_score,
    rank_subtitle_candidates,
    select_best_candidate,
)
from tests.conftest import SAMPLE_VALID_SRT


# -----------------------------------------------------------------------------
# Authentication Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auth_missing_credentials(monkeypatch):
    """Verify OpenSubtitlesAuthError is raised when credentials are missing."""
    settings = get_settings()
    monkeypatch.setattr(settings, "OPENSUBTITLES_API_KEY", "")

    auth = OpenSubtitlesAuthManager()
    client = httpx.AsyncClient()
    with pytest.raises(OpenSubtitlesAuthError, match="OPENSUBTITLES_API_KEY is not configured"):
        await auth.get_valid_token(client)


@pytest.mark.asyncio
async def test_auth_successful_login_and_token_caching():
    """Verify authentication requests token and caches it in memory."""
    auth = OpenSubtitlesAuthManager()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "status": 200,
        "token": "valid_mock_bearer_token_12345",
        "user": {"user_id": 99999, "level": "vip"},
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    # First call: authenticates
    token1 = await auth.get_valid_token(mock_client)
    assert token1 == "valid_mock_bearer_token_12345"
    assert mock_client.post.call_count == 1

    # Second call: uses cached token
    token2 = await auth.get_valid_token(mock_client)
    assert token2 == "valid_mock_bearer_token_12345"
    assert mock_client.post.call_count == 1  # No additional post call


@pytest.mark.asyncio
async def test_auth_invalidate_token():
    """Verify invalidating token forces re-authentication."""
    auth = OpenSubtitlesAuthManager()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "token": "refreshed_token_54321",
        "user": {"user_id": 99999},
    }
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    token1 = await auth.get_valid_token(mock_client)
    assert token1 == "refreshed_token_54321"

    # Invalidate
    await auth.invalidate_token()
    assert not auth.is_authenticated

    # Next call authenticates again
    token2 = await auth.get_valid_token(mock_client)
    assert token2 == "refreshed_token_54321"
    assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_auth_prevent_thundering_herd():
    """Verify that multiple concurrent login calls share the lock and trigger only one login call."""
    auth = OpenSubtitlesAuthManager()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "token": "shared_token_abc",
        "user": {"user_id": 123},
    }

    call_count = 0

    async def delayed_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.05)
        return mock_response

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.side_effect = delayed_post

    # Launch 5 concurrent calls
    tokens = await asyncio.gather(*[auth.get_valid_token(mock_client) for _ in range(5)])

    assert all(t == "shared_token_abc" for t in tokens)
    assert call_count == 1


# -----------------------------------------------------------------------------
# Ranking Algorithm Tests
# -----------------------------------------------------------------------------


def test_ranking_trusted_uploader_preference():
    """Verify candidate from trusted uploader outranks standard candidate with same downloads."""
    cand_trusted = SubtitleCandidate(
        id="1",
        subtitle_id="sub1",
        language="en",
        from_trusted=True,
        download_count=1000,
        ratings=8.0,
        votes=10,
        files=[SubtitleFileInfo(file_id=1, file_name="sub1.srt")],
    )
    cand_untrusted = SubtitleCandidate(
        id="2",
        subtitle_id="sub2",
        language="en",
        from_trusted=False,
        download_count=1000,
        ratings=8.0,
        votes=10,
        files=[SubtitleFileInfo(file_id=2, file_name="sub2.srt")],
    )

    ranked = rank_subtitle_candidates([cand_untrusted, cand_trusted])
    assert ranked[0].id == "1"
    assert ranked[0].ranking_score > ranked[1].ranking_score


def test_ranking_machine_translated_penalty():
    """Verify machine translated subtitles receive a severe penalty."""
    cand_human = SubtitleCandidate(
        id="1",
        subtitle_id="sub1",
        language="en",
        machine_translated=False,
        download_count=50,
        files=[SubtitleFileInfo(file_id=1, file_name="sub1.srt")],
    )
    cand_machine = SubtitleCandidate(
        id="2",
        subtitle_id="sub2",
        language="en",
        machine_translated=True,
        download_count=5000,  # even with more downloads
        files=[SubtitleFileInfo(file_id=2, file_name="sub2.srt")],
    )

    ranked = rank_subtitle_candidates([cand_machine, cand_human])
    assert ranked[0].id == "1"
    assert ranked[1].ranking_score < 0


def test_ranking_deterministic_tie_breaker():
    """Verify that identical candidates are sorted deterministically by primary file_id."""
    cand_a = SubtitleCandidate(
        id="a",
        subtitle_id="sub_a",
        language="en",
        download_count=100,
        ratings=5.0,
        files=[SubtitleFileInfo(file_id=200, file_name="a.srt")],
    )
    cand_b = SubtitleCandidate(
        id="b",
        subtitle_id="sub_b",
        language="en",
        download_count=100,
        ratings=5.0,
        files=[SubtitleFileInfo(file_id=100, file_name="b.srt")],
    )

    ranked1 = rank_subtitle_candidates([cand_a, cand_b])
    ranked2 = rank_subtitle_candidates([cand_b, cand_a])

    # file_id 100 should come first in tie-breaker
    assert ranked1[0].files[0].file_id == 100
    assert ranked2[0].files[0].file_id == 100


# -----------------------------------------------------------------------------
# OpenSubtitles Client Search & Download Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normalize_imdb_id():
    """Verify IMDb ID normalization handles tt prefix and pure numeric strings."""
    client = OpenSubtitlesClient()
    assert client.normalize_imdb_id("tt0111161") == 111161
    assert client.normalize_imdb_id("0111161") == 111161
    assert client.normalize_imdb_id("111161") == 111161
    assert client.normalize_imdb_id(None) is None


@pytest.mark.asyncio
async def test_search_subtitles_parsing():
    """Verify OpenSubtitles search response correctly extracts movie info and candidates."""
    mock_payload = {
        "total_count": 1,
        "data": [
            {
                "id": "item_123",
                "attributes": {
                    "subtitle_id": "123456",
                    "language": "ar",
                    "download_count": 3500,
                    "ratings": 8.5,
                    "votes": 25,
                    "hearing_impaired": False,
                    "hd": True,
                    "from_trusted": True,
                    "release": "The.Shawshank.Redemption.1994.1080p",
                    "feature_details": {
                        "title": "The Shawshank Redemption",
                        "imdb_id": 111161,
                        "year": 1994,
                    },
                    "files": [
                        {
                            "file_id": 998877,
                            "cd_number": 1,
                            "file_name": "Shawshank.ar.srt",
                        }
                    ],
                },
            }
        ],
    }

    client = OpenSubtitlesClient()
    with patch.object(client, "_send_request_with_retry", new_callable=AsyncMock) as mock_send:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload
        mock_send.return_value = mock_resp

        movie_info, candidates = await client.search_subtitles(imdb_id="tt0111161")

        assert movie_info is not None
        assert movie_info.title == "The Shawshank Redemption"
        assert movie_info.imdb_id == "tt0111161"

        assert len(candidates) == 1
        cand = candidates[0]
        assert cand.language == "ar"
        assert cand.from_trusted is True
        assert cand.download_count == 3500
        assert len(cand.files) == 1
        assert cand.files[0].file_id == 998877


@pytest.mark.asyncio
async def test_get_available_languages_aggregation():
    """Verify languages are aggregated with total file counts and proper human names."""
    mock_payload = {
        "total_count": 3,
        "data": [
            {
                "id": "1",
                "attributes": {
                    "language": "ar",
                    "files": [{"file_id": 101, "file_name": "ar1.srt"}, {"file_id": 102, "file_name": "ar2.srt"}],
                },
            },
            {
                "id": "2",
                "attributes": {
                    "language": "en",
                    "files": [{"file_id": 201, "file_name": "en1.srt"}],
                },
            },
            {
                "id": "3",
                "attributes": {
                    "language": "ar",
                    "files": [{"file_id": 103, "file_name": "ar3.srt"}],
                },
            },
        ],
    }

    client = OpenSubtitlesClient()
    with patch.object(client, "_send_request_with_retry", new_callable=AsyncMock) as mock_send:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload
        mock_send.return_value = mock_resp

        movie_info, languages = await client.get_available_languages(imdb_id="tt0111161")

        assert len(languages) == 2
        # Arabic has 3 total files (2 from item 1 + 1 from item 3)
        ar_item = next(l for l in languages if l.language_code == "ar")
        assert ar_item.available_files == 3
        assert ar_item.language_name == "Arabic"

        # English has 1 file
        en_item = next(l for l in languages if l.language_code == "en")
        assert en_item.available_files == 1
        assert en_item.language_name == "English"


@pytest.mark.asyncio
async def test_request_download_link_and_fetch():
    """Verify download link acquisition and subtitle byte fetching."""
    client = OpenSubtitlesClient()

    with patch("app.services.opensubtitles.auth_manager.get_valid_token", new_callable=AsyncMock) as mock_auth, \
         patch.object(client, "_send_request_with_retry", new_callable=AsyncMock) as mock_send, \
         patch.object(client, "get_http_client") as mock_get_client:

        mock_auth.return_value = "token_abc"

        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {
            "link": "https://download.opensubtitles.com/sub/998877.srt",
            "file_name": "shawshank.srt",
            "requests": 1,
            "remaining": 99,
        }
        mock_send.return_value = mock_post_resp

        link, fname = await client.request_download_link(998877)
        assert link == "https://download.opensubtitles.com/sub/998877.srt"
        assert fname == "shawshank.srt"

        # Mock fetching the bytes
        mock_http = AsyncMock()
        mock_get_resp = MagicMock()
        mock_get_resp.status_code = 200
        mock_get_resp.content = SAMPLE_VALID_SRT.encode("utf-8")
        mock_http.get.return_value = mock_get_resp
        mock_get_client.return_value = mock_http

        content = await client.fetch_file_bytes(link)
        assert content == SAMPLE_VALID_SRT.encode("utf-8")


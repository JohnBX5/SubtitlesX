"""Unit tests for OpenSubtitles REST API v1 client and deterministic candidate ranking."""

import asyncio
import json

import httpx
import pytest

from app.core.config import Settings
from app.opensubtitles.client import OpenSubtitlesClient
from app.opensubtitles.models import SubtitleCandidate, SubtitleFile, TitleQuery, normalize_imdb_id
from app.opensubtitles.ranking import rank_candidates, select_top
from tests.conftest import SAMPLE_VALID_SRT


@pytest.mark.asyncio
async def test_auth_login_and_prevent_thundering_herd(test_settings: Settings):
    login_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal login_calls
        if request.url.path.endswith("/login"):
            login_calls += 1
            return httpx.Response(
                200,
                json={"token": "shared_token_abc", "user": {"user_id": 123, "level": "vip"}},
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = OpenSubtitlesClient(test_settings, transport=transport)
    try:
        tokens = await asyncio.gather(*(client._get_token() for _ in range(5)))
        assert all(t == "shared_token_abc" for t in tokens)
        assert login_calls == 1
    finally:
        await client.close()


def test_ranking_trusted_and_machine_translated():
    cand_trusted = SubtitleCandidate(
        id="1",
        subtitle_id="sub1",
        language="en",
        from_trusted=True,
        download_count=1000,
        ratings=8.0,
        votes=10,
        files=[SubtitleFile(file_id=1, file_name="sub1.srt")],
    )
    cand_untrusted = SubtitleCandidate(
        id="2",
        subtitle_id="sub2",
        language="en",
        from_trusted=False,
        download_count=1000,
        ratings=8.0,
        votes=10,
        files=[SubtitleFile(file_id=2, file_name="sub2.srt")],
    )
    cand_machine = SubtitleCandidate(
        id="3",
        subtitle_id="sub3",
        language="en",
        machine_translated=True,
        download_count=5000,
        files=[SubtitleFile(file_id=3, file_name="sub3.srt")],
    )

    ranked = rank_candidates([cand_machine, cand_untrusted, cand_trusted])
    assert [c.id for c in ranked] == ["1", "2", "3"]
    assert ranked[2].ranking_score < 0


def test_ranking_deterministic_tie_breaker():
    cand_a = SubtitleCandidate(
        id="a",
        subtitle_id="sub_a",
        language="en",
        download_count=100,
        ratings=5.0,
        files=[SubtitleFile(file_id=200, file_name="a.srt")],
    )
    cand_b = SubtitleCandidate(
        id="b",
        subtitle_id="sub_b",
        language="en",
        download_count=100,
        ratings=5.0,
        files=[SubtitleFile(file_id=100, file_name="b.srt")],
    )
    ranked1 = rank_candidates([cand_a, cand_b])
    ranked2 = rank_candidates([cand_b, cand_a])
    assert ranked1[0].files[0].file_id == 100
    assert ranked2[0].files[0].file_id == 100
    assert len(select_top([cand_a, cand_b], limit=1)) == 1


def test_normalize_imdb_id():
    assert normalize_imdb_id("tt0111161") == 111161
    assert normalize_imdb_id("0111161") == 111161
    assert normalize_imdb_id("111161") == 111161
    assert normalize_imdb_id(None) is None


@pytest.mark.asyncio
async def test_search_and_languages_and_download(test_settings: Settings):
    search_payload = {
        "total_pages": 1,
        "data": [
            {
                "id": "item_1",
                "attributes": {
                    "subtitle_id": "101",
                    "language": "ar",
                    "download_count": 3500,
                    "ratings": 8.5,
                    "votes": 25,
                    "hd": True,
                    "from_trusted": True,
                    "release": "The.Shawshank.Redemption.1994.1080p",
                    "feature_details": {"title": "The Shawshank Redemption", "imdb_id": 111161, "year": 1994},
                    "files": [
                        {"file_id": 1001, "cd_number": 1, "file_name": "ar1.srt"},
                        {"file_id": 1002, "cd_number": 2, "file_name": "ar2.srt"},
                    ],
                },
            },
            {
                "id": "item_2",
                "attributes": {
                    "subtitle_id": "102",
                    "language": "en",
                    "files": [{"file_id": 2001, "cd_number": 1, "file_name": "en1.srt"}],
                },
            },
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/subtitles"):
            return httpx.Response(200, json=search_payload)
        if request.url.path.endswith("/login"):
            return httpx.Response(200, json={"token": "tok_123", "user": {"level": "vip"}})
        if request.url.path.endswith("/download"):
            body = json.loads(request.content.decode("utf-8"))
            assert body["file_id"] == 2001
            return httpx.Response(
                200,
                json={
                    "link": "https://dl.opensubtitles.mock/sub/2001.srt",
                    "file_name": "shawshank.srt",
                    "remaining": 99,
                },
            )
        if request.url.host == "dl.opensubtitles.mock":
            return httpx.Response(200, content=SAMPLE_VALID_SRT.encode("utf-8"))
        return httpx.Response(404)

    client = OpenSubtitlesClient(test_settings, transport=httpx.MockTransport(handler))
    try:
        movie, langs = await client.languages(TitleQuery(imdb_id="tt0111161"))
        assert movie is not None and movie.title == "The Shawshank Redemption"
        ar = next(l for l in langs if l.language_code == "ar")
        en = next(l for l in langs if l.language_code == "en")
        assert ar.available_files == 2
        assert en.available_files == 1

        raw, name = await client.download_file(2001)
        assert name == "shawshank.srt"
        assert raw == SAMPLE_VALID_SRT.encode("utf-8")
        assert client.remaining_downloads == 99
    finally:
        await client.close()


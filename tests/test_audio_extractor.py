"""Unit tests for the AudioExtractor service."""

from unittest.mock import patch

import pytest

from app.core.config import Settings
from app.core.errors import AudioExtractionError, StreamURLRejected
from app.sync.audio import (
    AudioExtractor,
    AudioWindow,
    build_headers_arg,
    classify_ffmpeg_error,
    detect_source_type,
)


def test_detect_source_type():
    assert detect_source_type("https://example.com/live/playlist.m3u8?token=123") == "hls"
    assert detect_source_type("https://example.com/vod/stream.mpd") == "dash"
    assert detect_source_type("https://example.com/movies/film.mp4") == "direct_video"
    assert detect_source_type("https://example.com/audio/track.mp3") == "direct_audio"
    assert detect_source_type("https://example.com/audio/track.m4a") == "direct_audio"


def test_build_headers_arg():
    headers = {"user-agent": "ExoPlayer", "referer": "https://cinema-hub.org", "origin": "https://cinema-hub.org"}
    arg = build_headers_arg(headers)
    assert arg is not None
    assert "User-Agent:" not in arg  # passed separately via -user_agent
    assert "Referer: https://cinema-hub.org\r\n" in arg
    assert "Origin: https://cinema-hub.org\r\n" in arg


def test_classify_ffmpeg_error():
    assert classify_ffmpeg_error("Widevine DRM encrypted stream") is True
    assert classify_ffmpeg_error("Connection timed out") is False


@pytest.mark.asyncio
async def test_extract_audio_ssrf_rejection(test_settings: Settings):
    extractor = AudioExtractor(test_settings)
    with pytest.raises(StreamURLRejected):
        await extractor.extract("http://127.0.0.1:8080/stream.mp4")


@pytest.mark.asyncio
async def test_extract_audio_ffmpeg_error(test_settings: Settings):
    extractor = AudioExtractor(test_settings)
    with patch("app.sync.audio.validate_stream_url", return_value="https://example.com/video.mp4"), patch.object(
        extractor, "_run_blocking", return_value=(b"", "Invalid data found when processing input", 1, False)
    ):
        with pytest.raises(AudioExtractionError, match="FFmpeg could not decode audio") as exc_info:
            await extractor.extract("https://example.com/video.mp4", windows=[AudioWindow(0.0, 60.0)])
        assert exc_info.value.unsupported is True


@pytest.mark.asyncio
async def test_extract_audio_timeout(test_settings: Settings):
    extractor = AudioExtractor(test_settings)
    with patch("app.sync.audio.validate_stream_url", return_value="https://example.com/video.mp4"), patch.object(
        extractor, "_run_blocking", return_value=(b"\x00" * 100, "", -9, True)
    ):
        with pytest.raises(AudioExtractionError, match="timed out"):
            await extractor.extract("https://example.com/video.mp4", windows=[AudioWindow(0.0, 60.0)])


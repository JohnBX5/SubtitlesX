"""Unit tests for the Audio Extractor service."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.audio_extractor import AudioExtractionError, audio_extractor


def test_detect_source_type():
    """Verify source type identification for HLS, DASH, and direct video/audio."""
    assert audio_extractor.detect_source_type("https://example.com/live/playlist.m3u8?token=123") == "hls"
    assert audio_extractor.detect_source_type("https://example.com/vod/stream.mpd") == "dash"
    assert audio_extractor.detect_source_type("https://example.com/movies/film.mp4") == "direct_video"
    assert audio_extractor.detect_source_type("https://example.com/audio/track.mp3") == "direct_audio"
    assert audio_extractor.detect_source_type("https://example.com/audio/track.m4a") == "direct_audio"


def test_build_ffmpeg_headers_arg():
    """Verify header formatting into CRLF lines for FFmpeg."""
    headers = {"User-Agent": "ExoPlayer", "Referer": "https://cinema-hub.org"}
    arg = audio_extractor._build_ffmpeg_headers_arg(headers)
    assert arg is not None
    assert "User-Agent: ExoPlayer\r\n" in arg
    assert "Referer: https://cinema-hub.org\r\n" in arg


@pytest.mark.asyncio
async def test_extract_audio_ssrf_rejection():
    """Verify that attempting to extract audio from an internal IP raises AudioExtractionError."""
    with pytest.raises(AudioExtractionError, match="Security validation failed"):
        await audio_extractor.extract_audio_from_stream("http://127.0.0.1:8080/stream.mp4")


@pytest.mark.asyncio
async def test_extract_audio_ffmpeg_error():
    """Verify non-zero FFmpeg return code raises AudioExtractionError."""
    with patch("app.services.audio_extractor.validate_stream_url", return_value="https://example.com/video.mp4"), \
         patch("asyncio.create_subprocess_exec") as mock_exec:

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate.return_value = (b"", b"Invalid data found when processing input")
        mock_exec.return_value = mock_proc

        with pytest.raises(AudioExtractionError, match="FFmpeg failed"):
            await audio_extractor.extract_audio_from_stream("https://example.com/video.mp4")


@pytest.mark.asyncio
async def test_extract_audio_timeout():
    """Verify timeout during extraction kills process and raises AudioExtractionError."""
    with patch("app.services.audio_extractor.validate_stream_url", return_value="https://example.com/video.mp4"), \
         patch("asyncio.create_subprocess_exec") as mock_exec, \
         patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):

        mock_proc = AsyncMock()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()
        mock_exec.return_value = mock_proc

        with pytest.raises(AudioExtractionError, match="timed out"):
            await audio_extractor.extract_audio_from_stream("https://example.com/video.mp4")

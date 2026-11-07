"""Unit tests for subtitle validation, encoding normalization, and format detection."""

import pytest
from app.config import get_settings
from app.services.subtitle_manager import (
    SubtitleValidationError,
    subtitle_manager,
)
from tests.conftest import (
    SAMPLE_ARABIC_SRT,
    SAMPLE_CLOUDFLARE_HTML,
    SAMPLE_HTML_ERROR,
    SAMPLE_VALID_ASS,
    SAMPLE_VALID_SRT,
    SAMPLE_VALID_TTML,
    SAMPLE_VALID_VTT,
)


def test_validate_valid_srt():
    """Verify that a valid SRT file is parsed, validated, and normalized."""
    raw_bytes = SAMPLE_VALID_SRT.encode("utf-8")
    processed = subtitle_manager.process_subtitle(raw_bytes, filename_hint="movie.en.srt")

    assert processed.format == "srt"
    assert processed.cue_count == 3
    assert "Welcome to Cinema Hub!" in processed.content_str
    assert "text/plain" in processed.media_type_header


def test_validate_valid_vtt():
    """Verify that a valid WebVTT file is recognized and validated."""
    raw_bytes = SAMPLE_VALID_VTT.encode("utf-8")
    processed = subtitle_manager.process_subtitle(raw_bytes, filename_hint="movie.en.vtt")

    assert processed.format == "vtt"
    assert processed.cue_count == 2
    assert "WEBVTT" in processed.content_str
    assert "text/vtt" in processed.media_type_header


def test_validate_valid_ass():
    """Verify that an Advanced SubStation Alpha (.ass) file is recognized."""
    raw_bytes = SAMPLE_VALID_ASS.encode("utf-8")
    processed = subtitle_manager.process_subtitle(raw_bytes, filename_hint="anime.ass")

    assert processed.format == "ass"
    assert processed.cue_count == 2
    assert "[Script Info]" in processed.content_str


def test_validate_valid_ttml():
    """Verify that a valid TTML XML subtitle file is recognized."""
    raw_bytes = SAMPLE_VALID_TTML.encode("utf-8")
    processed = subtitle_manager.process_subtitle(raw_bytes, filename_hint="stream.ttml")

    assert processed.format == "ttml"
    assert processed.cue_count == 2


def test_reject_non_subtitle_xml():
    """Verify that generic XML (e.g. RSS/SVG) is rejected."""
    generic_xml = b'<?xml version="1.0"?><rss version="2.0"><channel><title>News</title></channel></rss>'
    with pytest.raises(SubtitleValidationError, match="not a recognized subtitle format"):
        subtitle_manager.process_subtitle(generic_xml, filename_hint="feed.xml")


def test_utf8_bom_removal():
    """Verify that UTF-8 BOM is cleanly removed and does not corrupt text."""
    bom_srt = b"\xef\xbb\xbf" + SAMPLE_VALID_SRT.encode("utf-8")
    processed = subtitle_manager.process_subtitle(bom_srt, filename_hint="test.srt")

    assert not processed.content_bytes.startswith(b"\xef\xbb\xbf")
    assert processed.content_str.startswith("1\n00:00:01,000") or processed.content_str.startswith("1\r\n00:00:01,000")


def test_arabic_windows_1256_encoding():
    """Verify that Arabic subtitles encoded in Windows-1256 are normalized to UTF-8 without corruption."""
    windows_1256_bytes = SAMPLE_ARABIC_SRT.encode("windows-1256")
    processed = subtitle_manager.process_subtitle(windows_1256_bytes, filename_hint="arabic.srt")

    assert processed.format == "srt"
    assert "مرحبا بكم في سينما هاب!" in processed.content_str
    assert "مشاهدة ممتعة للجميع." in processed.content_str
    assert processed.cue_count == 3


def test_reject_empty_file():
    """Verify that an empty file (0 bytes) raises SubtitleValidationError."""
    with pytest.raises(SubtitleValidationError, match="empty"):
        subtitle_manager.process_subtitle(b"", filename_hint="empty.srt")


def test_reject_cloudflare_challenge():
    """Verify that Cloudflare anti-bot challenge pages are identified and rejected."""
    raw_bytes = SAMPLE_CLOUDFLARE_HTML.encode("utf-8")
    with pytest.raises(SubtitleValidationError, match="HTML error, CAPTCHA, or Cloudflare"):
        subtitle_manager.process_subtitle(raw_bytes, filename_hint="movie.srt")


def test_reject_html_error_page():
    """Verify that 502/404 HTML response bodies are rejected."""
    raw_bytes = SAMPLE_HTML_ERROR.encode("utf-8")
    with pytest.raises(SubtitleValidationError, match="HTML error"):
        subtitle_manager.process_subtitle(raw_bytes, filename_hint="movie.srt")


def test_reject_binary_garbage():
    """Verify that binary executable/garbage data containing null bytes is rejected."""
    binary_data = b"\x00\x01\x02\x03\x00\x00\xff\xfe\x00\x10\x45\x67"
    with pytest.raises(SubtitleValidationError, match="binary null bytes|Unrecognized"):
        subtitle_manager.process_subtitle(binary_data, filename_hint="corrupt.srt")


def test_reject_missing_timestamps():
    """Verify that a plain text file without valid subtitle timestamps is rejected."""
    plain_text = b"This is just a regular text file with no timestamps or dialogue cues."
    with pytest.raises(SubtitleValidationError, match="Unrecognized subtitle format|lacks valid timestamp"):
        subtitle_manager.process_subtitle(plain_text, filename_hint="notes.txt")


def test_reject_oversized_file(monkeypatch):
    """Verify that files exceeding the maximum size limit are rejected."""
    settings = get_settings()
    monkeypatch.setattr(settings, "MAX_SUBTITLE_FILE_SIZE_MB", 1)

    # 1.5 MB dummy content with timestamps
    chunk = "1\n00:00:01,000 --> 00:00:04,000\nHello\n\n"
    large_srt = (chunk * 40000).encode("utf-8")

    with pytest.raises(SubtitleValidationError, match="exceeds maximum permitted size"):
        subtitle_manager.process_subtitle(large_srt, filename_hint="huge.srt")


def test_reject_excessive_cue_count(monkeypatch):
    """Verify that files exceeding the maximum cue count limit are rejected."""
    settings = get_settings()
    monkeypatch.setattr(settings, "MAX_SUBTITLE_CUE_COUNT", 5)

    chunk = "1\n00:00:01,000 --> 00:00:04,000\nHello\n\n"
    srt_with_10_cues = (chunk * 10).encode("utf-8")

    with pytest.raises(SubtitleValidationError, match="exceeds limit"):
        subtitle_manager.process_subtitle(srt_with_10_cues, filename_hint="many_cues.srt")


def test_sanitize_filename_prevents_path_traversal():
    """Verify filename sanitization prevents path traversal and dangerous characters."""
    unsafe_names = [
        ("../../etc/passwd", "etc_passwd.srt"),
        ("..\\..\\Windows\\System32\\cmd.exe", "Windows_System32_cmd.exe"),
        ("movie/title/../../sub.srt", "sub.srt"),
        ("test\x00file.srt", "testfile.srt"),
        ("   ...   ", "subtitle.srt"),
    ]

    for raw, expected in unsafe_names:
        sanitized = subtitle_manager.sanitize_filename(raw, fallback_ext="srt")
        assert "/" not in sanitized
        assert "\\" not in sanitized
        assert ".." not in sanitized
        assert "\x00" not in sanitized


"""Unit tests for subtitle validation, encoding normalization, and format detection."""

import pytest

from app.core.errors import SubtitleValidationError
from app.subtitles.processing import load_subtitle_bytes, sanitize_filename
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
    doc = load_subtitle_bytes(SAMPLE_VALID_SRT.encode("utf-8"), filename_hint="movie.en.srt")
    assert doc.source_format == "srt"
    assert doc.cue_count == 3
    assert "Welcome to Cinema Hub!" in doc.cues[0].text


def test_validate_valid_vtt():
    doc = load_subtitle_bytes(SAMPLE_VALID_VTT.encode("utf-8"), filename_hint="movie.en.vtt")
    assert doc.source_format == "vtt"
    assert doc.cue_count == 2


def test_validate_valid_ass():
    doc = load_subtitle_bytes(SAMPLE_VALID_ASS.encode("utf-8"), filename_hint="anime.ass")
    assert doc.source_format == "ass"
    assert doc.cue_count == 2
    assert doc.cues[0].text == "Hello world from ASS!"
    assert "<b>Stylized line</b>" in doc.cues[1].text


def test_validate_valid_ttml():
    doc = load_subtitle_bytes(SAMPLE_VALID_TTML.encode("utf-8"), filename_hint="stream.ttml")
    assert doc.source_format == "ttml"
    assert doc.cue_count == 2


def test_reject_non_subtitle_xml():
    generic_xml = b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"><rect width="10" height="10"/></svg>'
    with pytest.raises(SubtitleValidationError, match="not a recognized subtitle format"):
        load_subtitle_bytes(generic_xml, filename_hint="image.svg")


def test_utf8_bom_removal():
    bom_srt = b"\xef\xbb\xbf" + SAMPLE_VALID_SRT.encode("utf-8")
    doc = load_subtitle_bytes(bom_srt, filename_hint="test.srt")
    assert doc.encoding == "utf-8-sig"
    assert not doc.cues[0].text.startswith("\ufeff")


def test_arabic_windows_1256_encoding():
    windows_1256_bytes = SAMPLE_ARABIC_SRT.encode("windows-1256")
    doc = load_subtitle_bytes(windows_1256_bytes, language="ar", filename_hint="arabic.srt")
    assert doc.source_format == "srt"
    assert doc.cue_count == 3
    assert "مرحبا بكم في سينما هاب!" in doc.cues[0].text
    assert "مشاهدة ممتعة للجميع." in doc.cues[2].text


def test_reject_empty_file():
    with pytest.raises(SubtitleValidationError, match="empty"):
        load_subtitle_bytes(b"", filename_hint="empty.srt")


def test_reject_cloudflare_challenge():
    with pytest.raises(SubtitleValidationError, match="HTML error, CAPTCHA, or Cloudflare"):
        load_subtitle_bytes(SAMPLE_CLOUDFLARE_HTML.encode("utf-8"), filename_hint="movie.srt")


def test_reject_html_error_page():
    with pytest.raises(SubtitleValidationError, match="HTML error"):
        load_subtitle_bytes(SAMPLE_HTML_ERROR.encode("utf-8"), filename_hint="movie.srt")


def test_reject_binary_garbage():
    binary_data = b"\x00\x01\x02\x03\x00\x00\xff\xfe\x00\x10\x45\x67"
    with pytest.raises(SubtitleValidationError, match="binary NUL bytes|Unrecognized"):
        load_subtitle_bytes(binary_data, filename_hint="corrupt.srt")


def test_reject_missing_timestamps():
    plain_text = b"This is just a regular text file with no timestamps or dialogue cues."
    with pytest.raises(SubtitleValidationError, match="Unrecognized subtitle format|lacks valid timestamp"):
        load_subtitle_bytes(plain_text, filename_hint="notes.txt")


def test_reject_oversized_file():
    chunk = "1\n00:00:01,000 --> 00:00:04,000\nHello\n\n"
    large_srt = (chunk * 40000).encode("utf-8")
    with pytest.raises(SubtitleValidationError, match="exceeds maximum permitted size"):
        load_subtitle_bytes(large_srt, filename_hint="huge.srt", max_bytes=1024 * 1024)


def test_reject_excessive_cue_count():
    chunk = "1\n00:00:01,000 --> 00:00:04,000\nHello\n\n"
    srt_with_10_cues = (chunk * 10).encode("utf-8")
    with pytest.raises(SubtitleValidationError, match="exceeds limit"):
        load_subtitle_bytes(srt_with_10_cues, filename_hint="many_cues.srt", max_cues=5)


def test_sanitize_filename_prevents_path_traversal():
    unsafe_names = [
        "../../etc/passwd",
        "..\\..\\Windows\\System32\\cmd.exe",
        "movie/title/../../sub.srt",
        "test\x00file.srt",
        "   ...   ",
    ]
    for raw in unsafe_names:
        sanitized = sanitize_filename(raw, fallback_ext="srt")
        assert "/" not in sanitized
        assert "\\" not in sanitized
        assert ".." not in sanitized
        assert "\x00" not in sanitized

"""Unit tests for subtitle format conversion and rendering."""

from app.subtitles.processing import load_subtitle_bytes, render_document
from tests.conftest import (
    SAMPLE_ARABIC_SRT,
    SAMPLE_VALID_ASS,
    SAMPLE_VALID_SRT,
    SAMPLE_VALID_VTT,
)


def test_srt_to_vtt_conversion_structure():
    doc = load_subtitle_bytes(SAMPLE_VALID_SRT.encode("utf-8"), filename_hint="movie.srt")
    rendered = render_document(doc, "vtt")

    assert rendered.format == "vtt"
    assert rendered.cue_count == 3
    assert rendered.media_type == "text/vtt; charset=utf-8"

    lines = rendered.content.splitlines()
    assert lines[0] == "WEBVTT"
    assert "00:00:01.000 --> 00:00:04.000" in rendered.content
    assert "00:00:05.500 --> 00:00:08.200" in rendered.content
    assert "00:00:09.000 --> 00:00:12.000" in rendered.content
    assert "<i>Enjoy high quality streaming with subtitles.</i>" in rendered.content
    assert "Multi-line subtitle text" in rendered.content
    assert "Line two of dialogue." in rendered.content


def test_srt_to_vtt_conversion_arabic_preservation():
    doc = load_subtitle_bytes(SAMPLE_ARABIC_SRT.encode("utf-8"), language="ar", filename_hint="arabic.srt")
    rendered = render_document(doc, "vtt")

    assert rendered.format == "vtt"
    assert "WEBVTT" in rendered.content
    assert "مرحبا بكم في سينما هاب!" in rendered.content
    assert "هذا ملف ترجمة باللغة العربية مع علامات ترقيم كاملة." in rendered.content
    assert "00:00:01.000 --> 00:00:04.500" in rendered.content


def test_ass_to_vtt_and_srt_rendering():
    doc = load_subtitle_bytes(SAMPLE_VALID_ASS.encode("utf-8"), filename_hint="anime.ass")
    rendered = render_document(doc, "vtt")
    assert rendered.format == "vtt"
    assert "Hello world from ASS!" in rendered.content
    assert "<b>Stylized line</b>" in rendered.content


def test_vtt_to_vtt_no_op():
    doc = load_subtitle_bytes(SAMPLE_VALID_VTT.encode("utf-8"), filename_hint="movie.vtt")
    rendered = render_document(doc, "vtt")
    assert rendered.format == "vtt"
    assert rendered.cue_count == 2
    assert "Welcome to Cinema Hub WebVTT!" in rendered.content


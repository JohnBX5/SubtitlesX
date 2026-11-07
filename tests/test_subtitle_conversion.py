"""Unit tests for subtitle format conversion."""

import pytest
from app.services.subtitle_manager import (
    SubtitleConversionError,
    subtitle_manager,
)
from tests.conftest import (
    SAMPLE_ARABIC_SRT,
    SAMPLE_VALID_ASS,
    SAMPLE_VALID_SRT,
    SAMPLE_VALID_VTT,
)


def test_srt_to_vtt_conversion_structure():
    """Verify SRT conversion to WebVTT preserves cues and transforms timestamps correctly."""
    raw_bytes = SAMPLE_VALID_SRT.encode("utf-8")
    processed = subtitle_manager.process_subtitle(
        raw_bytes,
        filename_hint="movie.srt",
        target_format="vtt",
    )

    assert processed.format == "vtt"
    assert processed.cue_count == 3
    assert processed.media_type_header == "text/vtt; charset=utf-8"

    lines = processed.content_str.splitlines()
    assert lines[0] == "WEBVTT"

    # Verify timestamps have periods instead of commas
    assert "00:00:01.000 --> 00:00:04.000" in processed.content_str
    assert "00:00:05.500 --> 00:00:08.200" in processed.content_str
    assert "00:00:09.000 --> 00:00:12.000" in processed.content_str
    assert "," not in [l for l in lines if "-->" in l][0]

    # Verify inline formatting is preserved
    assert "<i>Enjoy high quality streaming with subtitles.</i>" in processed.content_str
    assert "Multi-line subtitle text" in processed.content_str
    assert "Line two of dialogue." in processed.content_str


def test_srt_to_vtt_conversion_arabic_preservation():
    """Verify Arabic characters and right-to-left punctuation are preserved during conversion."""
    raw_bytes = SAMPLE_ARABIC_SRT.encode("utf-8")
    processed = subtitle_manager.process_subtitle(
        raw_bytes,
        filename_hint="arabic.srt",
        target_format="vtt",
    )

    assert processed.format == "vtt"
    assert "WEBVTT" in processed.content_str
    assert "مرحبا بكم في سينما هاب!" in processed.content_str
    assert "هذا ملف ترجمة باللغة العربية مع علامات ترقيم كاملة." in processed.content_str
    assert "00:00:01.000 --> 00:00:04.500" in processed.content_str


def test_refuse_ass_to_vtt_conversion():
    """Verify that attempting to convert ASS/SSA to WebVTT is explicitly refused to preserve styling."""
    raw_bytes = SAMPLE_VALID_ASS.encode("utf-8")
    with pytest.raises(SubtitleConversionError, match="Refusing to convert ASS to VTT"):
        subtitle_manager.process_subtitle(
            raw_bytes,
            filename_hint="anime.ass",
            target_format="vtt",
        )


def test_vtt_to_vtt_no_op():
    """Verify that requesting VTT when input is already VTT works without issues."""
    raw_bytes = SAMPLE_VALID_VTT.encode("utf-8")
    processed = subtitle_manager.process_subtitle(
        raw_bytes,
        filename_hint="movie.vtt",
        target_format="vtt",
    )

    assert processed.format == "vtt"
    assert processed.cue_count == 2
    assert "Welcome to Cinema Hub WebVTT!" in processed.content_str


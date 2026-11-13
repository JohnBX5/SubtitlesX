"""Subtitle domain: parsing, normalization, rendering."""

from app.subtitles.cleaner import clean_text, is_ad, is_dialogue, normalize_cues, plain_text, spoken_text
from app.subtitles.encoding import decode_subtitle_bytes
from app.subtitles.models import Cue, RenderedSubtitle, SubtitleDocument
from app.subtitles.parser import detect_format, parse_text
from app.subtitles.processing import (
    SUPPORTED_OUTPUT_FORMATS,
    build_download_filename,
    load_subtitle_bytes,
    load_subtitle_text,
    render_document,
    sanitize_filename,
)
from app.subtitles.writer import format_timestamp, render, render_srt, render_vtt

__all__ = [
    "Cue",
    "RenderedSubtitle",
    "SUPPORTED_OUTPUT_FORMATS",
    "SubtitleDocument",
    "build_download_filename",
    "clean_text",
    "decode_subtitle_bytes",
    "detect_format",
    "format_timestamp",
    "is_ad",
    "is_dialogue",
    "load_subtitle_bytes",
    "load_subtitle_text",
    "normalize_cues",
    "parse_text",
    "plain_text",
    "render",
    "render_document",
    "render_srt",
    "render_vtt",
    "sanitize_filename",
    "spoken_text",
]

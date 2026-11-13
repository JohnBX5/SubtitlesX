"""High-level subtitle pipeline: bytes -> validated document -> rendered output."""

from __future__ import annotations

import re
from pathlib import PurePath
from typing import Optional

from app.core.errors import SubtitleValidationError
from app.subtitles.cleaner import normalize_cues
from app.subtitles.encoding import decode_subtitle_bytes
from app.subtitles.models import RenderedSubtitle, SubtitleDocument
from app.subtitles.parser import parse_text
from app.subtitles.writer import render

SUPPORTED_OUTPUT_FORMATS = ("srt", "vtt")


def load_subtitle_text(
    text: str,
    *,
    filename_hint: str = "",
    encoding: str = "utf-8",
    max_cues: int = 50000,
    strip_ads: bool = True,
) -> SubtitleDocument:
    if "\x00" in text[:4096]:
        raise SubtitleValidationError("File contains binary NUL bytes; not a text subtitle.")
    fmt, cues = parse_text(text, filename_hint)
    if not cues:
        raise SubtitleValidationError(f"Subtitle file contains 0 valid dialogue cues (format '{fmt}').")
    if len(cues) > max_cues:
        raise SubtitleValidationError(f"Subtitle cue count ({len(cues)}) exceeds limit of {max_cues}.")
    cues, removed = normalize_cues(cues, strip_ads=strip_ads)
    if not cues:
        raise SubtitleValidationError("Subtitle file contains no displayable cues after normalization.")
    return SubtitleDocument(cues=cues, source_format=fmt, encoding=encoding, removed_ads=removed)


def load_subtitle_bytes(
    raw: bytes,
    *,
    language: Optional[str] = None,
    filename_hint: str = "",
    max_bytes: int = 10 * 1024 * 1024,
    max_cues: int = 50000,
    strip_ads: bool = True,
) -> SubtitleDocument:
    if not raw:
        raise SubtitleValidationError("Subtitle file is empty (0 bytes).")
    if len(raw) > max_bytes:
        raise SubtitleValidationError(
            f"Subtitle file exceeds maximum permitted size of {max_bytes // (1024 * 1024)} MB."
        )
    text, encoding = decode_subtitle_bytes(raw, language)
    return load_subtitle_text(
        text, filename_hint=filename_hint, encoding=encoding, max_cues=max_cues, strip_ads=strip_ads
    )


def render_document(doc: SubtitleDocument, fmt: str) -> RenderedSubtitle:
    fmt = (fmt or "srt").lower()
    if fmt not in SUPPORTED_OUTPUT_FORMATS:
        raise SubtitleValidationError(f"Unsupported output format '{fmt}'. Use 'srt' or 'vtt'.")
    return render(doc.cues, fmt)


def sanitize_filename(filename: str, fallback_ext: str = "srt") -> str:
    """Make a filename safe for Content-Disposition (no traversal, ASCII-safe)."""
    clean = PurePath(filename.replace("\\", "/")).name.replace("\x00", "")
    clean = re.sub(r"[^A-Za-z0-9._\-]", "_", clean)
    clean = re.sub(r"\.{2,}", ".", clean)
    clean = re.sub(r"_{2,}", "_", clean).strip("._")
    if not clean or "." not in clean:
        clean = f"{clean or 'subtitle'}.{fallback_ext}"
    if len(clean) > 200:
        stem, _, ext = clean.rpartition(".")
        clean = f"{stem[:190]}.{ext}"
    return clean


def build_download_filename(title: Optional[str], language: str, fmt: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", ".", (title or "").strip()).strip(".") or "subtitle"
    return sanitize_filename(f"{slug}.{language}.{fmt}", fallback_ext=fmt)


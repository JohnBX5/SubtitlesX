"""Renderers for the two formats delivered to ExoPlayer: SubRip and WebVTT."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence

from app.subtitles.models import Cue, RenderedSubtitle

_AN_TAG = re.compile(r"^\{\\an([1-9])\}")
_ASS_TAGS = re.compile(r"\{\\[^}]*\}")
_FONT_TAG = re.compile(r"</?font[^>]*>", re.IGNORECASE)
_ALLOWED_VTT_TAG = re.compile(r"</?(?:i|b|u)>", re.IGNORECASE)


def format_timestamp(seconds: float, sep: str = ",") -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _monotonic(cues: Sequence[Cue]) -> Iterable[tuple[float, float, str]]:
    """Yield safe timings: non-negative, end > start."""
    for cue in cues:
        start = max(0.0, cue.start)
        end = max(start + 0.05, cue.end)
        yield start, end, cue.text


def render_srt(cues: Sequence[Cue]) -> str:
    blocks: List[str] = []
    for idx, (start, end, text) in enumerate(_monotonic(cues), start=1):
        # ExoPlayer's SubripDecoder understands a leading {\anN}; drop other ASS overrides.
        m = _AN_TAG.match(text)
        body = _ASS_TAGS.sub("", text[m.end():] if m else text).strip()
        if m:
            body = f"{{\\an{m.group(1)}}}{body}"
        blocks.append(f"{idx}\n{format_timestamp(start)} --> {format_timestamp(end)}\n{body}")
    return "\n\n".join(blocks) + "\n" if blocks else ""


def _vtt_escape(text: str) -> str:
    """Escape &, <, > except the inline tags WebVTT supports."""
    out: List[str] = []
    pos = 0
    for m in _ALLOWED_VTT_TAG.finditer(text):
        out.append(text[pos:m.start()].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        out.append(m.group(0).lower())
        pos = m.end()
    out.append(text[pos:].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    return "".join(out)


_VTT_POSITION = {
    "7": " line:0 align:start position:10%",
    "8": " line:0",
    "9": " line:0 align:end position:90%",
    "4": " line:50% align:start position:10%",
    "5": " line:50%",
    "6": " line:50% align:end position:90%",
    "1": " align:start position:10%",
    "3": " align:end position:90%",
}


def render_vtt(cues: Sequence[Cue]) -> str:
    blocks: List[str] = ["WEBVTT"]
    for idx, (start, end, text) in enumerate(_monotonic(cues), start=1):
        settings = ""
        m = _AN_TAG.match(text)
        if m:
            settings = _VTT_POSITION.get(m.group(1), "")
            text = text[m.end():]
        text = _FONT_TAG.sub("", _ASS_TAGS.sub("", text)).replace("-->", "\u2192").strip()
        if not text:
            continue
        blocks.append(
            f"{idx}\n{format_timestamp(start, '.')} --> {format_timestamp(end, '.')}{settings}\n{_vtt_escape(text)}"
        )
    return "\n\n".join(blocks) + "\n"


def render(cues: Sequence[Cue], fmt: str) -> RenderedSubtitle:
    fmt = (fmt or "srt").lower()
    if fmt == "vtt":
        return RenderedSubtitle(render_vtt(cues), "vtt", len(cues))
    return RenderedSubtitle(render_srt(cues), "srt", len(cues))


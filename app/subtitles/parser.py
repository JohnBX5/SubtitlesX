"""Format detection and tolerant parsers for SRT, WebVTT, ASS/SSA and TTML/DFXP."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import List, Optional

from app.core.errors import SubtitleValidationError
from app.subtitles.models import Cue

# --------------------------------------------------------------------------
# Rejection of non-subtitle payloads (HTML error pages, anti-bot challenges)
# --------------------------------------------------------------------------
_HTML_SIGNATURES = [
    re.compile(p, re.IGNORECASE | re.DOTALL)
    for p in (
        # NOTE: <head>/<body> are deliberately absent: TTML documents use them too.
        r"<!DOCTYPE\s+html",
        r"<html[\s>]",
        r"cf-browser-verification",
        r"challenge-platform",
        r"Attention Required!\s*\|\s*Cloudflare",
        r"<title>[^<]{0,200}</title>",
    )
]


def reject_html(text: str) -> None:
    sample = text[:4096]
    if any(sig.search(sample) for sig in _HTML_SIGNATURES):
        raise SubtitleValidationError(
            "File content identified as an HTML error, CAPTCHA, or Cloudflare challenge page."
        )


# --------------------------------------------------------------------------
# Timestamps
# --------------------------------------------------------------------------
_SRT_TIMING = re.compile(
    r"(?P<s>(?:\d{1,3}:)?\d{1,2}:\d{1,2}(?:[,.:]\d{1,3})?)\s*-{1,2}>\s*"
    r"(?P<e>(?:\d{1,3}:)?\d{1,2}:\d{1,2}(?:[,.:]\d{1,3})?)(?P<settings>[^\n]*)"
)
_INDEX_LINE = re.compile(r"^\s*\d+\s*$")


def parse_clock(value: str) -> float:
    """Parse ``[hh:]mm:ss[,.:]mmm`` into seconds."""
    value = value.strip()
    frac = 0.0
    m = re.match(r"^(.*?)[,.](\d{1,3})$", value)
    if m:
        value, ms = m.group(1), m.group(2)
        frac = int(ms) / (10 ** len(ms))
    else:
        # Rare "hh:mm:ss:mmm" variant
        m = re.match(r"^(\d{1,3}:\d{1,2}:\d{1,2}):(\d{3})$", value)
        if m:
            value, frac = m.group(1), int(m.group(2)) / 1000.0
    parts = [int(p) for p in value.split(":")]
    if len(parts) == 2:
        h, mi, s = 0, parts[0], parts[1]
    elif len(parts) == 3:
        h, mi, s = parts
    else:
        raise ValueError(f"bad timestamp {value!r}")
    return h * 3600 + mi * 60 + s + frac


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
def detect_format(text: str, filename_hint: str = "") -> str:
    stripped = text.lstrip("\ufeff \t\r\n")
    head = stripped[:2048]
    if re.match(r"^WEBVTT", stripped):
        return "vtt"
    if "[Script Info]" in head or re.search(r"^\[Events\]", stripped, re.MULTILINE) or (
        "Format:" in stripped and re.search(r"^Dialogue:", stripped, re.MULTILINE)
    ):
        return "ass" if ("v4.00+" in head or "[V4+ Styles]" in stripped) else "ssa"
    if head.startswith("<?xml") or re.search(r"<tt[\s>]", head) or "<timedtext" in head.lower():
        lowered = head.lower()
        if "ttml" in lowered or "<tt" in lowered or "begin=" in stripped[:20000]:
            return "dfxp" if "dfxp" in filename_hint.lower() else "ttml"
        raise SubtitleValidationError("XML document is not a recognized subtitle format (TTML/DFXP).")
    if _SRT_TIMING.search(stripped[:20000]):
        return "srt"
    raise SubtitleValidationError(
        f"Unrecognized subtitle format: content lacks valid timestamp cues (hint: '{filename_hint}')."
    )


# --------------------------------------------------------------------------
# SRT (and timing-line based fallback)
# --------------------------------------------------------------------------
def parse_srt(text: str) -> List[Cue]:
    """Line-based state machine tolerant of missing indices / blank lines."""
    cues: List[Cue] = []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cur: Optional[Cue] = None
    buf: List[str] = []

    def flush() -> None:
        nonlocal cur, buf
        if cur is not None:
            while buf and not buf[-1].strip():
                buf.pop()
            cur.text = "\n".join(line.rstrip() for line in buf).strip("\n")
            cues.append(cur)
        cur, buf = None, []

    for line in lines:
        m = _SRT_TIMING.search(line)
        if m:
            # Drop a trailing cue index that belongs to the new cue.
            if buf and _INDEX_LINE.match(buf[-1]):
                buf.pop()
            flush()
            try:
                start, end = parse_clock(m.group("s")), parse_clock(m.group("e"))
            except ValueError:
                continue
            cur = Cue(start, end, "")
            continue
        if cur is not None:
            buf.append(line)
    flush()
    return cues


# --------------------------------------------------------------------------
# WebVTT
# --------------------------------------------------------------------------
_VTT_LINE_SETTING = re.compile(r"\bline:(\d+(?:\.\d+)?)%?")


def parse_vtt(text: str) -> List[Cue]:
    body = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n[ \t]*\n", body)
    cues: List[Cue] = []
    for block in blocks:
        lines = [ln for ln in block.split("\n")]
        while lines and not lines[0].strip():
            lines.pop(0)
        if not lines:
            continue
        first = lines[0].strip()
        if first.startswith(("WEBVTT", "NOTE", "STYLE", "REGION")) and "-->" not in first:
            continue
        idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if idx is None:
            continue
        m = _SRT_TIMING.search(lines[idx])
        if not m:
            continue
        try:
            start, end = parse_clock(m.group("s")), parse_clock(m.group("e"))
        except ValueError:
            continue
        cue_text = "\n".join(lines[idx + 1 :]).strip("\n")
        settings = m.group("settings") or ""
        lm = _VTT_LINE_SETTING.search(settings)
        if lm:
            value = float(lm.group(1))
            is_percent = f"line:{lm.group(1)}%" in settings
            if (is_percent and value < 30) or (not is_percent and value <= 2):
                cue_text = "{\\an8}" + cue_text
        cue_text = re.sub(r"</?(?:c|v|lang|ruby|rt)(?:[.\s][^>]*)?>", "", cue_text)
        cues.append(Cue(start, end, cue_text))
    return cues


# --------------------------------------------------------------------------
# ASS / SSA
# --------------------------------------------------------------------------
def _ass_time(value: str) -> float:
    h, m, s = value.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def ass_text_to_markup(raw: str) -> str:
    text = raw.replace("\\N", "\n").replace("\\n", " ").replace("\\h", " ")
    an = re.search(r"\{[^}]*\\an([1-9])[^}]*\}", text)
    a_legacy = re.search(r"\{[^}]*\\a(\d{1,2})[^}]*\}", text) if not an else None
    out: List[str] = []
    open_tags: List[str] = []
    pos = 0
    for m in re.finditer(r"\{([^}]*)\}", text):
        out.append(text[pos : m.start()])
        pos = m.end()
        for tag, val in re.findall(r"\\([ibu])(\d)", m.group(1)):
            if val == "1" and tag not in open_tags:
                out.append(f"<{tag}>")
                open_tags.append(tag)
            elif val == "0" and tag in open_tags:
                out.append(f"</{tag}>")
                open_tags.remove(tag)
    out.append(text[pos:])
    for tag in reversed(open_tags):
        out.append(f"</{tag}>")
    result = "".join(out).strip()
    position = None
    if an:
        position = an.group(1)
    elif a_legacy and a_legacy.group(1) in {"5", "6", "7"}:
        position = "8"
    if position and position in {"7", "8", "9"}:
        result = f"{{\\an{position}}}" + result
    return result


def parse_ass(text: str) -> List[Cue]:
    cues: List[Cue] = []
    fields: Optional[List[str]] = None
    in_events = False
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            in_events = s.lower() == "[events]"
            continue
        if not in_events:
            continue
        if s.lower().startswith("format:"):
            fields = [f.strip().lower() for f in s.split(":", 1)[1].split(",")]
            continue
        if not s.lower().startswith("dialogue:"):
            continue
        cols = fields or ["layer", "start", "end", "style", "name", "marginl", "marginr", "marginv", "effect", "text"]
        parts = s.split(":", 1)[1].split(",", len(cols) - 1)
        if len(parts) < len(cols):
            continue
        rec = dict(zip(cols, (p.strip() for p in parts)))
        raw_text = rec.get("text", "")
        if re.search(r"\\p[1-9]", raw_text):  # vector drawing, not dialogue
            continue
        try:
            start, end = _ass_time(rec["start"]), _ass_time(rec["end"])
        except (KeyError, ValueError):
            continue
        cue_text = ass_text_to_markup(raw_text)
        if cue_text:
            cues.append(Cue(start, end, cue_text))
    cues.sort(key=lambda c: (c.start, c.end))
    return cues


# --------------------------------------------------------------------------
# TTML / DFXP
# --------------------------------------------------------------------------
_TTML_OFFSET = re.compile(r"^(\d+(?:\.\d+)?)(h|m|s|ms|f|t)$")


def _ttml_time(value: str, frame_rate: float, tick_rate: float) -> float:
    value = value.strip()
    m = _TTML_OFFSET.match(value)
    if m:
        num, unit = float(m.group(1)), m.group(2)
        return {
            "h": num * 3600,
            "m": num * 60,
            "s": num,
            "ms": num / 1000.0,
            "f": num / frame_rate,
            "t": num / tick_rate,
        }[unit]
    parts = value.split(":")
    if len(parts) == 4:  # hh:mm:ss:frames
        h, mi, s, f = parts
        return int(h) * 3600 + int(mi) * 60 + float(s) + float(f) / frame_rate
    return parse_clock(value)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _ttml_text(elem: ET.Element) -> str:
    out: List[str] = []

    def walk(node: ET.Element) -> None:
        name = _local(node.tag)
        if name == "br":
            out.append("\n")
        else:
            italic = any(_local(k) == "fontStyle" and v == "italic" for k, v in node.attrib.items())
            bold = any(_local(k) == "fontWeight" and v == "bold" for k, v in node.attrib.items())
            if italic:
                out.append("<i>")
            if bold:
                out.append("<b>")
            if node.text:
                out.append(node.text)
            for child in node:
                walk(child)
            if bold:
                out.append("</b>")
            if italic:
                out.append("</i>")
        if node is not elem and node.tail:
            out.append(node.tail)

    walk(elem)
    text = "".join(out)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def parse_ttml(text: str) -> List[Cue]:
    if re.search(r"<!ENTITY", text[:10000], re.IGNORECASE):
        raise SubtitleValidationError("XML entity declarations are not allowed in subtitle files.")
    try:
        root = ET.fromstring(text.encode("utf-8"))
    except ET.ParseError as exc:
        raise SubtitleValidationError(f"Malformed TTML document: {exc}") from exc

    frame_rate, tick_rate = 30.0, 1.0
    for k, v in root.attrib.items():
        if _local(k) == "frameRate":
            try:
                frame_rate = float(v)
            except ValueError:
                pass
        elif _local(k) == "tickRate":
            try:
                tick_rate = float(v)
            except ValueError:
                pass

    cues: List[Cue] = []
    for p in root.iter():
        if _local(p.tag) != "p":
            continue
        attrs = {_local(k): v for k, v in p.attrib.items()}
        if "begin" not in attrs:
            continue
        try:
            start = _ttml_time(attrs["begin"], frame_rate, tick_rate)
            if "end" in attrs:
                end = _ttml_time(attrs["end"], frame_rate, tick_rate)
            elif "dur" in attrs:
                end = start + _ttml_time(attrs["dur"], frame_rate, tick_rate)
            else:
                end = start + 2.0
        except (ValueError, KeyError):
            continue
        body = _ttml_text(p)
        if body:
            cues.append(Cue(start, end, body))
    cues.sort(key=lambda c: (c.start, c.end))
    return cues


PARSERS = {
    "srt": parse_srt,
    "vtt": parse_vtt,
    "ass": parse_ass,
    "ssa": parse_ass,
    "ttml": parse_ttml,
    "dfxp": parse_ttml,
}


def parse_text(text: str, filename_hint: str = "") -> tuple[str, List[Cue]]:
    reject_html(text)
    fmt = detect_format(text, filename_hint)
    cues = PARSERS[fmt](text)
    if not cues and fmt != "srt" and _SRT_TIMING.search(text):
        cues = parse_srt(text)
    return fmt, cues

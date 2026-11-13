"""Cue normalization: text hygiene, timing repair, ad removal, dialogue detection.

Arabic and other RTL scripts are preserved verbatim: only C0/C1 control
characters are removed, never Unicode format characters such as RLM/LRM
(U+200E/U+200F) or the bidi embedding controls (U+202A..U+202E, U+2066..U+2069).
"""

from __future__ import annotations

import re
import unicodedata
from typing import List

from app.subtitles.models import Cue

# C0 controls except \t and \n, DEL, and C1 controls.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_MULTISPACE_RE = re.compile(r"[ \t\u00a0]{2,}")
_EMPTY_TAGS_RE = re.compile(r"<(i|b|u)>\s*</\1>", re.IGNORECASE)
_ANY_TAG_RE = re.compile(r"<[^>]{0,200}>|\{\\[^}]{0,200}\}")

# Injected advertising / release-group watermarks. A cue is only dropped when
# it *is* an ad, never because a word appears inside real dialogue.
_AD_PATTERNS = re.compile(
    r"(opensubtitles\.(?:org|com)|osdb\.link|subscene\.com|yts\.(?:mx|lt|am|ag)|yify|addic7ed\.com|"
    r"podnapisi\.net|subdl\.com|subsource\.net|"
    r"advertise\s+your\s+product|become\s+(?:a\s+)?vip\s+member|support\s+us\s+and\s+become|"
    r"remove\s+all\s+ads|api\.opensubtitles|www\.[a-z0-9\-]+\.[a-z]{2,}|https?://\S+)",
    re.IGNORECASE,
)

# Non-spoken annotations used to decide whether a cue is real dialogue.
_SDH_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)|（[^）]*）|♪[^♪]*♪?")
_MUSIC_RE = re.compile(r"[♪♫♬♩#]+")
_SPEAKER_RE = re.compile(r"^\s*[-–—]?\s*[A-Z0-9\u0600-\u06FF _.'-]{1,24}:\s*", re.MULTILINE)
_CREDIT_RE = re.compile(
    r"(synced?\s+(?:and|&)?\s*corrected|subtitles?\s+by|translated\s+by|encoded\s+by|ripped\s+by|"
    r"sync(?:ed)?\s+by|ترجمة\s*[:\-]|تعديل\s+(?:و\s*)?(?:مزامنة|توقيت)|تابعونا\s+على|ترجمة\s+و\s*تعديل)",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def clean_text(text: str) -> str:
    """Normalize cue text without touching script/bidi characters."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
    text = _CONTROL_RE.sub("", text)
    lines = [_MULTISPACE_RE.sub(" ", ln).strip() for ln in text.split("\n")]
    text = "\n".join(ln for ln in lines if ln)
    prev = None
    while prev != text:
        prev = text
        text = _EMPTY_TAGS_RE.sub("", text)
    return text.strip()


def plain_text(text: str) -> str:
    """Text with markup removed (for analysis, not output)."""
    return _ANY_TAG_RE.sub("", text).strip()


def is_ad(text: str) -> bool:
    return bool(_AD_PATTERNS.search(plain_text(text)))


def spoken_text(text: str) -> str:
    """Return the part of a cue that is actually spoken (empty if none)."""
    raw = plain_text(text)
    if not raw or _AD_PATTERNS.search(raw) or _CREDIT_RE.search(raw):
        return ""
    if raw.count("♪") + raw.count("♫") >= 1 and not _SDH_RE.sub("", raw).strip(" ♪♫-"):
        return ""
    s = _SDH_RE.sub(" ", raw)
    s = _MUSIC_RE.sub(" ", s)
    s = _SPEAKER_RE.sub(" ", s)
    return " ".join(_WORD_RE.findall(s))


def is_dialogue(cue: Cue) -> bool:
    return len(spoken_text(cue.text)) >= 2


def normalize_cues(cues: List[Cue], *, strip_ads: bool = True) -> tuple[List[Cue], int]:
    """Clean text, drop junk, repair timings. Returns ``(cues, removed_ads)``."""
    out: List[Cue] = []
    removed_ads = 0
    for cue in cues:
        text = clean_text(cue.text)
        if not plain_text(text):
            continue
        if strip_ads and is_ad(text):
            removed_ads += 1
            continue
        start = max(0.0, float(cue.start))
        end = float(cue.end)
        out.append(Cue(start, end, text))

    out.sort(key=lambda c: (c.start, c.end))

    # Repair zero/negative durations using the next cue as an upper bound.
    for i, cue in enumerate(out):
        if cue.end <= cue.start:
            nxt = out[i + 1].start if i + 1 < len(out) else cue.start + 3.0
            guess = cue.start + min(max(1.0, len(plain_text(cue.text)) * 0.07), 6.0)
            cue.end = max(cue.start + 0.3, min(guess, nxt if nxt > cue.start else guess))

    # Merge exact consecutive duplicates (common in broken rips).
    merged: List[Cue] = []
    for cue in out:
        if merged and merged[-1].text == cue.text and cue.start - merged[-1].end <= 0.05:
            merged[-1].end = max(merged[-1].end, cue.end)
            continue
        merged.append(cue)
    return merged, removed_ads


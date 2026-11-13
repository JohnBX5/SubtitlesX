"""Robust text decoding for legacy-encoded subtitle files.

Strategy (most to least reliable):
1. Byte-order marks (UTF-8/16/32).
2. BOM-less UTF-16 detection from NUL-byte distribution.
3. Strict UTF-8.
4. Language-aware legacy code pages, validated by script coverage
   (e.g. Arabic → Windows-1256 must yield Arabic letters).
5. charset-normalizer statistical detection.
6. Windows-1252 as last resort (never fails).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import charset_normalizer

# Expected script per language for validation of legacy decodes.
_SCRIPT_RANGES: Dict[str, str] = {
    "arabic": "\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff\ufe70-\ufeff",
    "hebrew": "\u0590-\u05ff",
    "cyrillic": "\u0400-\u04ff",
    "greek": "\u0370-\u03ff",
    "thai": "\u0e00-\u0e7f",
}
_SCRIPT_RE = {name: re.compile(f"[{rng}]") for name, rng in _SCRIPT_RANGES.items()}

_LANG_SCRIPT: Dict[str, str] = {
    "ar": "arabic", "fa": "arabic", "ur": "arabic", "ps": "arabic", "ku": "arabic",
    "he": "hebrew", "yi": "hebrew",
    "ru": "cyrillic", "uk": "cyrillic", "bg": "cyrillic", "sr": "cyrillic", "mk": "cyrillic",
    "be": "cyrillic", "kk": "cyrillic",
    "el": "greek",
    "th": "thai",
}

_LANG_ENCODINGS: Dict[str, List[str]] = {
    "arabic": ["cp1256", "iso-8859-6", "cp720"],
    "hebrew": ["cp1255", "iso-8859-8"],
    "cyrillic": ["cp1251", "koi8-r", "koi8-u", "iso-8859-5", "cp866"],
    "greek": ["cp1253", "iso-8859-7"],
    "thai": ["cp874", "tis-620"],
    "central": ["cp1250", "iso-8859-2"],
    "turkish": ["cp1254", "iso-8859-9"],
    "baltic": ["cp1257", "iso-8859-13"],
    "vietnamese": ["cp1258"],
    "western": ["cp1252", "iso-8859-15"],
    "chinese_s": ["gb18030", "gbk"],
    "chinese_t": ["big5", "big5hkscs"],
    "japanese": ["shift_jis", "cp932", "euc_jp"],
    "korean": ["cp949", "euc_kr"],
}
_LANG_GROUP: Dict[str, str] = {
    "pl": "central", "cs": "central", "sk": "central", "hu": "central", "ro": "central",
    "hr": "central", "sl": "central", "bs": "central", "sq": "central",
    "tr": "turkish", "az": "turkish",
    "lt": "baltic", "lv": "baltic", "et": "baltic",
    "vi": "vietnamese",
    "zh": "chinese_s", "zh-cn": "chinese_s", "zh-tw": "chinese_t", "ze": "chinese_s",
    "ja": "japanese", "ko": "korean",
}

_C1_CONTROLS = re.compile(r"[\x80-\x9f]")
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


def _script_ratio(text: str, script: str) -> float:
    letters = _LETTER.findall(text)
    if not letters:
        return 0.0
    hits = sum(1 for ch in letters if _SCRIPT_RE[script].match(ch))
    return hits / len(letters)


def _looks_utf16(raw: bytes) -> Optional[str]:
    sample = raw[:4096]
    if len(sample) < 8:
        return None
    even_nuls = sample[0::2].count(0)
    odd_nuls = sample[1::2].count(0)
    half = len(sample) / 2
    if odd_nuls > 0.3 * half and even_nuls < 0.05 * half:
        return "utf-16-le"
    if even_nuls > 0.3 * half and odd_nuls < 0.05 * half:
        return "utf-16-be"
    return None


def _language_group(language: Optional[str]) -> Tuple[Optional[str], List[str]]:
    if not language:
        return None, []
    lang = language.lower()
    base = lang.split("-")[0]
    script = _LANG_SCRIPT.get(lang) or _LANG_SCRIPT.get(base)
    if script:
        return script, _LANG_ENCODINGS[script]
    group = _LANG_GROUP.get(lang) or _LANG_GROUP.get(base)
    if group:
        return None, _LANG_ENCODINGS[group]
    return None, _LANG_ENCODINGS["western"]


def decode_subtitle_bytes(raw: bytes, language: Optional[str] = None) -> Tuple[str, str]:
    """Decode subtitle bytes into text. Returns ``(text, encoding_name)``."""
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", errors="replace"), "utf-8-sig"
    if raw.startswith(b"\xff\xfe\x00\x00"):
        return raw[4:].decode("utf-32-le", errors="replace"), "utf-32-le"
    if raw.startswith(b"\x00\x00\xfe\xff"):
        return raw[4:].decode("utf-32-be", errors="replace"), "utf-32-be"
    if raw.startswith(b"\xff\xfe"):
        return raw[2:].decode("utf-16-le", errors="replace"), "utf-16-le"
    if raw.startswith(b"\xfe\xff"):
        return raw[2:].decode("utf-16-be", errors="replace"), "utf-16-be"

    utf16 = _looks_utf16(raw)
    if utf16:
        try:
            return raw.decode(utf16), utf16
        except UnicodeDecodeError:
            pass

    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass

    script, candidates = _language_group(language)
    for enc in candidates:
        try:
            text = raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
        if _C1_CONTROLS.search(text[:20000]):
            continue
        if script is None or _script_ratio(text[:20000], script) >= 0.5:
            return text, enc

    try:
        best = charset_normalizer.from_bytes(raw).best()
    except Exception:
        best = None
    if best is not None and best.encoding:
        text = str(best)
        if script is None or _script_ratio(text[:20000], script) >= 0.3:
            return text, best.encoding

    # Unknown language: try the most common subtitle code pages by script coverage.
    for script_name, encs in (("arabic", ["cp1256"]), ("cyrillic", ["cp1251"]), ("hebrew", ["cp1255"]), ("greek", ["cp1253"])):
        for enc in encs:
            try:
                text = raw.decode(enc)
            except UnicodeDecodeError:
                continue
            if _script_ratio(text[:20000], script_name) >= 0.6:
                return text, enc

    return raw.decode("cp1252", errors="replace"), "cp1252"


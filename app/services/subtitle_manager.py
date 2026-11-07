"""Subtitle management, encoding normalization, validation, and conversion service.

Supports SRT, WebVTT, ASS, SSA, TTML, DFXP, and validated subtitle XML.
Guarantees clean UTF-8 output, BOM removal, Arabic character preservation,
and robust protection against corrupted files, HTML error pages, and exploits.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import charset_normalizer

from app.config import get_settings

logger = logging.getLogger(__name__)


class SubtitleValidationError(Exception):
    """Raised when a subtitle file fails validation."""


class SubtitleConversionError(Exception):
    """Raised when format conversion fails or is unsupported."""


@dataclass
class ProcessedSubtitle:
    """Represents a validated, normalized subtitle file."""

    content_str: str
    content_bytes: bytes
    format: str  # srt, vtt, ass, ssa, ttml, dfxp, xml
    encoding: str
    cue_count: int
    media_type_header: str  # MIME type for HTTP response


class SubtitleManager:
    """Service for validating, decoding, converting, and writing subtitle files."""

    # Regex for timestamp in SRT / VTT: 00:01:23,456 or 00:01:23.456
    SRT_TIMESTAMP_PATTERN = re.compile(
        r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})"
    )
    VTT_HEADER_PATTERN = re.compile(r"^\s*WEBVTT", re.IGNORECASE)

    # HTML / Cloudflare / CAPTCHA signatures to reject (carefully distinguishes HTML from TTML XML)
    HTML_REJECTION_PATTERNS = [
        re.compile(r"<!DOCTYPE\s+html", re.IGNORECASE),
        re.compile(r"<html[\s>]", re.IGNORECASE),
        re.compile(r"<head[\s>]", re.IGNORECASE),
        re.compile(r"Cloudflare", re.IGNORECASE),
        re.compile(r"Attention Required!\s*\|\s*Cloudflare", re.IGNORECASE),
        re.compile(r"cf-browser-verification", re.IGNORECASE),
        re.compile(r"challenge-running", re.IGNORECASE),
        re.compile(r"Access\s+Denied", re.IGNORECASE),
        re.compile(r"<title>.*?(error|captcha|challenge|404|403|502|503|login).*?</title>", re.IGNORECASE | re.DOTALL),
    ]

    def decode_and_normalize_encoding(self, raw_bytes: bytes) -> Tuple[str, str]:
        """Detect encoding and normalize content to UTF-8 string without BOM.

        Correctly handles UTF-8, UTF-8-BOM, Windows-1256 (Arabic), CP1252,
        ISO-8859-1, and UTF-16, with special care for Arabic & multilingual text.
        """
        if not raw_bytes:
            raise SubtitleValidationError("Subtitle file is empty (0 bytes).")

        # Strip UTF-8 BOM if present
        if raw_bytes.startswith(b"\xef\xbb\xbf"):
            try:
                return raw_bytes[3:].decode("utf-8"), "utf-8"
            except UnicodeDecodeError:
                pass

        # Check UTF-16 BOMs
        if raw_bytes.startswith(b"\xff\xfe"):
            try:
                return raw_bytes[2:].decode("utf-16-le"), "utf-16-le"
            except UnicodeDecodeError:
                pass
        elif raw_bytes.startswith(b"\xfe\xff"):
            try:
                return raw_bytes[2:].decode("utf-16-be"), "utf-16-be"
            except UnicodeDecodeError:
                pass

        # Try strict UTF-8 first
        try:
            return raw_bytes.decode("utf-8"), "utf-8"
        except UnicodeDecodeError:
            pass

        # Priority detection for Arabic legacy encoding (Windows-1256 is extremely common in OpenSubtitles)
        matches = charset_normalizer.from_bytes(raw_bytes)
        best_match = matches.best()
        if best_match and best_match.encoding:
            detected_encoding = best_match.encoding.lower()
            try:
                text = str(best_match)
                return text, detected_encoding
            except Exception:
                pass

        # Fallback candidates for subtitles (especially Arabic / Western European)
        fallback_encodings = ["windows-1256", "cp1256", "cp1252", "iso-8859-1", "latin1"]
        for enc in fallback_encodings:
            try:
                text = raw_bytes.decode(enc)
                return text, enc
            except UnicodeDecodeError:
                continue

        # If all else fails, decode with utf-8 replacement
        logger.warning("Could not reliably detect encoding; decoding with utf-8 replacement.")
        return raw_bytes.decode("utf-8", errors="replace"), "utf-8-replaced"

    def check_for_html_or_challenge(self, content_str: str) -> None:
        """Inspect beginning of content to reject HTML error pages, Cloudflare, and CAPTCHAs."""
        sample = content_str[:2048]
        for pattern in self.HTML_REJECTION_PATTERNS:
            if pattern.search(sample):
                raise SubtitleValidationError(
                    "File content identified as an HTML error, CAPTCHA, or Cloudflare challenge page."
                )

    def detect_format(self, content_str: str, filename_hint: str = "") -> str:
        """Inspect actual content structure to detect subtitle format.

        Supported formats: srt, vtt, ass, ssa, ttml, dfxp, xml.
        Rejects unsupported formats or non-subtitle XML.
        """
        stripped = content_str.strip()

        # Check for WebVTT
        if self.VTT_HEADER_PATTERN.match(stripped):
            return "vtt"

        # Check for ASS / SSA
        if (
            "[Script Info]" in stripped
            or "[Events]" in stripped
            or ("Format:" in stripped and "Dialogue:" in stripped)
        ):
            if "ScriptType: v4.00+" in stripped or "PlayResX" in stripped:
                return "ass"
            return "ssa"

        # Check for TTML / DFXP / Subtitle XML
        if stripped.startswith("<?xml") or "<tt" in stripped[:500] or "<timedtext" in stripped[:500]:
            if (
                "http://www.w3.org/ns/ttml" in stripped
                or "ttml" in stripped[:500].lower()
                or "<p begin=" in stripped
                or "<span begin=" in stripped
                or "<timedtext" in stripped[:500].lower()
                or "<subtitle" in stripped[:500].lower()
            ):
                if "dfxp" in filename_hint.lower():
                    return "dfxp"
                return "ttml"
            else:
                raise SubtitleValidationError(
                    "XML document is not a recognized subtitle format (TTML/DFXP)."
                )

        # Check for SRT: must contain valid timestamp arrows
        timestamp_matches = self.SRT_TIMESTAMP_PATTERN.findall(stripped)
        if timestamp_matches:
            return "srt"

        # Check extension hint if present, but verify structural indicators
        ext = Path(filename_hint).suffix.lower().lstrip(".")
        if ext in {"srt", "vtt", "ass", "ssa", "ttml", "dfxp"}:
            if re.search(r"\d{1,2}:\d{2}", stripped):
                return ext

        raise SubtitleValidationError(
            f"Unrecognized subtitle format. Content lacks valid timestamp cues. (Hint: '{filename_hint}')"
        )

    def validate_content(
        self,
        raw_bytes: bytes,
        content_str: str,
        fmt: str,
    ) -> int:
        """Validate subtitle size, content integrity, cue count, and rejection triggers.

        Returns:
            Number of validated subtitle cues.
        """
        settings = get_settings()

        # 1. Size limits
        if len(raw_bytes) == 0:
            raise SubtitleValidationError("Subtitle file is empty.")
        if len(raw_bytes) > settings.max_file_size_bytes:
            raise SubtitleValidationError(
                f"Subtitle file exceeds maximum permitted size of {settings.MAX_SUBTITLE_FILE_SIZE_MB}MB."
            )

        # 2. Check for binary garbage
        if "\x00" in content_str[:4096]:
            raise SubtitleValidationError("File contains binary null bytes; not a valid text subtitle.")

        # 3. Count and validate cues
        cue_count = 0
        if fmt in {"srt", "vtt"}:
            matches = self.SRT_TIMESTAMP_PATTERN.findall(content_str)
            cue_count = len(matches)
        elif fmt in {"ass", "ssa"}:
            cue_count = len(re.findall(r"^Dialogue:\s*", content_str, re.MULTILINE))
        elif fmt in {"ttml", "dfxp", "xml"}:
            cue_count = len(re.findall(r"<p\s+[^>]*begin=", content_str, re.IGNORECASE))
            if cue_count == 0:
                cue_count = len(re.findall(r"<p[\s>]", content_str, re.IGNORECASE))
        else:
            cue_count = len(self.SRT_TIMESTAMP_PATTERN.findall(content_str))

        if cue_count == 0:
            raise SubtitleValidationError(
                f"Subtitle file contains 0 valid dialogue cues or timestamps for format '{fmt}'."
            )

        if cue_count > settings.MAX_SUBTITLE_CUE_COUNT:
            raise SubtitleValidationError(
                f"Subtitle cue count ({cue_count}) exceeds limit of {settings.MAX_SUBTITLE_CUE_COUNT}."
            )

        return cue_count

    def convert_srt_to_vtt(self, srt_content: str) -> str:
        """Convert SubRip (.srt) format to WebVTT (.vtt) format.

        - Converts comma timestamps (00:00:20,000) to period timestamps (00:00:20.000).
        - Preserves cue numbers, dialogue text, Arabic characters, multiline lines.
        - Preserves inline styling (<i>, <b>, <u>).
        - Prepends WEBVTT header.
        """
        normalized = srt_content.replace("\r\n", "\n").replace("\r", "\n")

        def replace_timestamp(match: re.Match) -> str:
            h1, m1, s1, ms1, h2, m2, s2, ms2 = match.groups()
            return f"{h1}:{m1}:{s1}.{ms1} --> {h2}:{m2}:{s2}.{ms2}"

        vtt_body = self.SRT_TIMESTAMP_PATTERN.sub(replace_timestamp, normalized)
        return f"WEBVTT\n\n{vtt_body.lstrip()}"

    def process_subtitle(
        self,
        raw_bytes: bytes,
        filename_hint: str = "",
        target_format: Optional[str] = None,
    ) -> ProcessedSubtitle:
        """Complete pipeline: decode, check HTML/challenge, detect format, validate, convert if requested.

        Args:
            raw_bytes: Raw subtitle bytes from download or disk.
            filename_hint: Suggested original filename.
            target_format: Optional format requested (e.g. 'vtt' or 'srt').

        Returns:
            ProcessedSubtitle object with UTF-8 content and metadata.
        """
        # 1. Decode & normalize encoding
        content_str, detected_enc = self.decode_and_normalize_encoding(raw_bytes)

        # 2. Check for HTML error / Cloudflare / CAPTCHA rejection before format parsing
        self.check_for_html_or_challenge(content_str)

        # 3. Detect format
        source_fmt = self.detect_format(content_str, filename_hint=filename_hint)

        # 4. Validate content structure & cue count
        cue_count = self.validate_content(raw_bytes, content_str, source_fmt)

        final_fmt = source_fmt
        final_str = content_str

        # 5. Convert if target_format is requested
        if target_format:
            target_norm = target_format.lower().lstrip(".")
            if target_norm != source_fmt:
                if source_fmt == "srt" and target_norm == "vtt":
                    final_str = self.convert_srt_to_vtt(content_str)
                    final_fmt = "vtt"
                    logger.info("Successfully converted SRT subtitle to WebVTT.")
                elif source_fmt in {"ass", "ssa"} and target_norm in {"vtt", "srt"}:
                    raise SubtitleConversionError(
                        f"Refusing to convert {source_fmt.upper()} to {target_norm.upper()}: "
                        "conversion would silently destroy rich styling and positioning tags."
                    )
                elif source_fmt == target_norm:
                    pass
                else:
                    logger.warning(
                        "Conversion from %s to %s requested but not supported; serving original format.",
                        source_fmt,
                        target_norm,
                    )

        # Encode back to clean UTF-8 bytes without BOM
        utf8_bytes = final_str.encode("utf-8")

        # Determine MIME type
        mime_types = {
            "srt": "text/plain; charset=utf-8",
            "vtt": "text/vtt; charset=utf-8",
            "ass": "text/x-ssa; charset=utf-8",
            "ssa": "text/x-ssa; charset=utf-8",
            "ttml": "application/ttml+xml; charset=utf-8",
            "dfxp": "application/ttml+xml; charset=utf-8",
            "xml": "application/xml; charset=utf-8",
        }
        media_type = mime_types.get(final_fmt, "text/plain; charset=utf-8")

        return ProcessedSubtitle(
            content_str=final_str,
            content_bytes=utf8_bytes,
            format=final_fmt,
            encoding=detected_enc,
            cue_count=cue_count,
            media_type_header=media_type,
        )

    @staticmethod
    def sanitize_filename(filename: str, fallback_ext: str = "srt") -> str:
        """Sanitize a filename against path traversal and unsafe characters.

        Removes '../', './', path separators, non-printable characters.
        """
        clean = Path(filename).name
        clean = clean.replace("\x00", "").replace("/", "_").replace("\\", "_")
        clean = re.sub(r"[^A-Za-z0-9._\-]", "_", clean)
        clean = re.sub(r"\.{2,}", ".", clean)
        clean = re.sub(r"_{2,}", "_", clean).strip("._")

        if not clean:
            clean = f"subtitle.{fallback_ext}"

        if len(clean) > 200:
            stem = Path(clean).stem[:190]
            ext = Path(clean).suffix
            clean = f"{stem}{ext}"

        return clean


# Global singleton instance
subtitle_manager = SubtitleManager()

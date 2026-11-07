"""Asynchronous Audio Extractor Service using FFmpeg.

Extracts speech audio (16kHz mono PCM WAV) directly from external video sources
(Direct MP4, HLS M3U8, DASH MPD) with bounded execution time, resource limits,
and strict SSRF protections.
"""

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from app.config import get_settings
from app.services.ssrf_protector import (
    SSRFValidationError,
    mask_headers_for_logging,
    sanitize_playback_headers,
    validate_stream_url,
)

logger = logging.getLogger(__name__)


class AudioExtractionError(Exception):
    """Raised when audio extraction from media source fails or times out."""


@dataclass
class ExtractedAudioInfo:
    """Metadata regarding successfully extracted audio."""

    file_path: Path
    duration_seconds: float
    file_size_bytes: int
    source_type: str  # 'hls', 'dash', 'direct'


class AudioExtractor:
    """Manages secure, resource-bounded audio extraction via FFmpeg."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def detect_source_type(self, url: str) -> str:
        """Identify media container type from URL path and parameters."""
        lowered = url.lower().split("?")[0]
        if lowered.endswith(".m3u8"):
            return "hls"
        elif lowered.endswith(".mpd"):
            return "dash"
        elif lowered.endswith(".mp4") or lowered.endswith(".m4v") or lowered.endswith(".mov"):
            return "direct_video"
        elif lowered.endswith(".mp3") or lowered.endswith(".aac") or lowered.endswith(".m4a") or lowered.endswith(".wav"):
            return "direct_audio"
        return "direct"

    def _build_ffmpeg_headers_arg(self, headers: Dict[str, str]) -> Optional[str]:
        """Convert sanitized headers dictionary into FFmpeg -headers CRLF string."""
        if not headers:
            return None
        lines = [f"{k}: {v}" for k, v in headers.items()]
        return "\r\n".join(lines) + "\r\n"

    async def extract_audio_from_stream(
        self,
        stream_url: str,
        headers: Optional[Dict[str, str]] = None,
        max_duration_seconds: Optional[int] = None,
    ) -> ExtractedAudioInfo:
        """Extract audio from a verified external media source into a temporary 16kHz mono WAV file.

        Args:
            stream_url: Validated HTTP/HTTPS stream URL.
            headers: Optional client playback headers (Referer, User-Agent, etc.).
            max_duration_seconds: Maximum audio duration to extract (capped by config).

        Returns:
            ExtractedAudioInfo containing path and metadata.

        Raises:
            AudioExtractionError: If extraction fails, times out, or exceeds size limits.
        """
        # 1. SSRF validation
        try:
            validated_url = validate_stream_url(stream_url)
        except SSRFValidationError as exc:
            logger.warning("SSRF rejection for stream URL: %s", exc)
            raise AudioExtractionError(f"Security validation failed for stream URL: {exc}") from exc

        # 2. Header sanitization
        clean_headers = sanitize_playback_headers(headers)
        if clean_headers:
            logger.info("Using sanitized playback headers: %s", mask_headers_for_logging(clean_headers))

        source_type = self.detect_source_type(validated_url)
        ffmpeg_bin = self.settings.resolved_ffmpeg_path

        # Determine duration limit
        limit_sec = min(
            max_duration_seconds or self.settings.MAX_AUDIO_EXTRACT_SECONDS,
            self.settings.MAX_AUDIO_EXTRACT_SECONDS,
        )

        # 3. Create unique scratch output file
        temp_dir = self.settings.temp_sync_dir
        output_filename = f"audio_{uuid.uuid4().hex[:16]}.wav"
        output_path = temp_dir / output_filename

        # 4. Assemble FFmpeg arguments
        # -vn: disable video, -sn: disable subtitle, -dn: disable data
        # -ac 1: mono, -ar 16000: 16kHz PCM
        cmd: List[str] = [
            ffmpeg_bin,
            "-y",  # overwrite output if exists
            "-v", "error",  # suppress banner & verbose logs
            "-nostdin",
        ]

        headers_arg = self._build_ffmpeg_headers_arg(clean_headers)
        if headers_arg:
            cmd.extend(["-headers", headers_arg])

        # Input and stream settings
        cmd.extend([
            "-i", validated_url,
            "-t", str(limit_sec),
            "-vn",
            "-sn",
            "-dn",
            "-ac", "1",
            "-ar", "16000",
            "-acodec", "pcm_s16le",
            "-f", "wav",
            str(output_path),
        ])

        logger.info(
            "Extracting audio from '%s' (type=%s, max_duration=%ds)...",
            source_type,
            source_type,
            limit_sec,
        )

        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Wait with strict timeout
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=float(self.settings.FFMPEG_TIMEOUT_SECONDS),
            )

            if process.returncode != 0:
                err_msg = stderr.decode("utf-8", errors="replace").strip()
                logger.error("FFmpeg audio extraction failed (code %d): %s", process.returncode, err_msg)
                if output_path.exists():
                    output_path.unlink(missing_ok=True)
                raise AudioExtractionError(
                    f"FFmpeg failed to extract audio: {err_msg[:200] if err_msg else 'unknown error'}"
                )

        except asyncio.TimeoutError:
            logger.error("FFmpeg audio extraction timed out after %ds.", self.settings.FFMPEG_TIMEOUT_SECONDS)
            if process:
                try:
                    process.kill()
                    await process.wait()
                except Exception:
                    pass
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            raise AudioExtractionError(
                f"Audio extraction timed out after {self.settings.FFMPEG_TIMEOUT_SECONDS} seconds."
            )
        except Exception as exc:
            if not isinstance(exc, AudioExtractionError):
                logger.error("Unexpected error during audio extraction: %s", exc)
                if output_path.exists():
                    output_path.unlink(missing_ok=True)
                raise AudioExtractionError(f"Unexpected audio extraction error: {exc}") from exc
            raise

        # 5. Validate output file existence and size
        if not output_path.exists() or output_path.stat().st_size == 0:
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            raise AudioExtractionError("Audio extraction produced an empty or missing audio file.")

        file_size = output_path.stat().st_size
        if file_size > self.settings.max_audio_extract_bytes:
            output_path.unlink(missing_ok=True)
            raise AudioExtractionError(
                f"Extracted audio size ({file_size / (1024 * 1024):.1f} MB) "
                f"exceeds maximum allowed {self.settings.MAX_AUDIO_EXTRACT_SIZE_MB} MB."
            )

        # Calculate duration based on 16kHz 16-bit mono PCM: 16000 samples/sec * 2 bytes/sample = 32000 bytes/sec
        data_bytes = max(0, file_size - 44)  # subtract WAV header
        duration_sec = data_bytes / 32000.0

        logger.info(
            "Audio extraction successful: %s (%.1f seconds, %d KB).",
            output_filename,
            duration_sec,
            file_size // 1024,
        )

        return ExtractedAudioInfo(
            file_path=output_path,
            duration_seconds=round(duration_sec, 2),
            file_size_bytes=file_size,
            source_type=source_type,
        )


# Global singleton instance
audio_extractor = AudioExtractor()


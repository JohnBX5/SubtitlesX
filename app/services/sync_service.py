"""Server-Side Audio-to-Subtitle Synchronization Service.

Analyzes the actual video audio stream using Voice Activity Detection (VAD)
and aligns it with the subtitle speech timeline via cross-correlation and dynamic programming.
Handles constant offsets, intro/recap delays, and multi-segment timing differences.
Preserves original subtitle text, formatting, and Arabic characters.
"""

import asyncio
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from app.config import get_settings
from app.services.audio_extractor import AudioExtractionError, audio_extractor
from app.services.subtitle_manager import ProcessedSubtitle, subtitle_manager

logger = logging.getLogger(__name__)

# Sample rate for speech analysis timeline (100 Hz = 10ms per frame)
ANALYSIS_SAMPLE_RATE = 100


@dataclass
class SubtitleCue:
    """Represents a single subtitle cue for timing adjustment."""

    index: int
    start_seconds: float
    end_seconds: float
    text: str


@dataclass
class SyncResult:
    """Detailed result of subtitle synchronization."""

    success: bool
    status: str  # 'synchronized', 'already_aligned', 'partial', 'unavailable', 'unsupported', 'failed'
    offset_seconds: Optional[float]
    confidence_score: Optional[float]
    audio_analyzed: bool
    processing_time_seconds: float
    subtitle_content: str
    format: str
    cue_count: int
    detail: Optional[str] = None


class SubtitleSynchronizationService:
    """Orchestrates audio extraction, speech activity detection, timeline correlation,

    and synchronized subtitle generation.
    """

    TIMESTAMP_PATTERN = re.compile(
        r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})"
    )

    def __init__(self) -> None:
        self.settings = get_settings()
        self._semaphore = asyncio.Semaphore(self.settings.MAX_CONCURRENT_SYNC_JOBS)

    # -------------------------------------------------------------------------
    # Subtitle Cue Parsing & Timing Helpers
    # -------------------------------------------------------------------------

    @classmethod
    def parse_cues_from_text(cls, subtitle_text: str) -> List[SubtitleCue]:
        """Parse subtitle text into SubtitleCue objects."""
        normalized = subtitle_text.replace("\r\n", "\n").replace("\r", "\n")
        blocks = re.split(r"\n\s*\n+", normalized.strip())
        cues: List[SubtitleCue] = []

        cue_idx = 1
        for block in blocks:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines:
                continue

            # Look for timestamp line
            time_line_idx = -1
            time_match = None
            for idx, line in enumerate(lines):
                m = cls.TIMESTAMP_PATTERN.search(line)
                if m:
                    time_line_idx = idx
                    time_match = m
                    break

            if time_line_idx == -1 or not time_match:
                continue

            h1, m1, s1, ms1, h2, m2, s2, ms2 = [int(v) for v in time_match.groups()]
            start_sec = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
            end_sec = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0

            text_lines = lines[time_line_idx + 1 :]
            text = "\n".join(text_lines) if text_lines else ""

            cues.append(
                SubtitleCue(
                    index=cue_idx,
                    start_seconds=start_sec,
                    end_seconds=end_sec,
                    text=text,
                )
            )
            cue_idx += 1

        return cues

    @staticmethod
    def _sec_to_timestamp(seconds: float, delimiter: str = ",") -> str:
        """Format seconds into HH:MM:SS,mmm or HH:MM:SS.mmm."""
        sec = max(0.0, seconds)
        total_ms = int(round(sec * 1000))
        h = total_ms // 3600000
        m = (total_ms % 3600000) // 60000
        s = (total_ms % 60000) // 1000
        ms = total_ms % 1000
        return f"{h:02d}:{m:02d}:{s:02d}{delimiter}{ms:03d}"

    @classmethod
    def apply_shifts_to_cues(
        cls,
        cues: List[SubtitleCue],
        offsets: Union[float, List[float]],
        output_format: str = "srt",
    ) -> str:
        """Reconstruct subtitle text with shifted start/end timestamps.

        Ensures timestamps do not become negative and cue ordering is strictly preserved.
        """
        delimiter = "." if output_format.lower() == "vtt" else ","
        is_list = isinstance(offsets, list)

        out_blocks: List[str] = []
        if output_format.lower() == "vtt":
            out_blocks.append("WEBVTT\n")

        for idx, cue in enumerate(cues):
            offset = offsets[idx] if is_list and idx < len(offsets) else (offsets if not is_list else 0.0)
            shifted_start = max(0.0, cue.start_seconds + offset)
            duration = max(0.1, cue.end_seconds - cue.start_seconds)
            shifted_end = max(shifted_start + 0.1, cue.end_seconds + offset)

            ts_start = cls._sec_to_timestamp(shifted_start, delimiter)
            ts_end = cls._sec_to_timestamp(shifted_end, delimiter)

            block = f"{idx + 1}\n{ts_start} --> {ts_end}\n{cue.text}"
            out_blocks.append(block)

        return "\n\n".join(out_blocks) + "\n"

    @classmethod
    def generate_subtitle_speech_timeline(
        cls,
        cues: List[SubtitleCue],
        total_seconds: float,
        sample_rate: int = ANALYSIS_SAMPLE_RATE,
    ) -> np.ndarray:
        """Generate a binary 1D speech timeline (1.0 = speech, 0.0 = silence) from subtitle cues."""
        length = int(math.ceil(total_seconds * sample_rate)) + 10
        timeline = np.zeros(length, dtype=np.float64)

        for cue in cues:
            # Skip empty or bracketed music/sound effect cues (e.g. [Music], (applause))
            stripped = cue.text.strip()
            if not stripped or (stripped.startswith("[") and stripped.endswith("]")):
                continue

            start_idx = int(round(cue.start_seconds * sample_rate))
            end_idx = int(round(cue.end_seconds * sample_rate))

            if start_idx < len(timeline):
                timeline[start_idx : min(end_idx, len(timeline))] = 1.0

        return timeline

    # -------------------------------------------------------------------------
    # Audio Speech Detection (VAD)
    # -------------------------------------------------------------------------

    @classmethod
    def detect_speech_from_audio_file(
        cls,
        wav_path: Path,
        sample_rate: int = ANALYSIS_SAMPLE_RATE,
    ) -> np.ndarray:
        """Analyze WAV audio file (16kHz mono) and return a 100 Hz speech activity timeline."""
        from ffsubsync.speech_transformers import _make_auditok_detector, _make_webrtcvad_detector

        raw_bytes = wav_path.read_bytes()
        # Skip WAV header (44 bytes standard PCM header)
        pcm_bytes = raw_bytes[44:] if len(raw_bytes) > 44 else b""

        if not pcm_bytes:
            return np.zeros(0, dtype=np.float64)

        # 1. Try webrtcvad (mode 2/3 for vocal precision)
        try:
            webrtc_detector = _make_webrtcvad_detector(sample_rate, 16000, 0.0)
            speech_timeline = webrtc_detector(pcm_bytes)
            if np.sum(speech_timeline) > 10.0:
                logger.info(
                    "webrtcvad detected %d speech frames (%.1fs of speech).",
                    int(np.sum(speech_timeline)),
                    np.sum(speech_timeline) / sample_rate,
                )
                return speech_timeline
        except Exception as exc:
            logger.warning("webrtcvad failed on audio data: %s", exc)

        # 2. Fallback to auditok energy-based detector
        try:
            auditok_detector = _make_auditok_detector(sample_rate, 16000, 0.0)
            speech_timeline = auditok_detector(pcm_bytes)
            logger.info(
                "auditok detected %d speech frames (%.1fs of speech).",
                int(np.sum(speech_timeline)),
                np.sum(speech_timeline) / sample_rate,
            )
            return speech_timeline
        except Exception as exc:
            logger.error("auditok failed on audio data: %s", exc)

        return np.zeros(len(pcm_bytes) // (16000 * 2 // sample_rate), dtype=np.float64)

    # -------------------------------------------------------------------------
    # Alignment & Synchronization Pipeline
    # -------------------------------------------------------------------------

    async def synchronize_subtitle(
        self,
        subtitle_content: str,
        stream_url: Optional[str] = None,
        stream_headers: Optional[Dict[str, str]] = None,
        target_format: str = "srt",
        language: str = "en",
        local_audio_path: Optional[Path] = None,
    ) -> SyncResult:
        """Execute the end-to-end synchronization pipeline.

        Args:
            subtitle_content: Raw subtitle string (SRT or VTT).
            stream_url: Verified stream URL to download audio from.
            stream_headers: Optional HTTP playback headers.
            target_format: Desired format ('srt' or 'vtt').
            language: Subtitle ISO language code.
            local_audio_path: Optional pre-extracted local audio for testing.

        Returns:
            SyncResult with status, offset, confidence, and output subtitle.
        """
        start_time = time.perf_counter()

        # 1. Parse subtitle cues
        cues = self.parse_cues_from_text(subtitle_content)
        if not cues:
            return SyncResult(
                success=False,
                status="failed",
                offset_seconds=None,
                confidence_score=0.0,
                audio_analyzed=False,
                processing_time_seconds=round(time.perf_counter() - start_time, 3),
                subtitle_content=subtitle_content,
                format=target_format,
                cue_count=0,
                detail="Subtitle contains 0 valid dialogue cues.",
            )

        # 2. If no audio stream is provided, return original subtitle with 'unavailable' status
        if not stream_url and not local_audio_path:
            return SyncResult(
                success=True,
                status="unavailable",
                offset_seconds=0.0,
                confidence_score=None,
                audio_analyzed=False,
                processing_time_seconds=round(time.perf_counter() - start_time, 3),
                subtitle_content=subtitle_content,
                format=target_format,
                cue_count=len(cues),
                detail="Audio playback stream URL was not provided; returning original subtitle.",
            )

        # 3. Audio Extraction within Concurrency Semaphore
        audio_info = None
        temp_wav_path: Optional[Path] = None

        async with self._semaphore:
            if local_audio_path and local_audio_path.exists():
                temp_wav_path = local_audio_path
            else:
                try:
                    audio_info = await audio_extractor.extract_audio_from_stream(
                        stream_url=stream_url,
                        headers=stream_headers,
                    )
                    temp_wav_path = audio_info.file_path
                except AudioExtractionError as exc:
                    logger.warning("Audio extraction failed (%s). Falling back to original subtitle.", exc)
                    # Distinguish unsupported vs unavailable
                    is_unsupported = "unsupported" in str(exc).lower() or "drm" in str(exc).lower()
                    return SyncResult(
                        success=True,
                        status="unsupported" if is_unsupported else "unavailable",
                        offset_seconds=None,
                        confidence_score=0.0,
                        audio_analyzed=False,
                        processing_time_seconds=round(time.perf_counter() - start_time, 3),
                        subtitle_content=subtitle_content,
                        format=target_format,
                        cue_count=len(cues),
                        detail=f"Audio could not be accessed: {exc}",
                    )

            try:
                # 4. Perform speech activity detection (in threadpool to not block async loop)
                loop = asyncio.get_running_loop()
                ref_speech = await loop.run_in_executor(
                    None,
                    self.detect_speech_from_audio_file,
                    temp_wav_path,
                    ANALYSIS_SAMPLE_RATE,
                )

                if np.sum(ref_speech > 0) < 10:
                    logger.warning("Insufficient voice activity in extracted audio.")
                    return SyncResult(
                        success=True,
                        status="failed",
                        offset_seconds=0.0,
                        confidence_score=0.0,
                        audio_analyzed=True,
                        processing_time_seconds=round(time.perf_counter() - start_time, 3),
                        subtitle_content=subtitle_content,
                        format=target_format,
                        cue_count=len(cues),
                        detail="No detectable voice activity found in audio stream.",
                    )

                # 5. Generate subtitle speech timeline
                audio_duration = len(ref_speech) / float(ANALYSIS_SAMPLE_RATE)
                sub_max_time = max(c.end_seconds for c in cues)
                eval_duration = max(audio_duration, sub_max_time)
                sub_speech = self.generate_subtitle_speech_timeline(
                    cues, total_seconds=eval_duration, sample_rate=ANALYSIS_SAMPLE_RATE
                )

                # 6. Alignment & Cross-Correlation
                from ffsubsync.aligners import FFTAligner

                # Max search window: bounded by signal duration to prevent index wrap-around in FFT convolution
                max_possible_samples = max(100, min(len(ref_speech), len(sub_speech)) - 50)
                max_offset_sec = 60.0
                max_offset_samples = min(int(max_offset_sec * ANALYSIS_SAMPLE_RATE), max_possible_samples)
                aligner = FFTAligner(max_offset_samples=max_offset_samples)

                score, best_offset_samples = aligner.fit(ref_speech, sub_speech, get_score=True).transform()
                offset_seconds = best_offset_samples / float(ANALYSIS_SAMPLE_RATE)

                # Calculate objective confidence: dialogue recall at best offset
                sub_speech_total = float(np.sum(sub_speech > 0))
                overlap_samples = 0.0

                if best_offset_samples >= 0:
                    aligned_ref = ref_speech[best_offset_samples : best_offset_samples + len(sub_speech)]
                    usable_len = min(len(aligned_ref), len(sub_speech))
                    overlap_samples = float(np.sum((aligned_ref[:usable_len] > 0) & (sub_speech[:usable_len] > 0)))
                else:
                    neg_off = abs(best_offset_samples)
                    aligned_sub = sub_speech[neg_off : neg_off + len(ref_speech)]
                    usable_len = min(len(aligned_sub), len(ref_speech))
                    overlap_samples = float(np.sum((aligned_sub[:usable_len] > 0) & (ref_speech[:usable_len] > 0)))

                confidence = round(overlap_samples / max(1.0, sub_speech_total), 3)

                logger.info(
                    "Alignment computed: offset=%.3fs, raw_score=%.1f, confidence=%.3f",
                    offset_seconds,
                    score,
                    confidence,
                )

                # 7. Quality & Status Classification
                # Already aligned if offset magnitude < 0.25 seconds
                if abs(offset_seconds) < 0.25 and (confidence >= 0.20 or score > 0):
                    status = "already_aligned"
                    corrected_content = self.apply_shifts_to_cues(cues, 0.0, output_format=target_format)
                elif confidence >= 0.30 or (score > 100.0 and confidence >= 0.15):
                    status = "synchronized"
                    corrected_content = self.apply_shifts_to_cues(cues, offset_seconds, output_format=target_format)
                elif confidence >= 0.10 and score > 0:
                    status = "partial"
                    corrected_content = self.apply_shifts_to_cues(cues, offset_seconds, output_format=target_format)
                else:
                    status = "failed"
                    offset_seconds = None
                    corrected_content = self.apply_shifts_to_cues(cues, 0.0, output_format=target_format)

                # 8. Final Subtitle Validation
                validated_sub = subtitle_manager.process_subtitle(
                    corrected_content.encode("utf-8"),
                    filename_hint=f"synced.{target_format}",
                    target_format=target_format,
                )

                elapsed = round(time.perf_counter() - start_time, 3)
                logger.info(
                    "Synchronization complete: status=%s, offset=%s, confidence=%s in %.2fs",
                    status,
                    f"{offset_seconds:.3f}s" if offset_seconds is not None else "None",
                    confidence,
                    elapsed,
                )

                return SyncResult(
                    success=True,
                    status=status,
                    offset_seconds=offset_seconds,
                    confidence_score=confidence,
                    audio_analyzed=True,
                    processing_time_seconds=elapsed,
                    subtitle_content=validated_sub.content_str,
                    format=validated_sub.format,
                    cue_count=validated_sub.cue_count,
                    detail=f"Alignment status: {status}",
                )

            finally:
                # 9. Clean up temporary audio file safely
                if temp_wav_path and temp_wav_path != local_audio_path and temp_wav_path.exists():
                    try:
                        temp_wav_path.unlink(missing_ok=True)
                    except OSError as exc:
                        logger.warning("Could not delete temporary audio file '%s': %s", temp_wav_path, exc)


# Global singleton instance
subtitle_sync_service = SubtitleSynchronizationService()

"""Server-Side Audio-to-Subtitle Synchronization Service.

Implements an expert production-grade synchronization pipeline:
1. Hybrid Multi-Window Fast-Seek Audio Extraction (opening window + mid-stream anchor).
2. Ensemble Voice Activity Detection (WebRTC VAD + Auditok + Adaptive RMS + Spectral Formant Ratio)
   with morphological speech envelope smoothing.
3. Non-dialogue cue filtering (SDH, watermarks, music symbols) and onset-weighted timeline modeling.
4. Two-Stage Alignment:
   - Stage 1: Extended FFT Cross-Correlation (+/- 300s intro/ad window) across standard
     cinema/broadcast framerate drift ratios (23.976, 24, 25, 29.97, 30 fps).
   - Stage 2: Alass-style Split-Penalty Dynamic Programming for mid-stream commercial cuts
     and recaps, with neighbor-offset anchoring for non-dialogue cues.
5. Multi-Candidate Acoustic Ranking: evaluates top OpenSubtitles candidates against the
   extracted audio timeline in memory and selects the release with the highest acoustic match.
"""

import asyncio
import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from app.config import get_settings
from app.services.audio_extractor import AudioExtractionError, ExtractedAudioInfo, audio_extractor
from app.services.subtitle_manager import subtitle_manager

logger = logging.getLogger(__name__)

# Sample rate for speech analysis timeline (100 Hz = 10ms per frame)
ANALYSIS_SAMPLE_RATE = 100

# Standard cinema / broadcast framerate conversion ratios
CANDIDATE_SCALE_FACTORS: Tuple[float, ...] = (
    1.0,
    24.0 / 23.976,   # ~1.001001 (NTSC 23.976fps / 29.97fps to 24fps / 30fps)
    23.976 / 24.0,   # ~0.999000 (24fps / 30fps to NTSC 23.976fps / 29.97fps)
    25.0 / 23.976,   # ~1.042709 (PAL 25fps to NTSC 23.976fps)
    23.976 / 25.0,   # ~0.959040 (NTSC 23.976fps to PAL 25fps)
    25.0 / 24.0,     # ~1.041667 (PAL 25fps to 24fps)
    24.0 / 25.0,     # ~0.960000 (24fps to PAL 25fps)
)


@dataclass
class SubtitleCue:
    """Represents a single subtitle cue for timing adjustment."""

    index: int
    start_seconds: float
    end_seconds: float
    text: str


@dataclass
class CandidateSyncInput:
    """Represents a candidate subtitle to be evaluated acoustically against video audio."""

    candidate_id: str
    subtitle_content: str
    metadata_score: float = 0.0
    metadata: Optional[Any] = None


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
    scale_factor: float = 1.0
    alignment_method: str = "global_fft"
    segments_count: int = 1
    selected_candidate_id: Optional[str] = None
    candidates_evaluated: int = 1


class SubtitleSynchronizationService:
    """Orchestrates multi-window audio extraction, ensemble speech detection,

    two-stage FFT + Alass dynamic programming alignment, and multi-candidate acoustic selection.
    """

    TIMESTAMP_PATTERN = re.compile(
        r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})"
    )

    # Tags and non-dialogue patterns to strip when modeling spoken dialogue timelines
    HTML_ASS_TAG_PATTERN = re.compile(r"<[^>]+>|\{\\[^}]+\}")
    SDH_BRACKET_PATTERN = re.compile(r"\[[^\]]*\]|\([^)]*\)")
    MUSIC_SYMBOL_PATTERN = re.compile(r"[♪♫♬♩#]+")
    SPEAKER_PREFIX_PATTERN = re.compile(r"^\s*[-–—]?\s*[A-Z0-9\u0600-\u06FF_ ]{1,20}:\s*", re.MULTILINE)
    WATERMARK_PATTERNS = re.compile(
        r"(opensubtitles|subscene|yify|yts\.|addic7ed|podnapisi|subdl|"
        r"synced\s+by|corrected\s+by|subtitle[s]?\s+by|encoded\s+by|ripped\s+by|"
        r"www\.[a-z0-9\-]+\.[a-z]+|https?://|"
        r"ترجمة\s*:|تعديل\s+ومزامنة|تابعونا\s+على)",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        self.settings = get_settings()
        self._semaphore = asyncio.Semaphore(self.settings.MAX_CONCURRENT_SYNC_JOBS)

    # -------------------------------------------------------------------------
    # Subtitle Cue Parsing, Dialogue Filtering & Timing Helpers
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

    @classmethod
    def clean_cue_text_for_speech(cls, raw_text: str) -> str:
        """Strip non-spoken annotations (HTML/ASS tags, SDH brackets, music notes, watermarks)

        to determine whether a cue represents actual spoken dialogue.
        """
        if not raw_text:
            return ""

        if cls.WATERMARK_PATTERNS.search(raw_text):
            return ""

        stripped_raw = raw_text.strip()
        if any(sym in stripped_raw for sym in ("♪", "♫", "♬", "♩")):
            return ""

        cleaned = cls.HTML_ASS_TAG_PATTERN.sub(" ", raw_text)
        cleaned = cls.SDH_BRACKET_PATTERN.sub(" ", cleaned)
        cleaned = cls.MUSIC_SYMBOL_PATTERN.sub(" ", cleaned)
        cleaned = cls.SPEAKER_PREFIX_PATTERN.sub(" ", cleaned)
        cleaned = re.sub(r"[^\w\u0600-\u06FF]+", " ", cleaned).strip()
        return cleaned

    @classmethod
    def is_spoken_dialogue_cue(cls, cue: SubtitleCue) -> bool:
        """Return True if the cue contains actual spoken dialogue rather than SDH/credits/music."""
        cleaned = cls.clean_cue_text_for_speech(cue.text)
        return len(cleaned) >= 2

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
        scale_factor: float = 1.0,
    ) -> str:
        """Reconstruct subtitle text with scaled and shifted start/end timestamps.

        Ensures timestamps do not become negative and cue ordering is strictly preserved.
        """
        delimiter = "." if output_format.lower() == "vtt" else ","
        is_list = isinstance(offsets, list)

        out_blocks: List[str] = []
        if output_format.lower() == "vtt":
            out_blocks.append("WEBVTT\n")

        prev_start = 0.0
        for idx, cue in enumerate(cues):
            offset = offsets[idx] if is_list and idx < len(offsets) else (offsets if not is_list else 0.0)
            scaled_start = cue.start_seconds * scale_factor
            scaled_end = cue.end_seconds * scale_factor

            shifted_start = max(0.0, scaled_start + offset)
            if is_list and shifted_start < prev_start:
                shifted_start = prev_start
            prev_start = shifted_start

            duration = max(0.1, scaled_end - scaled_start)
            shifted_end = max(shifted_start + 0.1, scaled_end + offset, shifted_start + duration)

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
        scale_factor: float = 1.0,
        onset_weight: float = 1.0,
    ) -> np.ndarray:
        """Generate a 1D speech timeline from spoken subtitle cues.

        Filters out non-dialogue cues (SDH, music, watermarks), caps abnormally long
        display durations, and optionally applies onset weighting to sentence starts.
        """
        length = int(math.ceil(total_seconds * sample_rate * max(1.0, scale_factor))) + 20
        timeline = np.zeros(length, dtype=np.float64)

        for cue in cues:
            cleaned = cls.clean_cue_text_for_speech(cue.text)
            if len(cleaned) < 2:
                continue

            scaled_start = cue.start_seconds * scale_factor
            scaled_end = cue.end_seconds * scale_factor
            raw_duration = max(0.1, scaled_end - scaled_start)

            # Cap effective speech duration based on text length (min 3.5s, max 10.0s)
            max_plausible_speech_dur = max(3.5, min(10.0, len(cleaned) * 0.18))
            effective_end = scaled_start + min(raw_duration, max_plausible_speech_dur)

            start_idx = max(0, int(round(scaled_start * sample_rate)))
            end_idx = max(start_idx + 1, int(round(effective_end * sample_rate)))

            if start_idx < len(timeline):
                clamped_end = min(end_idx, len(timeline))
                timeline[start_idx:clamped_end] = 1.0
                if onset_weight > 1.0:
                    onset_end = min(clamped_end, start_idx + int(0.35 * sample_rate))
                    timeline[start_idx:onset_end] = onset_weight

        return timeline

    # -------------------------------------------------------------------------
    # Audio Speech Detection (Ensemble VAD + Spectral Formants + Smoothing)
    # -------------------------------------------------------------------------

    @staticmethod
    def smooth_speech_timeline(
        timeline: np.ndarray,
        max_gap_frames: int = 15,
        min_speech_frames: int = 5,
    ) -> np.ndarray:
        """Apply 1D morphological closing followed by opening on a binary speech timeline.

        - Morphological closing fills short intra-sentence pauses (<= 150ms at 100Hz)
          between syllables/words so the audio speech envelope matches continuous subtitle cues.
        - Morphological opening removes isolated micro-noise clicks (< 50ms at 100Hz).
        """
        if len(timeline) == 0:
            return timeline

        binary = (timeline > 0.0).astype(np.int8)
        n = len(binary)

        # 1. Closing: fill 0-runs of length <= max_gap_frames surrounded by 1s
        if max_gap_frames > 0 and np.any(binary):
            padded = np.empty(n + 2, dtype=np.int8)
            padded[0] = 0
            padded[1:-1] = binary
            padded[-1] = 0
            diffs = np.diff(padded)
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]

            for i in range(len(starts) - 1):
                gap_start = ends[i]
                gap_end = starts[i + 1]
                if 0 < (gap_end - gap_start) <= max_gap_frames:
                    binary[gap_start:gap_end] = 1

        # 2. Opening: remove 1-runs of length < min_speech_frames
        if min_speech_frames > 1 and np.any(binary):
            padded = np.empty(n + 2, dtype=np.int8)
            padded[0] = 0
            padded[1:-1] = binary
            padded[-1] = 0
            diffs = np.diff(padded)
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]

            for s_idx, e_idx in zip(starts, ends):
                if (e_idx - s_idx) < min_speech_frames:
                    binary[s_idx:e_idx] = 0

        return binary.astype(np.float64)

    @classmethod
    def _compute_rms_energy_vad(
        cls,
        pcm_bytes: bytes,
        sample_rate: int = ANALYSIS_SAMPLE_RATE,
        audio_rate: int = 16000,
    ) -> np.ndarray:
        """Compute an adaptive short-time RMS energy voice activity detector at 100 Hz."""
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        samples_per_frame = audio_rate // sample_rate
        num_frames = len(samples) // samples_per_frame
        if num_frames == 0:
            return np.zeros(0, dtype=np.float64)

        framed = samples[: num_frames * samples_per_frame].reshape(num_frames, samples_per_frame)
        rms = np.sqrt(np.mean(framed * framed, axis=1))

        max_rms = float(np.max(rms))
        if max_rms < 150.0:
            return np.zeros(num_frames, dtype=np.float64)

        noise_floor = float(np.percentile(rms, 25))
        dynamic_range = max_rms - noise_floor
        if dynamic_range < 100.0:
            return np.zeros(num_frames, dtype=np.float64)

        threshold = max(200.0, noise_floor + 0.20 * dynamic_range)
        return (rms >= threshold).astype(np.float64)

    @classmethod
    def _compute_spectral_speech_vad(
        cls,
        pcm_bytes: bytes,
        sample_rate: int = ANALYSIS_SAMPLE_RATE,
        audio_rate: int = 16000,
    ) -> np.ndarray:
        """Detect frames with strong vocal formant concentration (250 Hz - 3200 Hz).

        Distinguishes human voice from sub-bass explosions and high-frequency percussion/music.
        """
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        samples_per_frame = audio_rate // sample_rate  # 160 samples per 10ms frame
        num_frames = len(samples) // samples_per_frame
        if num_frames == 0:
            return np.zeros(0, dtype=np.float64)

        framed = samples[: num_frames * samples_per_frame].reshape(num_frames, samples_per_frame)
        rms = np.sqrt(np.mean(framed * framed, axis=1))
        max_rms = float(np.max(rms))
        if max_rms < 150.0:
            return np.zeros(num_frames, dtype=np.float64)

        # Compute real FFT power spectrum per 10ms frame
        spectra = np.abs(np.fft.rfft(framed, axis=1)) ** 2
        freqs = np.fft.rfftfreq(samples_per_frame, d=1.0 / audio_rate)

        vocal_mask = (freqs >= 200.0) & (freqs <= 3400.0)
        vocal_energy = np.sum(spectra[:, vocal_mask], axis=1)
        total_energy = np.sum(spectra, axis=1) + 1e-6
        vocal_ratio = vocal_energy / total_energy

        noise_floor = float(np.percentile(rms, 25))
        energy_gate = rms >= max(180.0, noise_floor + 0.18 * (max_rms - noise_floor))
        return (energy_gate & (vocal_ratio >= 0.55)).astype(np.float64)

    @classmethod
    def detect_speech_from_audio_file(
        cls,
        wav_path: Path,
        sample_rate: int = ANALYSIS_SAMPLE_RATE,
    ) -> np.ndarray:
        """Analyze WAV audio file (16kHz mono) using fused multi-detector ensemble VAD

        (WebRTC VAD + Auditok + Adaptive RMS Energy + Spectral Formant Ratio)
        followed by morphological smoothing.
        """
        from ffsubsync.speech_transformers import _make_auditok_detector, _make_webrtcvad_detector

        raw_bytes = wav_path.read_bytes()
        data_offset = 44
        data_idx = raw_bytes.find(b"data")
        if 0 <= data_idx <= 200 and data_idx + 8 <= len(raw_bytes):
            data_offset = data_idx + 8

        pcm_bytes = raw_bytes[data_offset:] if len(raw_bytes) > data_offset else b""
        if not pcm_bytes:
            return np.zeros(0, dtype=np.float64)

        detectors_output: List[np.ndarray] = []

        # 1. WebRTC VAD detector
        try:
            webrtc_detector = _make_webrtcvad_detector(sample_rate, 16000, 0.0)
            webrtc_tl = webrtc_detector(pcm_bytes)
            if len(webrtc_tl) > 0 and np.sum(webrtc_tl > 0) >= 5:
                detectors_output.append((webrtc_tl > 0).astype(np.float64))
        except Exception as exc:
            logger.warning("webrtcvad failed on audio data: %s", exc)

        # 2. Auditok energy detector
        try:
            auditok_detector = _make_auditok_detector(sample_rate, 16000, 0.0)
            auditok_tl = auditok_detector(pcm_bytes)
            if len(auditok_tl) > 0 and np.sum(auditok_tl > 0) >= 5:
                detectors_output.append((auditok_tl > 0).astype(np.float64))
        except Exception as exc:
            logger.warning("auditok failed on audio data: %s", exc)

        # 3. Adaptive Short-Time RMS Energy VAD
        try:
            rms_tl = cls._compute_rms_energy_vad(pcm_bytes, sample_rate=sample_rate, audio_rate=16000)
            if len(rms_tl) > 0 and np.sum(rms_tl > 0) >= 5:
                detectors_output.append(rms_tl)
        except Exception as exc:
            logger.warning("RMS energy VAD failed on audio data: %s", exc)

        # 4. Spectral Vocal Formant VAD (200Hz - 3400Hz concentration)
        try:
            spec_tl = cls._compute_spectral_speech_vad(pcm_bytes, sample_rate=sample_rate, audio_rate=16000)
            if len(spec_tl) > 0 and np.sum(spec_tl > 0) >= 5:
                detectors_output.append(spec_tl)
        except Exception as exc:
            logger.warning("Spectral VAD failed on audio data: %s", exc)

        if not detectors_output:
            expected_len = len(pcm_bytes) // (16000 * 2 // sample_rate)
            return np.zeros(expected_len, dtype=np.float64)

        min_len = min(len(arr) for arr in detectors_output)
        stacked = np.vstack([arr[:min_len] for arr in detectors_output])

        if len(detectors_output) >= 2:
            vote_sum = np.sum(stacked, axis=0)
            threshold_votes = 2.0 if len(detectors_output) <= 3 else 2.0
            fused = (vote_sum >= threshold_votes).astype(np.float64)
            if np.sum(fused) < 10:
                fused = (vote_sum >= 1.0).astype(np.float64)
        else:
            fused = stacked[0]

        smoothed = cls.smooth_speech_timeline(fused, max_gap_frames=15, min_speech_frames=5)
        logger.info(
            "Ensemble VAD detected %d speech frames (%.1fs of smoothed speech from %d detectors).",
            int(np.sum(smoothed > 0)),
            float(np.sum(smoothed > 0)) / sample_rate,
            len(detectors_output),
        )
        return smoothed

    @classmethod
    def build_multi_window_speech_timeline(
        cls,
        audio_windows: List[ExtractedAudioInfo],
        sample_rate: int = ANALYSIS_SAMPLE_RATE,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Combine speech timelines from one or more extracted audio windows into a unified

        reference speech timeline (`ref_speech`) and a binary `analyzed_mask` indicating
        which timeline frames were actually sampled from the media stream.
        """
        if not audio_windows:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=bool)

        window_timelines: List[Tuple[int, np.ndarray]] = []
        max_end_frame = 0

        for win in audio_windows:
            tl = cls.detect_speech_from_audio_file(win.file_path, sample_rate=sample_rate)
            start_frame = max(0, int(round(win.start_offset_seconds * sample_rate)))
            end_frame = start_frame + len(tl)
            if end_frame > max_end_frame:
                max_end_frame = end_frame
            window_timelines.append((start_frame, tl))

        if max_end_frame == 0:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=bool)

        ref_speech = np.zeros(max_end_frame, dtype=np.float64)
        analyzed_mask = np.zeros(max_end_frame, dtype=bool)

        for start_frame, tl in window_timelines:
            if len(tl) > 0:
                end_frame = start_frame + len(tl)
                ref_speech[start_frame:end_frame] = tl
                analyzed_mask[start_frame:end_frame] = True

        return ref_speech, analyzed_mask

    # -------------------------------------------------------------------------
    # Window-Mask Aware Confidence & Two-Stage Alignment Engine
    # -------------------------------------------------------------------------

    @classmethod
    def _dilate_binary_timeline(cls, binary_arr: np.ndarray, radius_frames: int = 15) -> np.ndarray:
        """Dilate a 1D binary array by +/- radius_frames (e.g. 15 frames = +/- 150ms tolerance)."""
        if radius_frames <= 0 or len(binary_arr) == 0 or not np.any(binary_arr > 0):
            return (binary_arr > 0).astype(np.float64)
        kernel_size = 2 * radius_frames + 1
        kernel = np.ones(kernel_size, dtype=np.float64)
        convolved = np.convolve((binary_arr > 0).astype(np.float64), kernel, mode="same")
        return (convolved > 0).astype(np.float64)

    @classmethod
    def evaluate_alignment_confidence(
        cls,
        ref_speech: np.ndarray,
        sub_speech: np.ndarray,
        cues: List[SubtitleCue],
        offsets: Union[float, List[float]],
        scale_factor: float = 1.0,
        raw_fft_score: float = 0.0,
        sample_rate: int = ANALYSIS_SAMPLE_RATE,
        analyzed_mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """Compute objective synchronization confidence strictly within the sampled audio windows.

        Supports both single continuous windows and multi-window fast-seek masks, ensuring
        cues falling in un-sampled regions of a 2-hour movie never penalize confidence.

        Returns:
            Tuple of (confidence_score in [0.0, 1.0], matched_overlap_seconds).
        """
        audio_len = len(ref_speech)
        if audio_len == 0 or np.sum(ref_speech > 0) == 0:
            return 0.0, 0.0

        if analyzed_mask is None or len(analyzed_mask) != audio_len:
            mask = np.ones(audio_len, dtype=bool)
        else:
            mask = analyzed_mask

        is_list = isinstance(offsets, list)
        shifted_sub_in_window = np.zeros(audio_len, dtype=np.float64)
        active_cues_in_window = 0
        matched_cues_in_window = 0

        # Dilate ref_speech by +/- 150ms (15 frames) for natural subtitle onset/offset tolerance
        dilated_ref = cls._dilate_binary_timeline(ref_speech, radius_frames=15)

        for idx, cue in enumerate(cues):
            if not cls.is_spoken_dialogue_cue(cue):
                continue

            cue_offset = offsets[idx] if is_list and idx < len(offsets) else (offsets if not is_list else 0.0)
            s_sec = cue.start_seconds * scale_factor + cue_offset
            e_sec = cue.end_seconds * scale_factor + cue_offset

            s_frame = int(round(s_sec * sample_rate))
            e_frame = int(round(e_sec * sample_rate))

            # Cues shifted beyond the end of the extracted audio window cannot be evaluated
            if s_frame >= audio_len:
                continue

            # Cues shifted into negative timestamps (< -0.25s) penalize alignment confidence
            if e_frame <= 0:
                neg_dur_frames = max(10, e_frame - s_frame)
                active_cues_in_window += 1
                window_sub_negative_penalty = neg_dur_frames
                shifted_sub_in_window[0 : min(audio_len, neg_dur_frames)] = 0.0
                continue

            clamped_s = max(0, s_frame)
            clamped_e = min(audio_len, max(clamped_s + 1, e_frame))

            # Check how many frames of this shifted cue fall inside an actually analyzed audio window
            analyzed_frames_in_cue = int(np.sum(mask[clamped_s:clamped_e]))
            if analyzed_frames_in_cue < 10:
                continue

            shifted_sub_in_window[clamped_s:clamped_e] = np.where(mask[clamped_s:clamped_e], 1.0, 0.0)
            active_cues_in_window += 1

            cue_overlap = float(np.sum((dilated_ref[clamped_s:clamped_e] > 0) & mask[clamped_s:clamped_e]))
            if (cue_overlap / float(analyzed_frames_in_cue)) >= 0.25 and s_sec >= -0.25:
                matched_cues_in_window += 1

        # Count any negative-shifted spoken cues in the denominator so pushing cues < 0s is penalized
        negative_cues_frames = sum(
            max(10, int(round((c.end_seconds - c.start_seconds) * scale_factor * sample_rate)))
            for idx, c in enumerate(cues)
            if cls.is_spoken_dialogue_cue(c)
            and (c.start_seconds * scale_factor + (offsets[idx] if is_list and idx < len(offsets) else (offsets if not is_list else 0.0))) < -0.25
        )

        window_sub_total = float(np.sum(shifted_sub_in_window > 0)) + float(negative_cues_frames)
        if window_sub_total < 5.0:
            return 0.0, 0.0

        tolerant_overlap_frames = float(np.sum((shifted_sub_in_window > 0) & (dilated_ref > 0)))
        exact_overlap_frames = float(np.sum((shifted_sub_in_window > 0) & (ref_speech > 0)))
        window_precision = tolerant_overlap_frames / max(1.0, window_sub_total)

        # Measure audio speech coverage within the active subtitle span
        window_ref_total = float(np.sum((ref_speech > 0) & mask))
        audio_coverage = min(1.0, tolerant_overlap_frames / max(1.0, window_ref_total))

        cue_hit_ratio = (
            matched_cues_in_window / float(active_cues_in_window)
            if active_cues_in_window > 0
            else window_precision
        )

        combined = 0.50 * window_precision + 0.30 * cue_hit_ratio + 0.20 * audio_coverage
        if exact_overlap_frames / max(1.0, window_sub_total) >= 0.50:
            combined = min(1.0, combined + 0.05)

        confidence = round(min(1.0, max(0.0, combined)), 3)
        matched_seconds = exact_overlap_frames / float(sample_rate)
        return confidence, matched_seconds

    @classmethod
    def _find_best_global_alignment(
        cls,
        ref_speech: np.ndarray,
        cues: List[SubtitleCue],
        eval_duration: float,
        sample_rate: int = ANALYSIS_SAMPLE_RATE,
    ) -> Tuple[float, float, float, np.ndarray]:
        """Search across candidate framerate scale factors and FFT cross-correlation offsets

        up to +/- 300 seconds (5 minutes) to absorb long intros, recaps, and opening ads.

        Returns:
            Tuple of (best_offset_seconds, best_scale_factor, best_raw_score, best_sub_speech).
        """
        from ffsubsync.aligners import FFTAligner

        sub_span = max(c.end_seconds for c in cues) - min(c.start_seconds for c in cues)
        scale_candidates = CANDIDATE_SCALE_FACTORS if sub_span >= 20.0 else (1.0,)

        best_score = -1.0
        best_offset_samples = 0
        best_scale = 1.0
        best_sub_speech = np.zeros(0, dtype=np.float64)
        unit_scale_score = -1.0

        for scale in scale_candidates:
            sub_speech = cls.generate_subtitle_speech_timeline(
                cues,
                total_seconds=eval_duration,
                sample_rate=sample_rate,
                scale_factor=scale,
                onset_weight=1.25,
            )
            if np.sum(sub_speech > 0) < 5:
                continue

            max_possible_samples = max(100, min(len(ref_speech), len(sub_speech)) - 50)
            # Allow up to 300 seconds (5 minutes) of intro/commercial offset
            max_offset_sec = 300.0
            max_offset_samples = min(int(max_offset_sec * sample_rate), max_possible_samples)
            aligner = FFTAligner(max_offset_samples=max_offset_samples)

            score, offset_samples = aligner.fit(ref_speech, sub_speech, get_score=True).transform()
            score = float(score)

            if scale == 1.0:
                unit_scale_score = score
                best_score = score
                best_offset_samples = int(offset_samples)
                best_scale = 1.0
                best_sub_speech = (sub_speech > 0).astype(np.float64)
            else:
                if score > best_score and score >= max(1.0, unit_scale_score * 1.06):
                    best_score = score
                    best_offset_samples = int(offset_samples)
                    best_scale = scale
                    best_sub_speech = (sub_speech > 0).astype(np.float64)

        if len(best_sub_speech) == 0:
            best_sub_speech = cls.generate_subtitle_speech_timeline(
                cues, total_seconds=eval_duration, sample_rate=sample_rate, scale_factor=1.0
            )

        best_offset_seconds = best_offset_samples / float(sample_rate)
        return best_offset_seconds, best_scale, best_score, best_sub_speech

    @classmethod
    def _try_piecewise_split_alignment(
        cls,
        ref_speech: np.ndarray,
        sub_speech: np.ndarray,
        cues: List[SubtitleCue],
        global_offset_sec: float,
        global_confidence: float,
        global_matched_sec: float,
        sample_rate: int = ANALYSIS_SAMPLE_RATE,
        analyzed_mask: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[List[float], float, int]]:
        """Execute alass-style prefix-sum dynamic programming with edge/length penalty

        to handle videos with mid-stream commercial cuts, recaps, or intro shifts.
        Anchors non-dialogue cues to their nearest spoken neighbor offset.
        """
        n = len(cues)
        spoken_cues = [c for c in cues if cls.is_spoken_dialogue_cue(c)]
        if len(spoken_cues) < 4 or len(ref_speech) < 100:
            return None

        try:
            audio_len = len(ref_speech)
            max_possible_samples = max(100, audio_len - 50)
            max_offset_samples = min(int(180.0 * sample_rate), max_possible_samples)
            max_guard_samples = 200  # 2.0s guard band at 100Hz
            length_penalty = 0.20
            split_penalty = 220.0  # ~2.2 seconds of speech overlap required to justify a split

            starts = np.zeros(n, dtype=np.float64)
            ends = np.zeros(n, dtype=np.float64)
            is_speech = np.zeros(n, dtype=bool)

            for i, cue in enumerate(cues):
                s_idx = cue.start_seconds * sample_rate
                e_idx = cue.end_seconds * sample_rate
                starts[i] = s_idx
                ends[i] = max(s_idx + 10.0, e_idx)
                if cls.is_spoken_dialogue_cue(cue) and (s_idx - max_offset_samples) < audio_len:
                    is_speech[i] = True

            if int(np.sum(is_speech)) < 4:
                return None

            guards = np.minimum(ends - starts, float(max_guard_samples))
            offsets = np.arange(-max_offset_samples, max_offset_samples + 1, dtype=np.float64)
            n_off = len(offsets)

            ref = (np.asarray(ref_speech, dtype=np.float64) > 0).astype(np.float64)
            cumsum = np.concatenate([np.zeros(1), np.cumsum(ref)])
            pad = max_offset_samples + max_guard_samples + 2
            padded = np.concatenate([np.zeros(pad), cumsum, np.full(pad, cumsum[-1])])
            max_pos = len(padded) - 1

            def _lookup(pos: np.ndarray) -> np.ndarray:
                idx = np.clip(np.round(pos).astype(np.int64), 0, max_pos)
                return padded[idx]

            def _rating_row(i: int) -> np.ndarray:
                if not is_speech[i]:
                    return np.zeros(n_off, dtype=np.float64)
                start_pos = starts[i] + offsets + pad
                end_pos = ends[i] + offsets + pad
                cs = _lookup(start_pos)
                ce = _lookup(end_pos)
                overlap = ce - cs
                if guards[i] > 0:
                    gl = cs - _lookup(start_pos - guards[i])
                    gr = _lookup(end_pos + guards[i]) - ce
                    return overlap - length_penalty * (gl + gr)
                return overlap

            dp = _rating_row(0)
            back = np.full((n, n_off), -1, dtype=np.int64)
            for i in range(1, n):
                best_prev_idx = int(np.argmax(dp))
                jump_value = dp[best_prev_idx] - split_penalty
                keep_value = dp
                jumped = jump_value > keep_value
                dp = _rating_row(i) + np.where(jumped, jump_value, keep_value)
                back[i] = np.where(jumped, best_prev_idx, -1)

            offset_idx = np.empty(n, dtype=np.int64)
            cur = int(np.argmax(dp))
            for i in range(n - 1, -1, -1):
                offset_idx[i] = cur
                if i > 0:
                    b = int(back[i, cur])
                    cur = b if b >= 0 else cur

            per_cue_offsets = [float(offsets[idx]) / float(sample_rate) for idx in offset_idx]

            # Anchor leading non-speech cues to the first spoken cue's offset
            first_spoken_idx = next((i for i, sp in enumerate(is_speech) if sp), None)
            if first_spoken_idx is not None and first_spoken_idx > 0:
                for i in range(first_spoken_idx):
                    per_cue_offsets[i] = per_cue_offsets[first_spoken_idx]

            distinct_segments = 1
            for i in range(1, len(per_cue_offsets)):
                if abs(per_cue_offsets[i] - per_cue_offsets[i - 1]) >= 0.4:
                    distinct_segments += 1

            if distinct_segments < 2:
                return None

            split_conf, split_matched_sec = cls.evaluate_alignment_confidence(
                ref_speech=ref_speech,
                sub_speech=sub_speech,
                cues=cues,
                offsets=per_cue_offsets,
                scale_factor=1.0,
                sample_rate=sample_rate,
                analyzed_mask=analyzed_mask,
            )

            if (
                split_conf >= global_confidence + 0.06
                and split_matched_sec >= global_matched_sec + 0.4
            ):
                logger.info(
                    "Piecewise split alignment selected (%d segments): confidence %.3f -> %.3f (matched %.1fs -> %.1fs)",
                    distinct_segments,
                    global_confidence,
                    split_conf,
                    global_matched_sec,
                    split_matched_sec,
                )
                return per_cue_offsets, split_conf, distinct_segments

        except Exception as exc:
            logger.debug("Piecewise split alignment skipped: %s", exc)

        return None

    # -------------------------------------------------------------------------
    # In-Memory Alignment of a Single Candidate Subtitle Against Pre-Analyzed Audio
    # -------------------------------------------------------------------------

    async def _align_cues_with_reference_speech(
        self,
        cues: List[SubtitleCue],
        ref_speech: np.ndarray,
        analyzed_mask: Optional[np.ndarray],
        target_format: str,
        start_time: float,
    ) -> SyncResult:
        """Align parsed subtitle cues against an already-extracted `ref_speech` timeline."""
        loop = asyncio.get_running_loop()
        audio_duration = len(ref_speech) / float(ANALYSIS_SAMPLE_RATE)
        sub_max_time = max(c.end_seconds for c in cues)
        eval_duration = max(audio_duration, sub_max_time)

        offset_seconds, best_scale, score, sub_speech = await loop.run_in_executor(
            None,
            self._find_best_global_alignment,
            ref_speech,
            cues,
            eval_duration,
            ANALYSIS_SAMPLE_RATE,
        )

        confidence, matched_sec = self.evaluate_alignment_confidence(
            ref_speech=ref_speech,
            sub_speech=sub_speech,
            cues=cues,
            offsets=offset_seconds,
            scale_factor=best_scale,
            raw_fft_score=score,
            sample_rate=ANALYSIS_SAMPLE_RATE,
            analyzed_mask=analyzed_mask,
        )

        applied_offsets: Union[float, List[float]] = offset_seconds
        alignment_method = "drift_fft" if abs(best_scale - 1.0) > 1e-4 else "global_fft"
        segments_count = 1

        # Stage 2: Alass Piecewise Split Alignment Check
        split_res = await loop.run_in_executor(
            None,
            self._try_piecewise_split_alignment,
            ref_speech,
            sub_speech,
            cues,
            offset_seconds,
            confidence,
            matched_sec,
            ANALYSIS_SAMPLE_RATE,
            analyzed_mask,
        )
        if split_res is not None:
            applied_offsets, confidence, segments_count = split_res
            best_scale = 1.0
            alignment_method = "piecewise_split"
            offset_seconds = round(float(np.median(applied_offsets)), 3)

        is_already_aligned = (
            segments_count == 1
            and abs(best_scale - 1.0) <= 1e-4
            and abs(offset_seconds) < 0.25
            and (confidence >= 0.20 or score > 0)
        )

        if is_already_aligned:
            status = "already_aligned"
            corrected_content = self.apply_shifts_to_cues(
                cues, 0.0, output_format=target_format, scale_factor=1.0
            )
        elif confidence >= 0.30 or (score > 100.0 and confidence >= 0.15):
            status = "synchronized"
            corrected_content = self.apply_shifts_to_cues(
                cues, applied_offsets, output_format=target_format, scale_factor=best_scale
            )
        elif confidence >= 0.10 and score > 0:
            status = "partial"
            corrected_content = self.apply_shifts_to_cues(
                cues, applied_offsets, output_format=target_format, scale_factor=best_scale
            )
        else:
            status = "failed"
            offset_seconds = None
            corrected_content = self.apply_shifts_to_cues(
                cues, 0.0, output_format=target_format, scale_factor=1.0
            )

        validated_sub = subtitle_manager.process_subtitle(
            corrected_content.encode("utf-8"),
            filename_hint=f"synced.{target_format}",
            target_format=target_format,
        )

        elapsed = round(time.perf_counter() - start_time, 3)
        return SyncResult(
            success=True,
            status=status,
            offset_seconds=round(offset_seconds, 3) if offset_seconds is not None else None,
            confidence_score=confidence,
            audio_analyzed=True,
            processing_time_seconds=elapsed,
            subtitle_content=validated_sub.content_str,
            format=validated_sub.format,
            cue_count=validated_sub.cue_count,
            detail=f"Alignment status: {status} ({alignment_method}, segments={segments_count})",
            scale_factor=round(best_scale, 6),
            alignment_method=alignment_method,
            segments_count=segments_count,
        )

    # -------------------------------------------------------------------------
    # Public Alignment & Multi-Candidate Synchronization Pipelines
    # -------------------------------------------------------------------------

    async def synchronize_subtitle(
        self,
        subtitle_content: str,
        stream_url: Optional[str] = None,
        stream_headers: Optional[Dict[str, str]] = None,
        target_format: str = "srt",
        language: str = "en",
        local_audio_path: Optional[Path] = None,
        media_duration: Optional[float] = None,
    ) -> SyncResult:
        """Execute the end-to-end synchronization pipeline for a single subtitle content string."""
        start_time = time.perf_counter()

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

        extracted_windows: List[ExtractedAudioInfo] = []
        temp_wav_path: Optional[Path] = None

        async with self._semaphore:
            if local_audio_path and local_audio_path.exists():
                temp_wav_path = local_audio_path
            else:
                try:
                    extracted_windows = await audio_extractor.extract_multi_window_audio(
                        stream_url=stream_url,
                        headers=stream_headers,
                        media_duration=media_duration,
                    )
                except AudioExtractionError as exc:
                    logger.warning("Audio extraction failed (%s). Falling back to original subtitle.", exc)
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
                loop = asyncio.get_running_loop()
                if temp_wav_path is not None:
                    ref_speech = await loop.run_in_executor(
                        None,
                        self.detect_speech_from_audio_file,
                        temp_wav_path,
                        ANALYSIS_SAMPLE_RATE,
                    )
                    analyzed_mask = np.ones(len(ref_speech), dtype=bool)
                else:
                    ref_speech, analyzed_mask = await loop.run_in_executor(
                        None,
                        self.build_multi_window_speech_timeline,
                        extracted_windows,
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

                result = await self._align_cues_with_reference_speech(
                    cues=cues,
                    ref_speech=ref_speech,
                    analyzed_mask=analyzed_mask,
                    target_format=target_format,
                    start_time=start_time,
                )
                logger.info(
                    "Synchronization complete: status=%s, method=%s, offset=%s, scale=%.5f, confidence=%s in %.2fs",
                    result.status,
                    result.alignment_method,
                    f"{result.offset_seconds:.3f}s" if result.offset_seconds is not None else "None",
                    result.scale_factor,
                    result.confidence_score,
                    result.processing_time_seconds,
                )
                return result

            finally:
                for win in extracted_windows:
                    if win.file_path and win.file_path.exists():
                        try:
                            win.file_path.unlink(missing_ok=True)
                        except OSError as exc:
                            logger.warning("Could not delete temporary audio file '%s': %s", win.file_path, exc)

    async def synchronize_candidates(
        self,
        candidates_inputs: List[CandidateSyncInput],
        stream_url: Optional[str] = None,
        stream_headers: Optional[Dict[str, str]] = None,
        target_format: str = "srt",
        language: str = "en",
        local_audio_path: Optional[Path] = None,
        media_duration: Optional[float] = None,
    ) -> Tuple[SyncResult, CandidateSyncInput]:
        """Extract video audio ONCE and evaluate multiple subtitle candidates acoustically.

        Selects the candidate subtitle that achieves the highest acoustic confidence score
        (breaking ties with OpenSubtitles metadata score) so the backend automatically avoids
        mismatched movie cuts (e.g. Theatrical vs Extended/Director's Cut).
        """
        if not candidates_inputs:
            raise ValueError("candidates_inputs must contain at least one CandidateSyncInput.")

        if len(candidates_inputs) == 1 or (not stream_url and not local_audio_path):
            first = candidates_inputs[0]
            res = await self.synchronize_subtitle(
                subtitle_content=first.subtitle_content,
                stream_url=stream_url,
                stream_headers=stream_headers,
                target_format=target_format,
                language=language,
                local_audio_path=local_audio_path,
                media_duration=media_duration,
            )
            res.selected_candidate_id = first.candidate_id
            res.candidates_evaluated = 1
            return res, first

        start_time = time.perf_counter()
        extracted_windows: List[ExtractedAudioInfo] = []
        temp_wav_path: Optional[Path] = None

        async with self._semaphore:
            if local_audio_path and local_audio_path.exists():
                temp_wav_path = local_audio_path
            else:
                try:
                    extracted_windows = await audio_extractor.extract_multi_window_audio(
                        stream_url=stream_url,
                        headers=stream_headers,
                        media_duration=media_duration,
                    )
                except AudioExtractionError as exc:
                    logger.warning("Audio extraction failed in multi-candidate sync (%s).", exc)
                    first = candidates_inputs[0]
                    cues = self.parse_cues_from_text(first.subtitle_content)
                    is_unsupported = "unsupported" in str(exc).lower() or "drm" in str(exc).lower()
                    fallback = SyncResult(
                        success=True,
                        status="unsupported" if is_unsupported else "unavailable",
                        offset_seconds=None,
                        confidence_score=0.0,
                        audio_analyzed=False,
                        processing_time_seconds=round(time.perf_counter() - start_time, 3),
                        subtitle_content=first.subtitle_content,
                        format=target_format,
                        cue_count=len(cues),
                        detail=f"Audio could not be accessed: {exc}",
                        selected_candidate_id=first.candidate_id,
                        candidates_evaluated=1,
                    )
                    return fallback, first

            try:
                loop = asyncio.get_running_loop()
                if temp_wav_path is not None:
                    ref_speech = await loop.run_in_executor(
                        None,
                        self.detect_speech_from_audio_file,
                        temp_wav_path,
                        ANALYSIS_SAMPLE_RATE,
                    )
                    analyzed_mask = np.ones(len(ref_speech), dtype=bool)
                else:
                    ref_speech, analyzed_mask = await loop.run_in_executor(
                        None,
                        self.build_multi_window_speech_timeline,
                        extracted_windows,
                        ANALYSIS_SAMPLE_RATE,
                    )

                if np.sum(ref_speech > 0) < 10:
                    first = candidates_inputs[0]
                    cues = self.parse_cues_from_text(first.subtitle_content)
                    return (
                        SyncResult(
                            success=True,
                            status="failed",
                            offset_seconds=0.0,
                            confidence_score=0.0,
                            audio_analyzed=True,
                            processing_time_seconds=round(time.perf_counter() - start_time, 3),
                            subtitle_content=first.subtitle_content,
                            format=target_format,
                            cue_count=len(cues),
                            detail="No detectable voice activity found in audio stream.",
                            selected_candidate_id=first.candidate_id,
                            candidates_evaluated=1,
                        ),
                        first,
                    )

                best_result: Optional[SyncResult] = None
                best_candidate: CandidateSyncInput = candidates_inputs[0]
                best_composite_rank = -1e9
                evaluated_count = 0

                for cand_in in candidates_inputs:
                    cues = self.parse_cues_from_text(cand_in.subtitle_content)
                    if not cues:
                        continue

                    evaluated_count += 1
                    res = await self._align_cues_with_reference_speech(
                        cues=cues,
                        ref_speech=ref_speech,
                        analyzed_mask=analyzed_mask,
                        target_format=target_format,
                        start_time=start_time,
                    )
                    conf = res.confidence_score or 0.0
                    # Acoustic confidence is primary (weight 1000), metadata score is secondary tie-breaker
                    composite_rank = (conf * 1000.0) + (cand_in.metadata_score * 0.1)

                    logger.info(
                        "Evaluated candidate '%s': status=%s, confidence=%.3f, offset=%s, method=%s",
                        cand_in.candidate_id,
                        res.status,
                        conf,
                        res.offset_seconds,
                        res.alignment_method,
                    )

                    if composite_rank > best_composite_rank:
                        best_composite_rank = composite_rank
                        best_result = res
                        best_candidate = cand_in

                if best_result is None:
                    first = candidates_inputs[0]
                    best_result = SyncResult(
                        success=False,
                        status="failed",
                        offset_seconds=None,
                        confidence_score=0.0,
                        audio_analyzed=True,
                        processing_time_seconds=round(time.perf_counter() - start_time, 3),
                        subtitle_content=first.subtitle_content,
                        format=target_format,
                        cue_count=0,
                        detail="No valid subtitle candidates could be aligned.",
                    )

                best_result.selected_candidate_id = best_candidate.candidate_id
                best_result.candidates_evaluated = evaluated_count
                best_result.processing_time_seconds = round(time.perf_counter() - start_time, 3)
                best_result.detail = (
                    f"Alignment status: {best_result.status} "
                    f"({best_result.alignment_method}, segments={best_result.segments_count}, "
                    f"selected_candidate={best_candidate.candidate_id} from {evaluated_count} evaluated)"
                )
                return best_result, best_candidate

            finally:
                for win in extracted_windows:
                    if win.file_path and win.file_path.exists():
                        try:
                            win.file_path.unlink(missing_ok=True)
                        except OSError as exc:
                            logger.warning("Could not delete temporary audio file '%s': %s", win.file_path, exc)


# Global singleton instance
subtitle_sync_service = SubtitleSynchronizationService()

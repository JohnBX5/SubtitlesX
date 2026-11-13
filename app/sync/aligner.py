"""Two-stage audio-to-subtitle alignment engine.

Stage 1: Extended FFT cross-correlation across standard cinema/broadcast
framerate drift ratios (23.976, 24, 25, 29.97, 30 fps).
Stage 2: Alass-style prefix-sum dynamic programming for piecewise split
alignment (intros, recaps, commercial cuts), with window-mask aware confidence.
"""

from __future__ import annotations

import io
import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from app.subtitles.cleaner import is_dialogue, spoken_text
from app.subtitles.models import Cue
from app.sync.audio import ExtractedAudio
from app.sync.vad import FRAME_RATE, detect_speech

logger = logging.getLogger(__name__)

ANALYSIS_SAMPLE_RATE = FRAME_RATE  # 100 Hz (10ms per frame)

CANDIDATE_SCALE_FACTORS: Tuple[float, ...] = (
    1.0,
    24.0 / 23.976,
    23.976 / 24.0,
    25.0 / 23.976,
    23.976 / 25.0,
    25.0 / 24.0,
    24.0 / 25.0,
)


@dataclass(slots=True)
class SpeechTimeline:
    """Unified 100 Hz speech activity timeline and sampled-window mask."""

    speech: np.ndarray
    mask: np.ndarray

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            speech=(self.speech > 0).astype(np.uint8),
            mask=self.mask.astype(np.uint8),
        )
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, raw: bytes) -> "SpeechTimeline":
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            speech = data["speech"].astype(np.float64)
            mask = data["mask"].astype(bool)
        return cls(speech=speech, mask=mask)


@dataclass(slots=True)
class AlignmentResult:
    """Result of aligning subtitle cues against a speech timeline."""

    status: str  # 'synchronized', 'already_aligned', 'partial', 'failed'
    cues: List[Cue]
    offset_seconds: Optional[float]
    scale_factor: float
    confidence_score: float
    alignment_method: str  # 'global_fft', 'drift_fft', 'piecewise_split'
    segments_count: int
    matched_seconds: float


def build_speech_timeline(
    windows: Sequence[ExtractedAudio],
    sample_rate: int = ANALYSIS_SAMPLE_RATE,
) -> SpeechTimeline:
    """Combine speech timelines from extracted audio windows into a unified timeline."""
    if not windows:
        return SpeechTimeline(
            speech=np.zeros(0, dtype=np.float64),
            mask=np.zeros(0, dtype=bool),
        )

    window_timelines: List[Tuple[int, np.ndarray]] = []
    max_end_frame = 0

    for win in windows:
        tl = detect_speech(win.pcm).astype(np.float64)
        start_frame = max(0, int(round(win.start * sample_rate)))
        end_frame = start_frame + len(tl)
        if end_frame > max_end_frame:
            max_end_frame = end_frame
        window_timelines.append((start_frame, tl))

    if max_end_frame == 0:
        return SpeechTimeline(
            speech=np.zeros(0, dtype=np.float64),
            mask=np.zeros(0, dtype=bool),
        )

    ref_speech = np.zeros(max_end_frame, dtype=np.float64)
    analyzed_mask = np.zeros(max_end_frame, dtype=bool)

    for start_frame, tl in window_timelines:
        if len(tl) > 0:
            end_frame = start_frame + len(tl)
            ref_speech[start_frame:end_frame] = tl
            analyzed_mask[start_frame:end_frame] = True

    return SpeechTimeline(speech=ref_speech, mask=analyzed_mask)


def generate_subtitle_speech_timeline(
    cues: Sequence[Cue],
    total_seconds: float,
    sample_rate: int = ANALYSIS_SAMPLE_RATE,
    scale_factor: float = 1.0,
    onset_weight: float = 1.0,
) -> np.ndarray:
    """Generate a 1D speech timeline from spoken subtitle cues."""
    length = int(math.ceil(total_seconds * sample_rate * max(1.0, scale_factor))) + 20
    timeline = np.zeros(length, dtype=np.float64)

    for cue in cues:
        cleaned = spoken_text(cue.text)
        if len(cleaned) < 2:
            continue

        scaled_start = cue.start * scale_factor
        scaled_end = cue.end * scale_factor
        raw_duration = max(0.1, scaled_end - scaled_start)

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


def dilate_binary_timeline(binary_arr: np.ndarray, radius_frames: int = 15) -> np.ndarray:
    """Dilate a 1D binary array by +/- radius_frames (e.g. 15 frames = +/- 150ms)."""
    if radius_frames <= 0 or len(binary_arr) == 0 or not np.any(binary_arr > 0):
        return (binary_arr > 0).astype(np.float64)
    kernel_size = 2 * radius_frames + 1
    kernel = np.ones(kernel_size, dtype=np.float64)
    convolved = np.convolve((binary_arr > 0).astype(np.float64), kernel, mode="same")
    return (convolved > 0).astype(np.float64)


def evaluate_alignment_confidence(
    ref_speech: np.ndarray,
    sub_speech: np.ndarray,
    cues: Sequence[Cue],
    offsets: Union[float, Sequence[float]],
    scale_factor: float = 1.0,
    raw_fft_score: float = 0.0,
    sample_rate: int = ANALYSIS_SAMPLE_RATE,
    analyzed_mask: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Compute objective synchronization confidence strictly within sampled audio windows."""
    del sub_speech, raw_fft_score  # preserved in signature for parity
    audio_len = len(ref_speech)
    if audio_len == 0 or np.sum(ref_speech > 0) == 0:
        return 0.0, 0.0

    if analyzed_mask is None or len(analyzed_mask) != audio_len:
        mask = np.ones(audio_len, dtype=bool)
    else:
        mask = analyzed_mask

    is_list = isinstance(offsets, (list, tuple, np.ndarray))
    shifted_sub_in_window = np.zeros(audio_len, dtype=np.float64)
    active_cues_in_window = 0
    matched_cues_in_window = 0

    dilated_ref = dilate_binary_timeline(ref_speech, radius_frames=15)

    for idx, cue in enumerate(cues):
        if not is_dialogue(cue):
            continue

        cue_offset = float(offsets[idx]) if is_list and idx < len(offsets) else (float(offsets) if not is_list else 0.0)
        s_sec = cue.start * scale_factor + cue_offset
        e_sec = cue.end * scale_factor + cue_offset

        s_frame = int(round(s_sec * sample_rate))
        e_frame = int(round(e_sec * sample_rate))

        if s_frame >= audio_len:
            continue

        if e_frame <= 0:
            neg_dur_frames = max(10, e_frame - s_frame)
            active_cues_in_window += 1
            shifted_sub_in_window[0 : min(audio_len, neg_dur_frames)] = 0.0
            continue

        clamped_s = max(0, s_frame)
        clamped_e = min(audio_len, max(clamped_s + 1, e_frame))

        analyzed_frames_in_cue = int(np.sum(mask[clamped_s:clamped_e]))
        if analyzed_frames_in_cue < 10:
            continue

        shifted_sub_in_window[clamped_s:clamped_e] = np.where(mask[clamped_s:clamped_e], 1.0, 0.0)
        active_cues_in_window += 1

        cue_overlap = float(np.sum((dilated_ref[clamped_s:clamped_e] > 0) & mask[clamped_s:clamped_e]))
        if (cue_overlap / float(analyzed_frames_in_cue)) >= 0.25 and s_sec >= -0.25:
            matched_cues_in_window += 1

    negative_cues_frames = sum(
        max(10, int(round((c.end - c.start) * scale_factor * sample_rate)))
        for idx, c in enumerate(cues)
        if is_dialogue(c)
        and (
            c.start * scale_factor
            + (float(offsets[idx]) if is_list and idx < len(offsets) else (float(offsets) if not is_list else 0.0))
        )
        < -0.25
    )

    window_sub_total = float(np.sum(shifted_sub_in_window > 0)) + float(negative_cues_frames)
    if window_sub_total < 5.0:
        return 0.0, 0.0

    tolerant_overlap_frames = float(np.sum((shifted_sub_in_window > 0) & (dilated_ref > 0)))
    exact_overlap_frames = float(np.sum((shifted_sub_in_window > 0) & (ref_speech > 0)))
    window_precision = tolerant_overlap_frames / max(1.0, window_sub_total)

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


def find_best_global_alignment(
    ref_speech: np.ndarray,
    cues: Sequence[Cue],
    eval_duration: float,
    sample_rate: int = ANALYSIS_SAMPLE_RATE,
    max_offset_sec: float = 300.0,
) -> Tuple[float, float, float, np.ndarray]:
    """Search across candidate framerate scale factors and FFT cross-correlation offsets."""
    from ffsubsync.aligners import FFTAligner

    sub_span = max(c.end for c in cues) - min(c.start for c in cues)
    scale_candidates = CANDIDATE_SCALE_FACTORS if sub_span >= 20.0 else (1.0,)

    best_score = -1.0
    best_offset_samples = 0
    best_scale = 1.0
    best_sub_speech = np.zeros(0, dtype=np.float64)
    unit_scale_score = -1.0
    ref_arr = np.asarray(ref_speech, dtype=np.float64)

    for scale in scale_candidates:
        sub_speech = generate_subtitle_speech_timeline(
            cues,
            total_seconds=eval_duration,
            sample_rate=sample_rate,
            scale_factor=scale,
            onset_weight=1.25,
        )
        if np.sum(sub_speech > 0) < 5:
            continue

        max_possible_samples = max(100, min(len(ref_arr), len(sub_speech)) - 50)
        max_offset_samples = min(int(max_offset_sec * sample_rate), max_possible_samples)
        aligner = FFTAligner(max_offset_samples=max_offset_samples)

        score, offset_samples = aligner.fit(ref_arr, sub_speech, get_score=True).transform()
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
        best_sub_speech = generate_subtitle_speech_timeline(
            cues, total_seconds=eval_duration, sample_rate=sample_rate, scale_factor=1.0
        )

    best_offset_seconds = best_offset_samples / float(sample_rate)
    return best_offset_seconds, best_scale, best_score, best_sub_speech


def try_piecewise_split_alignment(
    ref_speech: np.ndarray,
    sub_speech: np.ndarray,
    cues: Sequence[Cue],
    global_offset_sec: float,
    global_confidence: float,
    global_matched_sec: float,
    sample_rate: int = ANALYSIS_SAMPLE_RATE,
    analyzed_mask: Optional[np.ndarray] = None,
) -> Optional[Tuple[List[float], float, int]]:
    """Alass-style prefix-sum dynamic programming for mid-stream cuts/recaps."""
    del global_offset_sec  # preserved in signature
    n = len(cues)
    spoken_cues = [c for c in cues if is_dialogue(c)]
    if len(spoken_cues) < 4 or len(ref_speech) < 100:
        return None

    try:
        audio_len = len(ref_speech)
        max_possible_samples = max(100, audio_len - 50)
        max_offset_samples = min(int(180.0 * sample_rate), max_possible_samples)
        max_guard_samples = 200
        length_penalty = 0.20
        split_penalty = 220.0

        starts = np.zeros(n, dtype=np.float64)
        ends = np.zeros(n, dtype=np.float64)
        is_speech = np.zeros(n, dtype=bool)

        for i, cue in enumerate(cues):
            s_idx = cue.start * sample_rate
            e_idx = cue.end * sample_rate
            starts[i] = s_idx
            ends[i] = max(s_idx + 10.0, e_idx)
            if is_dialogue(cue) and (s_idx - max_offset_samples) < audio_len:
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

        split_conf, split_matched_sec = evaluate_alignment_confidence(
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


def shift_cues(
    cues: Sequence[Cue],
    offsets: Union[float, Sequence[float]],
    scale_factor: float = 1.0,
) -> List[Cue]:
    """Return new Cue objects with scaled and shifted start/end timestamps."""
    is_list = isinstance(offsets, (list, tuple, np.ndarray))
    shifted: List[Cue] = []
    prev_start = 0.0
    for idx, cue in enumerate(cues):
        offset = float(offsets[idx]) if is_list and idx < len(offsets) else (float(offsets) if not is_list else 0.0)
        scaled_start = cue.start * scale_factor
        scaled_end = cue.end * scale_factor

        shifted_start = max(0.0, scaled_start + offset)
        if is_list and shifted_start < prev_start:
            shifted_start = prev_start
        prev_start = shifted_start

        duration = max(0.1, scaled_end - scaled_start)
        shifted_end = max(shifted_start + 0.1, scaled_end + offset, shifted_start + duration)
        shifted.append(Cue(start=shifted_start, end=shifted_end, text=cue.text))
    return shifted


def align_cues_to_timeline(
    cues: Sequence[Cue],
    timeline: SpeechTimeline,
    max_offset_seconds: float = 300.0,
) -> AlignmentResult:
    """Run Stage 1 (Global/Drift FFT) + Stage 2 (Piecewise Split DP) against SpeechTimeline."""
    ref_speech = timeline.speech
    analyzed_mask = timeline.mask
    audio_duration = len(ref_speech) / float(ANALYSIS_SAMPLE_RATE)
    sub_max_time = max(c.end for c in cues)
    eval_duration = max(audio_duration, sub_max_time)

    offset_seconds, best_scale, score, sub_speech = find_best_global_alignment(
        ref_speech=ref_speech,
        cues=cues,
        eval_duration=eval_duration,
        sample_rate=ANALYSIS_SAMPLE_RATE,
        max_offset_sec=max_offset_seconds,
    )

    confidence, matched_sec = evaluate_alignment_confidence(
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

    split_res = try_piecewise_split_alignment(
        ref_speech=ref_speech,
        sub_speech=sub_speech,
        cues=cues,
        global_offset_sec=offset_seconds,
        global_confidence=confidence,
        global_matched_sec=matched_sec,
        sample_rate=ANALYSIS_SAMPLE_RATE,
        analyzed_mask=analyzed_mask,
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
        aligned_cues = shift_cues(cues, 0.0, scale_factor=1.0)
        final_offset: Optional[float] = round(offset_seconds, 3)
    elif confidence >= 0.30 or (score > 100.0 and confidence >= 0.15):
        status = "synchronized"
        aligned_cues = shift_cues(cues, applied_offsets, scale_factor=best_scale)
        final_offset = round(offset_seconds, 3)
    elif confidence >= 0.10 and score > 0:
        status = "partial"
        aligned_cues = shift_cues(cues, applied_offsets, scale_factor=best_scale)
        final_offset = round(offset_seconds, 3)
    else:
        status = "failed"
        final_offset = None
        aligned_cues = shift_cues(cues, 0.0, scale_factor=1.0)

    return AlignmentResult(
        status=status,
        cues=aligned_cues,
        offset_seconds=final_offset,
        scale_factor=round(best_scale, 6),
        confidence_score=confidence,
        alignment_method=alignment_method,
        segments_count=segments_count,
        matched_seconds=matched_sec,
    )


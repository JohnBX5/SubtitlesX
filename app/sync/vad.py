"""Voice activity detection: 16 kHz PCM -> 100 Hz binary speech timeline.

Three complementary detectors are fused by weighted vote:

* WebRTC VAD (GMM based, robust to stationary noise),
* adaptive log-energy detector (percentile noise floor),
* speech-band spectral ratio (300-3400 Hz share of frame energy),

followed by morphological closing/opening so the speech envelope resembles
subtitle cue blocks (words joined into phrases, clicks removed).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

FRAME_RATE = 100  # analysis frames per second (10 ms)
AUDIO_RATE = 16000
SAMPLES_PER_FRAME = AUDIO_RATE // FRAME_RATE

try:  # pragma: no cover - import guard
    import webrtcvad  # type: ignore

    _HAS_WEBRTC = True
except Exception:  # pragma: no cover
    webrtcvad = None
    _HAS_WEBRTC = False


def _frames(pcm: np.ndarray) -> np.ndarray:
    n = len(pcm) // SAMPLES_PER_FRAME
    return pcm[: n * SAMPLES_PER_FRAME].reshape(n, SAMPLES_PER_FRAME).astype(np.float32)


def webrtc_vad(pcm: np.ndarray, aggressiveness: int = 2) -> Optional[np.ndarray]:
    if not _HAS_WEBRTC:
        return None
    vad = webrtcvad.Vad(aggressiveness)
    raw = pcm.astype("<i2", copy=False).tobytes()
    step = SAMPLES_PER_FRAME * 2
    n = len(raw) // step
    out = np.zeros(n, dtype=np.float32)
    is_speech = vad.is_speech
    for i in range(n):
        try:
            if is_speech(raw[i * step:(i + 1) * step], AUDIO_RATE):
                out[i] = 1.0
        except Exception:  # malformed frame: treat as silence
            pass
    return out


def energy_vad(frames: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-9)
    db = 20.0 * np.log10(rms + 1e-9)
    if len(db) == 0 or float(np.max(rms)) < 120.0:  # near-digital silence
        return np.zeros(len(db), dtype=np.float32)
    floor = float(np.percentile(db, 15))
    peak = float(np.percentile(db, 98))
    if peak - floor < 6.0:  # no dynamics: constant noise or constant tone
        flat = float(np.median(rms)) > 1500.0
        return np.full(len(db), 1.0 if flat else 0.0, dtype=np.float32)
    threshold = floor + 0.35 * (peak - floor)
    return (db >= max(threshold, 40.0)).astype(np.float32)


def spectral_vad(frames: np.ndarray) -> np.ndarray:
    if len(frames) == 0:
        return np.zeros(0, dtype=np.float32)
    window = np.hanning(SAMPLES_PER_FRAME).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    freqs = np.fft.rfftfreq(SAMPLES_PER_FRAME, d=1.0 / AUDIO_RATE)
    band = (freqs >= 300.0) & (freqs <= 3400.0)
    ratio = spec[:, band].sum(axis=1) / (spec.sum(axis=1) + 1e-6)
    return (ratio >= 0.6).astype(np.float32)


def _runs(binary: np.ndarray, value: int) -> tuple[np.ndarray, np.ndarray]:
    padded = np.concatenate(([1 - value], binary.astype(np.int8), [1 - value]))
    d = np.diff((padded == value).astype(np.int8))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def smooth(timeline: np.ndarray, max_gap: int = 20, min_run: int = 10) -> np.ndarray:
    """Fill silences <= ``max_gap`` frames between speech, drop speech runs < ``min_run``."""
    b = (timeline > 0).astype(np.int8)
    if not b.any():
        return b.astype(np.float32)
    starts, ends = _runs(b, 0)
    for s, e in zip(starts, ends):
        if s > 0 and e < len(b) and e - s <= max_gap:
            b[s:e] = 1
    starts, ends = _runs(b, 1)
    for s, e in zip(starts, ends):
        if e - s < min_run:
            b[s:e] = 0
    return b.astype(np.float32)


def detect_speech(pcm: np.ndarray) -> np.ndarray:
    """Return a float32 0/1 speech timeline at ``FRAME_RATE`` for int16 mono 16 kHz PCM."""
    frames = _frames(pcm)
    if len(frames) == 0:
        return np.zeros(0, dtype=np.float32)
    energy = energy_vad(frames)
    gate = energy  # spectral evidence only counts when there is signal
    spectral = spectral_vad(frames) * (np.sqrt(np.mean(frames * frames, axis=1)) > 120.0)
    web = webrtc_vad(pcm)
    if web is not None and len(web) >= len(frames):
        web = web[: len(frames)]
        score = 0.5 * web + 0.25 * gate + 0.25 * spectral
    else:
        score = 0.5 * gate + 0.5 * spectral
    fused = (score >= 0.5).astype(np.float32)
    result = smooth(fused)
    logger.debug("VAD: %.1fs speech of %.1fs audio", result.sum() / FRAME_RATE, len(result) / FRAME_RATE)
    return result


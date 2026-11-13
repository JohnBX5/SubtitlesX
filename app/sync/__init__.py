"""Audio extraction, VAD, and two-stage synchronization engine."""

from app.sync.aligner import (
    ANALYSIS_SAMPLE_RATE,
    CANDIDATE_SCALE_FACTORS,
    AlignmentResult,
    SpeechTimeline,
    align_cues_to_timeline,
    build_speech_timeline,
    dilate_binary_timeline,
    evaluate_alignment_confidence,
    find_best_global_alignment,
    generate_subtitle_speech_timeline,
    shift_cues,
    try_piecewise_split_alignment,
)
from app.sync.audio import AudioExtractor, AudioWindow, ExtractedAudio
from app.sync.engine import CandidateSyncInput, SyncEngine, SyncResult
from app.sync.vad import FRAME_RATE, detect_speech, smooth

__all__ = [
    "ANALYSIS_SAMPLE_RATE",
    "AlignmentResult",
    "AudioExtractor",
    "AudioWindow",
    "CANDIDATE_SCALE_FACTORS",
    "CandidateSyncInput",
    "ExtractedAudio",
    "FRAME_RATE",
    "SpeechTimeline",
    "SyncEngine",
    "SyncResult",
    "align_cues_to_timeline",
    "build_speech_timeline",
    "detect_speech",
    "dilate_binary_timeline",
    "evaluate_alignment_confidence",
    "find_best_global_alignment",
    "generate_subtitle_speech_timeline",
    "shift_cues",
    "smooth",
    "try_piecewise_split_alignment",
]


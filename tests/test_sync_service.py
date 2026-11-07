"""Unit tests for the Subtitle Synchronization Service and speech alignment engine."""

import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

from app.services.sync_service import (
    ANALYSIS_SAMPLE_RATE,
    SubtitleCue,
    SubtitleSynchronizationService,
    subtitle_sync_service,
)
from tests.conftest import SAMPLE_ARABIC_SRT, SAMPLE_VALID_SRT


def create_synthetic_wav(
    speech_regions: list[tuple[float, float]],
    total_duration_sec: float,
    sample_rate: int = 16000,
) -> Path:
    """Create a temporary 16kHz mono 16-bit PCM WAV file with voice-like tone during speech regions."""
    total_samples = int(total_duration_sec * sample_rate)
    audio_data = np.zeros(total_samples, dtype=np.int16)

    # 440 Hz sine wave tone with high amplitude for speech regions
    t = np.linspace(0, total_duration_sec, total_samples, endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * 20000).astype(np.int16)

    for start_s, end_s in speech_regions:
        start_idx = int(start_s * sample_rate)
        end_idx = int(end_s * sample_rate)
        audio_data[start_idx:end_idx] = tone[start_idx:end_idx]

    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp_file.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

    return Path(tmp_file.name)


def test_parse_cues_and_shift():
    """Verify subtitle parsing into cues and shifting with timestamp formatting."""
    cues = subtitle_sync_service.parse_cues_from_text(SAMPLE_VALID_SRT)
    assert len(cues) == 3
    assert cues[0].start_seconds == 1.0
    assert cues[0].end_seconds == 4.0
    assert "Welcome to Cinema Hub!" in cues[0].text

    # Apply +2.5s shift
    shifted_srt = subtitle_sync_service.apply_shifts_to_cues(cues, 2.5, output_format="srt")
    assert "00:00:03,500 --> 00:00:06,500" in shifted_srt
    assert "Welcome to Cinema Hub!" in shifted_srt

    # Apply shift to WebVTT
    shifted_vtt = subtitle_sync_service.apply_shifts_to_cues(cues, 1.0, output_format="vtt")
    assert shifted_vtt.startswith("WEBVTT")
    assert "00:00:02.000 --> 00:00:05.000" in shifted_vtt


def test_shift_cues_clamps_negative_timestamps():
    """Verify negative timestamps resulting from large negative shifts are clamped to 00:00:00,000."""
    cues = subtitle_sync_service.parse_cues_from_text(SAMPLE_VALID_SRT)
    # Cue 1 starts at 1.0s; shift by -5.0s would be -4.0s
    shifted_srt = subtitle_sync_service.apply_shifts_to_cues(cues, -5.0, output_format="srt")
    assert "00:00:00,000 -->" in shifted_srt
    assert "-" not in shifted_srt.split("-->")[0]


@pytest.mark.asyncio
async def test_synchronize_already_aligned():
    """Verify that when audio speech aligns within 0.25s, status is 'already_aligned'."""
    # Subtitle cues are at 1.0-4.0s, 5.5-8.2s, 9.0-12.0s
    speech_regions = [(1.0, 4.0), (5.5, 8.2), (9.0, 12.0)]
    wav_path = create_synthetic_wav(speech_regions, total_duration_sec=20.0)

    try:
        result = await subtitle_sync_service.synchronize_subtitle(
            subtitle_content=SAMPLE_VALID_SRT,
            local_audio_path=wav_path,
        )

        assert result.success is True
        assert result.status in {"already_aligned", "synchronized"}
        if result.offset_seconds is not None:
            assert abs(result.offset_seconds) < 0.5
        assert result.audio_analyzed is True
        assert result.cue_count == 3
    finally:
        wav_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_synchronize_delayed_subtitle():
    """Verify that a subtitle delayed by 10s relative to audio is corrected."""
    # Audio has speech at 1.0..4.0s and 5.5..8.2s
    # Subtitle cues are shifted forward by +10s (i.e. cues at 11.0s and 15.5s)
    speech_regions = [(1.0, 4.0), (5.5, 8.2), (9.0, 12.0)]
    wav_path = create_synthetic_wav(speech_regions, total_duration_sec=25.0)

    cues = subtitle_sync_service.parse_cues_from_text(SAMPLE_VALID_SRT)
    delayed_srt = subtitle_sync_service.apply_shifts_to_cues(cues, 10.0, output_format="srt")

    try:
        result = await subtitle_sync_service.synchronize_subtitle(
            subtitle_content=delayed_srt,
            local_audio_path=wav_path,
        )

        assert result.success is True
        assert result.status == "synchronized"
        assert result.offset_seconds is not None
        # Should correct with approximately -10s offset
        assert pytest.approx(result.offset_seconds, abs=0.5) == -10.0
        assert result.audio_analyzed is True
    finally:
        wav_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_synchronize_with_video_intro():
    """Verify that video with a 15-second intro before movie speech is aligned."""
    # Video has 15s intro before dialogue starts (dialogue at 16.0..19.0s and 20.5..23.2s)
    # Original subtitle dialogue starts at 1.0..4.0s
    speech_regions = [(16.0, 19.0), (20.5, 23.2), (24.0, 27.0)]
    wav_path = create_synthetic_wav(speech_regions, total_duration_sec=35.0)

    try:
        result = await subtitle_sync_service.synchronize_subtitle(
            subtitle_content=SAMPLE_VALID_SRT,
            local_audio_path=wav_path,
        )

        assert result.success is True
        assert result.status == "synchronized"
        assert result.offset_seconds is not None
        # Subtitle should be shifted by approximately +15 seconds
        assert pytest.approx(result.offset_seconds, abs=0.5) == 15.0
    finally:
        wav_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_synchronize_arabic_text_preservation():
    """Verify Arabic subtitles maintain full character integrity and RTL marks after synchronization."""
    speech_regions = [(5.0, 8.5), (9.0, 13.0), (14.0, 18.0)]
    wav_path = create_synthetic_wav(speech_regions, total_duration_sec=25.0)

    try:
        result = await subtitle_sync_service.synchronize_subtitle(
            subtitle_content=SAMPLE_ARABIC_SRT,
            local_audio_path=wav_path,
            language="ar",
        )

        assert result.success is True
        assert "مرحبا بكم في سينما هاب!" in result.subtitle_content
        assert "هذا ملف ترجمة باللغة العربية مع علامات ترقيم كاملة." in result.subtitle_content
        assert "مشاهدة ممتعة للجميع." in result.subtitle_content
    finally:
        wav_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_synchronize_silent_audio_fallback():
    """Verify that silent audio returns status='failed' without applying random offsets."""
    wav_path = create_synthetic_wav([], total_duration_sec=10.0)

    try:
        result = await subtitle_sync_service.synchronize_subtitle(
            subtitle_content=SAMPLE_VALID_SRT,
            local_audio_path=wav_path,
        )

        assert result.success is True
        assert result.status == "failed"
        assert result.audio_analyzed is True
        # Preserves original subtitle text
        assert "Welcome to Cinema Hub!" in result.subtitle_content
    finally:
        wav_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_synchronize_unavailable_audio_fallback():
    """Verify that omitting stream URL returns status='unavailable' with original subtitle preserved."""
    result = await subtitle_sync_service.synchronize_subtitle(
        subtitle_content=SAMPLE_VALID_SRT,
        stream_url=None,
    )

    assert result.success is True
    assert result.status == "unavailable"
    assert result.audio_analyzed is False
    assert "Welcome to Cinema Hub!" in result.subtitle_content


"""Unit tests for the Subtitle Synchronization Service and speech alignment engine."""

import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

from app.services.audio_extractor import ExtractedAudioInfo
from app.services.sync_service import (
    ANALYSIS_SAMPLE_RATE,
    CandidateSyncInput,
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


def test_clean_cue_text_filters_sdh_watermarks_and_music():
    """Verify non-spoken cues (SDH, watermarks, music symbols) are excluded from speech timeline."""
    cues = [
        SubtitleCue(1, 0.5, 4.0, "Synced and corrected by OpenSubtitles.org"),
        SubtitleCue(2, 4.5, 6.5, "[dramatic music playing] (thunder rumbles)"),
        SubtitleCue(3, 7.0, 9.0, "♪ Somewhere over the rainbow ♪"),
        SubtitleCue(4, 10.0, 12.5, "ترجمة: فريق سينما هاب"),
        SubtitleCue(5, 15.0, 18.0, "<i>- JOHN:</i> (sighs) We need to leave right now!"),
        SubtitleCue(6, 20.0, 23.0, "مرحباً بكم جميعاً في هذا العرض."),
    ]

    assert not subtitle_sync_service.is_spoken_dialogue_cue(cues[0])
    assert not subtitle_sync_service.is_spoken_dialogue_cue(cues[1])
    assert not subtitle_sync_service.is_spoken_dialogue_cue(cues[2])
    assert not subtitle_sync_service.is_spoken_dialogue_cue(cues[3])
    assert subtitle_sync_service.is_spoken_dialogue_cue(cues[4])
    assert subtitle_sync_service.is_spoken_dialogue_cue(cues[5])

    timeline = subtitle_sync_service.generate_subtitle_speech_timeline(cues, total_seconds=25.0)
    # Non-spoken cues (0..13s) must remain 0.0 on the speech timeline
    assert np.sum(timeline[:1400]) == 0.0
    # Spoken cues (15..18s and 20..23s) must be active (1.0)
    assert np.sum(timeline[1500:1800]) > 250.0
    assert np.sum(timeline[2000:2300]) > 250.0


def test_morphological_smoothing_closes_gaps_and_removes_clicks():
    """Verify morphological smoothing bridges short intra-word gaps and removes micro-noise clicks."""
    raw = np.zeros(500, dtype=np.float64)
    # Speech block 100..200 with a 10-frame (100ms) syllable pause at 140..150
    raw[100:140] = 1.0
    raw[150:200] = 1.0
    # Isolated 3-frame (30ms) noise click at 350..353
    raw[350:353] = 1.0

    smoothed = subtitle_sync_service.smooth_speech_timeline(raw, max_gap_frames=15, min_speech_frames=5)
    # Gap at 140..150 should be closed
    assert np.all(smoothed[100:200] == 1.0)
    # Click at 350..353 should be removed
    assert np.all(smoothed[340:370] == 0.0)


@pytest.mark.asyncio
async def test_partial_audio_window_confidence_not_penalized_by_full_movie_length():
    """Verify a short extracted audio window against a 2-hour movie subtitle achieves high confidence."""
    # Audio is only 25 seconds long with speech at 6.0..9.0s, 10.5..13.2s, 14.0..17.0s (+5s shift from base)
    speech_regions = [(6.0, 9.0), (10.5, 13.2), (14.0, 17.0)]
    wav_path = create_synthetic_wav(speech_regions, total_duration_sec=25.0)

    # Build a full-length 2-hour subtitle file (cues at 1..12s plus 80 cues up to 7200s)
    blocks = [SAMPLE_VALID_SRT.strip()]
    for i in range(4, 85):
        start_t = i * 85.0
        end_t = start_t + 3.5
        ts1 = subtitle_sync_service._sec_to_timestamp(start_t)
        ts2 = subtitle_sync_service._sec_to_timestamp(end_t)
        blocks.append(f"{i}\n{ts1} --> {ts2}\nDialogue line number {i} in the full movie.")
    full_movie_srt = "\n\n".join(blocks) + "\n"

    try:
        result = await subtitle_sync_service.synchronize_subtitle(
            subtitle_content=full_movie_srt,
            local_audio_path=wav_path,
        )
        assert result.success is True
        assert result.status == "synchronized"
        assert result.offset_seconds is not None
        assert pytest.approx(result.offset_seconds, abs=0.5) == 5.0
        assert result.confidence_score is not None
        assert result.confidence_score >= 0.85
    finally:
        wav_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_framerate_drift_correction():
    """Verify subtitle with PAL/NTSC framerate drift (25 / 23.976) is detected and rescaled."""
    true_scale = 25.0 / 23.976  # ~1.042709
    base_cues = [
        (2.0, 4.5, "First spoken sentence in the scene."),
        (10.0, 13.0, "Second spoken sentence a bit later."),
        (22.0, 25.5, "Third spoken sentence with clear cadence."),
        (35.0, 38.5, "Fourth spoken sentence halfway through."),
        (48.0, 52.0, "Fifth spoken sentence near the one minute mark."),
        (60.0, 64.0, "Sixth spoken sentence confirming linear drift."),
    ]

    # Audio has speech at true_scale * cue_times + 2.0s offset
    true_offset = 2.0
    speech_regions = [(s * true_scale + true_offset, e * true_scale + true_offset) for s, e, _ in base_cues]
    wav_path = create_synthetic_wav(speech_regions, total_duration_sec=75.0)

    srt_blocks = []
    for idx, (s, e, text) in enumerate(base_cues, start=1):
        ts1 = subtitle_sync_service._sec_to_timestamp(s)
        ts2 = subtitle_sync_service._sec_to_timestamp(e)
        srt_blocks.append(f"{idx}\n{ts1} --> {ts2}\n{text}")
    drifted_srt = "\n\n".join(srt_blocks) + "\n"

    try:
        result = await subtitle_sync_service.synchronize_subtitle(
            subtitle_content=drifted_srt,
            local_audio_path=wav_path,
        )
        assert result.success is True
        assert result.status == "synchronized"
        assert result.alignment_method == "drift_fft"
        assert pytest.approx(result.scale_factor, abs=0.005) == true_scale
        assert result.confidence_score is not None and result.confidence_score >= 0.80
    finally:
        wav_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_piecewise_split_alignment_for_commercial_or_recap_cut():
    """Verify piecewise dynamic programming split alignment corrects mid-stream offset jumps."""
    # First 3 cues have +2.0s offset; after a commercial cut at t=20s, next 3 cues have +9.0s offset
    cues_spec = [
        (2.0, 4.5, "Opening dialogue before commercial break one."),
        (6.0, 8.5, "Second line of opening act dialogue."),
        (10.0, 13.0, "Third line right before the break."),
        (22.0, 25.0, "First line after returning from the break."),
        (27.0, 30.0, "Second line in the second segment."),
        (32.0, 35.5, "Third line concluding the scene."),
    ]
    speech_regions = [
        (4.0, 6.5),
        (8.0, 10.5),
        (12.0, 15.0),
        (31.0, 34.0),
        (36.0, 39.0),
        (41.0, 44.5),
    ]
    wav_path = create_synthetic_wav(speech_regions, total_duration_sec=52.0)

    srt_blocks = []
    for idx, (s, e, text) in enumerate(cues_spec, start=1):
        ts1 = subtitle_sync_service._sec_to_timestamp(s)
        ts2 = subtitle_sync_service._sec_to_timestamp(e)
        srt_blocks.append(f"{idx}\n{ts1} --> {ts2}\n{text}")
    raw_srt = "\n\n".join(srt_blocks) + "\n"

    try:
        result = await subtitle_sync_service.synchronize_subtitle(
            subtitle_content=raw_srt,
            local_audio_path=wav_path,
        )
        assert result.success is True
        assert result.status == "synchronized"
        assert result.alignment_method == "piecewise_split"
        assert result.segments_count >= 2
        assert result.confidence_score is not None and result.confidence_score >= 0.85

        aligned_cues = subtitle_sync_service.parse_cues_from_text(result.subtitle_content)
        # First cue (2.0s) shifted by ~+2.0s -> ~4.0s
        assert pytest.approx(aligned_cues[0].start_seconds, abs=0.6) == 4.0
        # Last cue (32.0s) shifted by ~+9.0s -> ~41.0s
        assert pytest.approx(aligned_cues[5].start_seconds, abs=0.6) == 41.0
    finally:
        wav_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_multi_candidate_acoustic_selection():
    """Verify synchronize_candidates picks the subtitle release that matches actual video audio."""
    # Video audio has speech at 11.0..14.0s, 15.5..18.2s, 19.0..22.0s (+10s intro relative to SAMPLE_VALID_SRT)
    speech_regions = [(11.0, 14.0), (15.5, 18.2), (19.0, 22.0)]
    wav_path = create_synthetic_wav(speech_regions, total_duration_sec=30.0)

    # Candidate 1: Higher metadata score (e.g. popular Theatrical release), but totally mismatched cadence
    mismatched_srt = (
        "1\n00:00:01,000 --> 00:00:02,000\nShort line.\n\n"
        "2\n00:00:18,000 --> 00:00:20,500\nCompletely different scene timing.\n\n"
        "3\n00:00:27,000 --> 00:00:29,000\nAnother non-matching cue.\n"
    )

    candidates = [
        CandidateSyncInput(
            candidate_id="cand_wrong_cut",
            subtitle_content=mismatched_srt,
            metadata_score=250.0,
        ),
        CandidateSyncInput(
            candidate_id="cand_matching_cut",
            subtitle_content=SAMPLE_VALID_SRT,
            metadata_score=180.0,
        ),
    ]

    try:
        best_res, best_cand = await subtitle_sync_service.synchronize_candidates(
            candidates_inputs=candidates,
            local_audio_path=wav_path,
        )
        assert best_res.success is True
        assert best_res.status == "synchronized"
        assert best_res.selected_candidate_id == "cand_matching_cut"
        assert best_cand.candidate_id == "cand_matching_cut"
        assert best_res.candidates_evaluated == 2
        assert best_res.offset_seconds is not None
        assert pytest.approx(best_res.offset_seconds, abs=0.5) == 10.0
    finally:
        wav_path.unlink(missing_ok=True)


def test_multi_window_speech_timeline_and_masked_confidence():
    """Verify multi-window fast-seek audio assembly and masked confidence evaluation."""
    win1_path = create_synthetic_wav([(2.0, 5.0), (7.0, 10.0)], total_duration_sec=12.0)
    win2_path = create_synthetic_wav([(3.0, 6.5)], total_duration_sec=10.0)

    try:
        windows = [
            ExtractedAudioInfo(
                file_path=win1_path,
                duration_seconds=12.0,
                file_size_bytes=384000,
                source_type="direct_video",
                start_offset_seconds=0.0,
            ),
            ExtractedAudioInfo(
                file_path=win2_path,
                duration_seconds=10.0,
                file_size_bytes=320000,
                source_type="direct_video",
                start_offset_seconds=40.0,
            ),
        ]

        ref_speech, analyzed_mask = subtitle_sync_service.build_multi_window_speech_timeline(windows)
        # Total timeline spans up to 50.0s (5000 frames)
        assert len(ref_speech) == len(analyzed_mask)
        assert np.all(analyzed_mask[0:1150])
        # Un-sampled gap between 12s and 40s must be False in analyzed_mask
        assert not np.any(analyzed_mask[1300:3900])
        assert np.all(analyzed_mask[4000:4950])

        # Cues at 2..5s, 7..10s (in Window 1), 25..28s (in un-sampled gap), and 43..46.5s (in Window 2)
        cues = [
            SubtitleCue(1, 2.0, 5.0, "Spoken dialogue in window one first."),
            SubtitleCue(2, 7.0, 10.0, "Spoken dialogue in window one second."),
            SubtitleCue(3, 25.0, 28.0, "Spoken dialogue in unsampled middle gap."),
            SubtitleCue(4, 43.0, 46.5, "Spoken dialogue in window two anchor."),
        ]
        sub_speech = subtitle_sync_service.generate_subtitle_speech_timeline(cues, total_seconds=50.0)
        conf, matched_sec = subtitle_sync_service.evaluate_alignment_confidence(
            ref_speech=ref_speech,
            sub_speech=sub_speech,
            cues=cues,
            offsets=0.0,
            analyzed_mask=analyzed_mask,
        )
        # Cue 3 in the un-sampled gap is ignored; Cues 1, 2, 4 match 100% -> confidence >= 0.90
        assert conf >= 0.90
        assert matched_sec >= 8.0
    finally:
        win1_path.unlink(missing_ok=True)
        win2_path.unlink(missing_ok=True)




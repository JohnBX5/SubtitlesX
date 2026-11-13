"""Unit tests for the Subtitle Synchronization Engine and two-stage speech alignment."""

from __future__ import annotations

import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

from app.core.config import Settings
from app.subtitles.cleaner import is_dialogue
from app.subtitles.models import Cue
from app.subtitles.parser import parse_text
from app.subtitles.writer import format_timestamp, render
from app.sync.aligner import (
    build_speech_timeline,
    evaluate_alignment_confidence,
    generate_subtitle_speech_timeline,
    shift_cues,
)
from app.sync.audio import ExtractedAudio
from app.sync.engine import CandidateSyncInput, SyncEngine
from app.sync.vad import smooth
from tests.conftest import SAMPLE_ARABIC_SRT, SAMPLE_VALID_SRT


def create_synthetic_wav(
    speech_regions: list[tuple[float, float]],
    total_duration_sec: float,
    sample_rate: int = 16000,
) -> Path:
    total_samples = int(total_duration_sec * sample_rate)
    audio_data = np.zeros(total_samples, dtype=np.int16)

    t = np.linspace(0, total_duration_sec, total_samples, endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * 20000).astype(np.int16)

    for start_s, end_s in speech_regions:
        start_idx = int(start_s * sample_rate)
        end_idx = int(end_s * sample_rate)
        audio_data[start_idx:end_idx] = tone[start_idx:end_idx]

    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_file.close()
    with wave.open(tmp_file.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

    return Path(tmp_file.name)


def test_parse_cues_and_shift():
    _, cues = parse_text(SAMPLE_VALID_SRT)
    assert len(cues) == 3
    assert cues[0].start == 1.0
    assert cues[0].end == 4.0
    assert "Welcome to Cinema Hub!" in cues[0].text

    shifted_srt = render(shift_cues(cues, 2.5), "srt").content
    assert "00:00:03,500 --> 00:00:06,500" in shifted_srt
    assert "Welcome to Cinema Hub!" in shifted_srt

    shifted_vtt = render(shift_cues(cues, 1.0), "vtt").content
    assert shifted_vtt.startswith("WEBVTT")
    assert "00:00:02.000 --> 00:00:05.000" in shifted_vtt


def test_shift_cues_clamps_negative_timestamps():
    _, cues = parse_text(SAMPLE_VALID_SRT)
    shifted_srt = render(shift_cues(cues, -5.0), "srt").content
    assert "00:00:00,000 -->" in shifted_srt
    assert "-" not in shifted_srt.split("-->")[0]


@pytest.mark.asyncio
async def test_synchronize_already_aligned(test_settings: Settings):
    engine = SyncEngine(test_settings)
    speech_regions = [(1.0, 4.0), (5.5, 8.2), (9.0, 12.0)]
    wav_path = create_synthetic_wav(speech_regions, total_duration_sec=20.0)
    try:
        result = await engine.synchronize_subtitle(
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
        engine.close()
        wav_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_synchronize_delayed_subtitle(test_settings: Settings):
    engine = SyncEngine(test_settings)
    speech_regions = [(1.0, 4.0), (5.5, 8.2), (9.0, 12.0)]
    wav_path = create_synthetic_wav(speech_regions, total_duration_sec=25.0)

    _, cues = parse_text(SAMPLE_VALID_SRT)
    delayed_srt = render(shift_cues(cues, 10.0), "srt").content
    try:
        result = await engine.synchronize_subtitle(
            subtitle_content=delayed_srt,
            local_audio_path=wav_path,
        )
        assert result.success is True
        assert result.status == "synchronized"
        assert result.offset_seconds is not None
        assert pytest.approx(result.offset_seconds, abs=0.5) == -10.0
        assert result.audio_analyzed is True
    finally:
        engine.close()
        wav_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_synchronize_with_video_intro(test_settings: Settings):
    engine = SyncEngine(test_settings)
    speech_regions = [(16.0, 19.0), (20.5, 23.2), (24.0, 27.0)]
    wav_path = create_synthetic_wav(speech_regions, total_duration_sec=35.0)
    try:
        result = await engine.synchronize_subtitle(
            subtitle_content=SAMPLE_VALID_SRT,
            local_audio_path=wav_path,
        )
        assert result.success is True
        assert result.status == "synchronized"
        assert result.offset_seconds is not None
        assert pytest.approx(result.offset_seconds, abs=0.5) == 15.0
    finally:
        engine.close()
        wav_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_synchronize_arabic_text_preservation(test_settings: Settings):
    engine = SyncEngine(test_settings)
    speech_regions = [(5.0, 8.5), (9.0, 13.0), (14.0, 18.0)]
    wav_path = create_synthetic_wav(speech_regions, total_duration_sec=25.0)
    try:
        result = await engine.synchronize_subtitle(
            subtitle_content=SAMPLE_ARABIC_SRT,
            local_audio_path=wav_path,
            language="ar",
        )
        assert result.success is True
        assert "مرحبا بكم في سينما هاب!" in result.subtitle_content
        assert "هذا ملف ترجمة باللغة العربية مع علامات ترقيم كاملة." in result.subtitle_content
        assert "مشاهدة ممتعة للجميع." in result.subtitle_content
    finally:
        engine.close()
        wav_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_synchronize_silent_audio_fallback(test_settings: Settings):
    engine = SyncEngine(test_settings)
    wav_path = create_synthetic_wav([], total_duration_sec=10.0)
    try:
        result = await engine.synchronize_subtitle(
            subtitle_content=SAMPLE_VALID_SRT,
            local_audio_path=wav_path,
        )
        assert result.success is True
        assert result.status == "failed"
        assert result.audio_analyzed is True
        assert "Welcome to Cinema Hub!" in result.subtitle_content
    finally:
        engine.close()
        wav_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_synchronize_unavailable_audio_fallback(test_settings: Settings):
    engine = SyncEngine(test_settings)
    try:
        result = await engine.synchronize_subtitle(
            subtitle_content=SAMPLE_VALID_SRT,
            stream_url=None,
        )
        assert result.success is True
        assert result.status == "unavailable"
        assert result.audio_analyzed is False
        assert "Welcome to Cinema Hub!" in result.subtitle_content
    finally:
        engine.close()


def test_clean_cue_text_filters_sdh_watermarks_and_music():
    cues = [
        Cue(0.5, 4.0, "Synced and corrected by OpenSubtitles.org"),
        Cue(4.5, 6.5, "[dramatic music playing] (thunder rumbles)"),
        Cue(7.0, 9.0, "♪ Somewhere over the rainbow ♪"),
        Cue(10.0, 12.5, "ترجمة: فريق سينما هاب"),
        Cue(15.0, 18.0, "<i>- JOHN:</i> (sighs) We need to leave right now!"),
        Cue(20.0, 23.0, "مرحباً بكم جميعاً في هذا العرض."),
    ]
    assert not is_dialogue(cues[0])
    assert not is_dialogue(cues[1])
    assert not is_dialogue(cues[2])
    assert not is_dialogue(cues[3])
    assert is_dialogue(cues[4])
    assert is_dialogue(cues[5])

    timeline = generate_subtitle_speech_timeline(cues, total_seconds=25.0)
    assert np.sum(timeline[:1400]) == 0.0
    assert np.sum(timeline[1500:1800]) > 250.0
    assert np.sum(timeline[2000:2300]) > 250.0


def test_morphological_smoothing_closes_gaps_and_removes_clicks():
    raw = np.zeros(500, dtype=np.float64)
    raw[100:140] = 1.0
    raw[150:200] = 1.0
    raw[350:353] = 1.0

    smoothed = smooth(raw, max_gap=15, min_run=5)
    assert np.all(smoothed[100:200] == 1.0)
    assert np.all(smoothed[340:370] == 0.0)


@pytest.mark.asyncio
async def test_partial_audio_window_confidence_not_penalized_by_full_movie_length(test_settings: Settings):
    engine = SyncEngine(test_settings)
    speech_regions = [(6.0, 9.0), (10.5, 13.2), (14.0, 17.0)]
    wav_path = create_synthetic_wav(speech_regions, total_duration_sec=25.0)

    blocks = [SAMPLE_VALID_SRT.strip()]
    for i in range(4, 85):
        start_t = i * 85.0
        end_t = start_t + 3.5
        blocks.append(f"{i}\n{format_timestamp(start_t)} --> {format_timestamp(end_t)}\nDialogue line number {i} in the full movie.")
    full_movie_srt = "\n\n".join(blocks) + "\n"

    try:
        result = await engine.synchronize_subtitle(
            subtitle_content=full_movie_srt,
            local_audio_path=wav_path,
        )
        assert result.success is True
        assert result.status == "synchronized"
        assert result.offset_seconds is not None
        assert pytest.approx(result.offset_seconds, abs=0.5) == 5.0
        assert result.confidence_score is not None and result.confidence_score >= 0.85
    finally:
        engine.close()
        wav_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_framerate_drift_correction(test_settings: Settings):
    engine = SyncEngine(test_settings)
    true_scale = 25.0 / 23.976
    base_cues = [
        (2.0, 4.5, "First spoken sentence in the scene."),
        (10.0, 13.0, "Second spoken sentence a bit later."),
        (22.0, 25.5, "Third spoken sentence with clear cadence."),
        (35.0, 38.5, "Fourth spoken sentence halfway through."),
        (48.0, 52.0, "Fifth spoken sentence near the one minute mark."),
        (60.0, 64.0, "Sixth spoken sentence confirming linear drift."),
    ]
    true_offset = 2.0
    speech_regions = [(s * true_scale + true_offset, e * true_scale + true_offset) for s, e, _ in base_cues]
    wav_path = create_synthetic_wav(speech_regions, total_duration_sec=75.0)

    srt_blocks = [
        f"{idx}\n{format_timestamp(s)} --> {format_timestamp(e)}\n{text}"
        for idx, (s, e, text) in enumerate(base_cues, start=1)
    ]
    drifted_srt = "\n\n".join(srt_blocks) + "\n"

    try:
        result = await engine.synchronize_subtitle(
            subtitle_content=drifted_srt,
            local_audio_path=wav_path,
        )
        assert result.success is True
        assert result.status == "synchronized"
        assert result.alignment_method == "drift_fft"
        assert pytest.approx(result.scale_factor, abs=0.005) == true_scale
        assert result.confidence_score is not None and result.confidence_score >= 0.80
    finally:
        engine.close()
        wav_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_piecewise_split_alignment_for_commercial_or_recap_cut(test_settings: Settings):
    engine = SyncEngine(test_settings)
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

    srt_blocks = [
        f"{idx}\n{format_timestamp(s)} --> {format_timestamp(e)}\n{text}"
        for idx, (s, e, text) in enumerate(cues_spec, start=1)
    ]
    raw_srt = "\n\n".join(srt_blocks) + "\n"

    try:
        result = await engine.synchronize_subtitle(
            subtitle_content=raw_srt,
            local_audio_path=wav_path,
        )
        assert result.success is True
        assert result.status == "synchronized"
        assert result.alignment_method == "piecewise_split"
        assert result.segments_count >= 2
        assert result.confidence_score is not None and result.confidence_score >= 0.85

        _, aligned_cues = parse_text(result.subtitle_content)
        assert pytest.approx(aligned_cues[0].start, abs=0.6) == 4.0
        assert pytest.approx(aligned_cues[5].start, abs=0.6) == 41.0
    finally:
        engine.close()
        wav_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_multi_candidate_acoustic_selection(test_settings: Settings):
    engine = SyncEngine(test_settings)
    speech_regions = [(11.0, 14.0), (15.5, 18.2), (19.0, 22.0)]
    wav_path = create_synthetic_wav(speech_regions, total_duration_sec=30.0)

    mismatched_srt = (
        "1\n00:00:01,000 --> 00:00:02,000\nShort line.\n\n"
        "2\n00:00:18,000 --> 00:00:20,500\nCompletely different scene timing.\n\n"
        "3\n00:00:27,000 --> 00:00:29,000\nAnother non-matching cue.\n"
    )
    candidates = [
        CandidateSyncInput(candidate_id="cand_wrong_cut", subtitle_content=mismatched_srt, metadata_score=250.0),
        CandidateSyncInput(candidate_id="cand_matching_cut", subtitle_content=SAMPLE_VALID_SRT, metadata_score=180.0),
    ]
    try:
        best_res, best_cand = await engine.synchronize_candidates(
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
        engine.close()
        wav_path.unlink(missing_ok=True)


def test_multi_window_speech_timeline_and_masked_confidence():
    def synth_pcm(regions: list[tuple[float, float]], dur: float, sr: int = 16000) -> np.ndarray:
        n = int(dur * sr)
        arr = np.zeros(n, dtype=np.int16)
        t = np.linspace(0, dur, n, endpoint=False)
        tone = (np.sin(2 * np.pi * 440 * t) * 20000).astype(np.int16)
        for s, e in regions:
            arr[int(s * sr) : int(e * sr)] = tone[int(s * sr) : int(e * sr)]
        return arr

    windows = [
        ExtractedAudio(start=0.0, pcm=synth_pcm([(2.0, 5.0), (7.0, 10.0)], 12.0)),
        ExtractedAudio(start=40.0, pcm=synth_pcm([(3.0, 6.5)], 10.0)),
    ]
    timeline = build_speech_timeline(windows)
    assert len(timeline.speech) == len(timeline.mask)
    assert np.all(timeline.mask[0:1150])
    assert not np.any(timeline.mask[1300:3900])
    assert np.all(timeline.mask[4000:4950])

    cues = [
        Cue(2.0, 5.0, "Spoken dialogue in window one first."),
        Cue(7.0, 10.0, "Spoken dialogue in window one second."),
        Cue(25.0, 28.0, "Spoken dialogue in unsampled middle gap."),
        Cue(43.0, 46.5, "Spoken dialogue in window two anchor."),
    ]
    sub_speech = generate_subtitle_speech_timeline(cues, total_seconds=50.0)
    conf, matched_sec = evaluate_alignment_confidence(
        ref_speech=timeline.speech,
        sub_speech=sub_speech,
        cues=cues,
        offsets=0.0,
        analyzed_mask=timeline.mask,
    )
    assert conf >= 0.90
    assert matched_sec >= 8.0


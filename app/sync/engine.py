"""Synchronization engine orchestrating audio extraction, VAD, caching, and alignment."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.cache.store import CacheStore, make_key
from app.core.concurrency import SingleFlight
from app.core.config import Settings, get_settings
from app.core.errors import AudioExtractionError, SubtitleValidationError
from app.core.security import stream_fingerprint
from app.subtitles.cleaner import normalize_cues
from app.subtitles.models import Cue
from app.subtitles.parser import parse_text
from app.subtitles.writer import render
from app.sync.aligner import (
    ANALYSIS_SAMPLE_RATE,
    AlignmentResult,
    SpeechTimeline,
    align_cues_to_timeline,
    build_speech_timeline,
)
from app.sync.audio import AudioExtractor
from app.sync.vad import AUDIO_RATE, detect_speech

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CandidateSyncInput:
    """Candidate subtitle to evaluate acoustically against video audio."""

    candidate_id: str
    subtitle_content: str
    metadata_score: float = 0.0
    metadata: Optional[Any] = None


@dataclass(slots=True)
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


class SyncEngine:
    """Orchestrates audio extraction, speech timeline caching, and two-stage alignment."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        cache: Optional[CacheStore] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cache = cache
        self._semaphore = asyncio.Semaphore(max(1, self.settings.MAX_CONCURRENT_SYNC_JOBS))
        self._speech_flight: SingleFlight[SpeechTimeline] = SingleFlight()
        self._executor = ThreadPoolExecutor(
            max_workers=self.settings.sync_worker_threads,
            thread_name_prefix="sync-worker",
        )
        self.extractor = AudioExtractor(self.settings, executor=self._executor)

    def close(self) -> None:
        self.extractor.kill_all()
        self._executor.shutdown(wait=False)

    @staticmethod
    def _parse_cues(subtitle_content: str) -> Tuple[str, List[Cue]]:
        fmt, cues = parse_text(subtitle_content)
        cues, _ = normalize_cues(cues, strip_ads=False)
        return fmt, cues

    @staticmethod
    def detect_speech_from_wav(wav_path: Path) -> np.ndarray:
        """Read 16 kHz mono 16-bit PCM WAV file and detect speech activity at 100 Hz."""
        raw_bytes = wav_path.read_bytes()
        data_offset = 44
        data_idx = raw_bytes.find(b"data")
        if 0 <= data_idx <= 200 and data_idx + 8 <= len(raw_bytes):
            data_offset = data_idx + 8

        pcm_bytes = raw_bytes[data_offset:] if len(raw_bytes) > data_offset else b""
        usable = len(pcm_bytes) - (len(pcm_bytes) % 2)
        if usable <= 0:
            return np.zeros(0, dtype=np.float64)

        pcm = np.frombuffer(pcm_bytes[:usable], dtype=np.int16)
        speech = detect_speech(pcm).astype(np.float64)

        if np.sum(speech > 0) < 10:
            try:
                from ffsubsync.speech_transformers import _make_auditok_detector

                auditok_detector = _make_auditok_detector(ANALYSIS_SAMPLE_RATE, AUDIO_RATE, 0.0)
                auditok_tl = np.asarray(auditok_detector(pcm_bytes[:usable]), dtype=np.float64)
                if len(auditok_tl) > 0 and np.sum(auditok_tl > 0) >= 10:
                    return (auditok_tl > 0).astype(np.float64)
            except Exception as exc:
                logger.debug("Auditok fallback skipped: %s", exc)

        return speech

    async def _get_speech_timeline(
        self,
        stream_url: Optional[str],
        stream_headers: Optional[Dict[str, str]],
        media_duration: Optional[float],
        local_audio_path: Optional[Path],
    ) -> SpeechTimeline:
        loop = asyncio.get_running_loop()

        if local_audio_path is not None and local_audio_path.exists():
            speech = await loop.run_in_executor(self._executor, self.detect_speech_from_wav, local_audio_path)
            return SpeechTimeline(speech=speech, mask=np.ones(len(speech), dtype=bool))

        assert stream_url is not None
        speech_key = make_key("speech", stream_fingerprint(stream_url), round(media_duration or 0.0, -1))

        if self.cache is not None:
            cached = await self.cache.get("speech", speech_key)
            if cached is not None:
                try:
                    return SpeechTimeline.from_bytes(cached.value)
                except Exception as exc:
                    logger.warning("Corrupt cached speech timeline ignored: %s", exc)
                    await self.cache.delete("speech", speech_key)

        async def _extract_and_analyze() -> SpeechTimeline:
            if self.cache is not None:
                cached_inner = await self.cache.get("speech", speech_key)
                if cached_inner is not None:
                    try:
                        return SpeechTimeline.from_bytes(cached_inner.value)
                    except Exception:
                        pass

            windows = self.extractor.plan_windows(media_duration)
            extracted = await self.extractor.extract(stream_url, headers=stream_headers, windows=windows)
            timeline = await loop.run_in_executor(self._executor, build_speech_timeline, extracted)

            if self.cache is not None and np.sum(timeline.speech > 0) >= 10:
                try:
                    await self.cache.set(
                        "speech",
                        speech_key,
                        timeline.to_bytes(),
                        ttl_seconds=float(self.settings.SPEECH_CACHE_TTL_HOURS) * 3600.0,
                    )
                except Exception as exc:
                    logger.warning("Failed to cache speech timeline: %s", exc)

            return timeline

        return await self._speech_flight.run(speech_key, _extract_and_analyze)

    def _render_result(
        self,
        cues: List[Cue],
        alignment: AlignmentResult,
        target_format: str,
        start_time: float,
    ) -> SyncResult:
        rendered = render(alignment.cues if alignment.cues else cues, target_format)
        elapsed = round(time.perf_counter() - start_time, 3)
        return SyncResult(
            success=True,
            status=alignment.status,
            offset_seconds=alignment.offset_seconds,
            confidence_score=alignment.confidence_score,
            audio_analyzed=True,
            processing_time_seconds=elapsed,
            subtitle_content=rendered.content,
            format=rendered.format,
            cue_count=rendered.cue_count,
            detail=(
                f"Alignment status: {alignment.status} "
                f"({alignment.alignment_method}, segments={alignment.segments_count})"
            ),
            scale_factor=alignment.scale_factor,
            alignment_method=alignment.alignment_method,
            segments_count=alignment.segments_count,
        )

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
        """Execute end-to-end synchronization for a single subtitle string."""
        del language  # preserved in signature
        start_time = time.perf_counter()
        fmt = (target_format or "srt").lower()

        try:
            source_fmt, cues = self._parse_cues(subtitle_content)
        except SubtitleValidationError:
            cues = []
            source_fmt = fmt

        if not cues:
            return SyncResult(
                success=False,
                status="failed",
                offset_seconds=None,
                confidence_score=0.0,
                audio_analyzed=False,
                processing_time_seconds=round(time.perf_counter() - start_time, 3),
                subtitle_content=subtitle_content,
                format=fmt,
                cue_count=0,
                detail="Subtitle contains 0 valid dialogue cues.",
            )

        fallback_text = subtitle_content if source_fmt == fmt else render(cues, fmt).content

        if not stream_url and not (local_audio_path and local_audio_path.exists()):
            return SyncResult(
                success=True,
                status="unavailable",
                offset_seconds=0.0,
                confidence_score=None,
                audio_analyzed=False,
                processing_time_seconds=round(time.perf_counter() - start_time, 3),
                subtitle_content=fallback_text,
                format=fmt,
                cue_count=len(cues),
                detail="Audio playback stream URL was not provided; returning original subtitle.",
            )

        async with self._semaphore:
            try:
                timeline = await self._get_speech_timeline(
                    stream_url=stream_url,
                    stream_headers=stream_headers,
                    media_duration=media_duration,
                    local_audio_path=local_audio_path,
                )
            except AudioExtractionError as exc:
                logger.warning("Audio extraction failed (%s). Falling back to original subtitle.", exc)
                is_unsupported = exc.unsupported or "unsupported" in str(exc).lower() or "drm" in str(exc).lower()
                return SyncResult(
                    success=True,
                    status="unsupported" if is_unsupported else "unavailable",
                    offset_seconds=None,
                    confidence_score=0.0,
                    audio_analyzed=False,
                    processing_time_seconds=round(time.perf_counter() - start_time, 3),
                    subtitle_content=fallback_text,
                    format=fmt,
                    cue_count=len(cues),
                    detail=f"Audio could not be accessed: {exc}",
                )

            if np.sum(timeline.speech > 0) < 10:
                logger.warning("Insufficient voice activity in extracted audio.")
                return SyncResult(
                    success=True,
                    status="failed",
                    offset_seconds=0.0,
                    confidence_score=0.0,
                    audio_analyzed=True,
                    processing_time_seconds=round(time.perf_counter() - start_time, 3),
                    subtitle_content=fallback_text,
                    format=fmt,
                    cue_count=len(cues),
                    detail="No detectable voice activity found in audio stream.",
                )

            loop = asyncio.get_running_loop()
            alignment = await loop.run_in_executor(
                self._executor,
                align_cues_to_timeline,
                cues,
                timeline,
                self.settings.SYNC_MAX_OFFSET_SECONDS,
            )

        result = self._render_result(cues, alignment, fmt, start_time)
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
        """Extract audio once and select the candidate with highest acoustic match."""
        if not candidates_inputs:
            raise ValueError("candidates_inputs must contain at least one CandidateSyncInput.")

        fmt = (target_format or "srt").lower()

        if len(candidates_inputs) == 1 or (not stream_url and not (local_audio_path and local_audio_path.exists())):
            first = candidates_inputs[0]
            res = await self.synchronize_subtitle(
                subtitle_content=first.subtitle_content,
                stream_url=stream_url,
                stream_headers=stream_headers,
                target_format=fmt,
                language=language,
                local_audio_path=local_audio_path,
                media_duration=media_duration,
            )
            res.selected_candidate_id = first.candidate_id
            res.candidates_evaluated = 1
            return res, first

        start_time = time.perf_counter()
        first = candidates_inputs[0]

        async with self._semaphore:
            try:
                timeline = await self._get_speech_timeline(
                    stream_url=stream_url,
                    stream_headers=stream_headers,
                    media_duration=media_duration,
                    local_audio_path=local_audio_path,
                )
            except AudioExtractionError as exc:
                logger.warning("Audio extraction failed in multi-candidate sync (%s).", exc)
                try:
                    src_fmt, first_cues = self._parse_cues(first.subtitle_content)
                except SubtitleValidationError:
                    src_fmt, first_cues = fmt, []
                fallback_text = (
                    first.subtitle_content
                    if src_fmt == fmt or not first_cues
                    else render(first_cues, fmt).content
                )
                is_unsupported = exc.unsupported or "unsupported" in str(exc).lower() or "drm" in str(exc).lower()
                return (
                    SyncResult(
                        success=True,
                        status="unsupported" if is_unsupported else "unavailable",
                        offset_seconds=None,
                        confidence_score=0.0,
                        audio_analyzed=False,
                        processing_time_seconds=round(time.perf_counter() - start_time, 3),
                        subtitle_content=fallback_text,
                        format=fmt,
                        cue_count=len(first_cues),
                        detail=f"Audio could not be accessed: {exc}",
                        selected_candidate_id=first.candidate_id,
                        candidates_evaluated=1,
                    ),
                    first,
                )

            if np.sum(timeline.speech > 0) < 10:
                try:
                    src_fmt, first_cues = self._parse_cues(first.subtitle_content)
                except SubtitleValidationError:
                    src_fmt, first_cues = fmt, []
                fallback_text = (
                    first.subtitle_content
                    if src_fmt == fmt or not first_cues
                    else render(first_cues, fmt).content
                )
                return (
                    SyncResult(
                        success=True,
                        status="failed",
                        offset_seconds=0.0,
                        confidence_score=0.0,
                        audio_analyzed=True,
                        processing_time_seconds=round(time.perf_counter() - start_time, 3),
                        subtitle_content=fallback_text,
                        format=fmt,
                        cue_count=len(first_cues),
                        detail="No detectable voice activity found in audio stream.",
                        selected_candidate_id=first.candidate_id,
                        candidates_evaluated=1,
                    ),
                    first,
                )

            loop = asyncio.get_running_loop()
            best_result: Optional[SyncResult] = None
            best_candidate: CandidateSyncInput = first
            best_composite_rank = -1e9
            evaluated_count = 0

            for cand_in in candidates_inputs:
                try:
                    _, cues = self._parse_cues(cand_in.subtitle_content)
                except SubtitleValidationError:
                    cues = []
                if not cues:
                    continue

                evaluated_count += 1
                alignment = await loop.run_in_executor(
                    self._executor,
                    align_cues_to_timeline,
                    cues,
                    timeline,
                    self.settings.SYNC_MAX_OFFSET_SECONDS,
                )
                res = self._render_result(cues, alignment, fmt, start_time)
                conf = res.confidence_score or 0.0
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
            best_result = SyncResult(
                success=False,
                status="failed",
                offset_seconds=None,
                confidence_score=0.0,
                audio_analyzed=True,
                processing_time_seconds=round(time.perf_counter() - start_time, 3),
                subtitle_content=first.subtitle_content,
                format=fmt,
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


"""Bounded audio extraction from remote media with FFmpeg.

FFmpeg decodes straight to 16 kHz mono signed 16-bit PCM on stdout, so no
scratch files are written. It runs in a worker thread (``subprocess.Popen``),
which works under every asyncio event loop policy, including the Windows
selector loop that cannot spawn asyncio subprocesses.

If the stream is slow and the timeout fires, the audio decoded so far is kept
(when long enough), so slow CDNs degrade accuracy rather than fail outright.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import threading
import time
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from app.core.config import Settings
from app.core.errors import AudioExtractionError
from app.core.security import mask_headers, redact_url, sanitize_playback_headers, validate_stream_url

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2
MIN_USABLE_SECONDS = 20.0

_UNSUPPORTED_MARKERS = (
    "drm",
    "encrypt",
    "widevine",
    "playready",
    "fairplay",
    "does not contain any stream",
    "matches no streams",
    "invalid data found when processing input",
    "protocol not found",
    "not on whitelist",
    "unsupported",
    "could not find codec parameters",
)


@dataclass(frozen=True, slots=True)
class AudioWindow:
    start: float
    duration: float


@dataclass(slots=True)
class ExtractedAudio:
    start: float
    pcm: np.ndarray  # int16 mono @ 16 kHz
    partial: bool = False

    @property
    def duration(self) -> float:
        return len(self.pcm) / SAMPLE_RATE


def detect_source_type(url: str) -> str:
    path = url.lower().split("?", 1)[0]
    if path.endswith(".m3u8"):
        return "hls"
    if path.endswith(".mpd"):
        return "dash"
    if path.endswith((".mp3", ".aac", ".m4a", ".wav", ".ogg", ".opus", ".flac")):
        return "direct_audio"
    if path.endswith((".mp4", ".m4v", ".mov", ".mkv", ".webm", ".ts", ".avi")):
        return "direct_video"
    return "direct"


def build_headers_arg(headers: Dict[str, str]) -> Optional[str]:
    """FFmpeg ``-headers`` value. Input must already be sanitized (no CR/LF)."""
    lines = [f"{k.title()}: {v}" for k, v in headers.items() if k.lower() != "user-agent"]
    return "\r\n".join(lines) + "\r\n" if lines else None


def classify_ffmpeg_error(stderr: str) -> bool:
    """True when the failure is permanent (DRM / no audio / unknown container)."""
    low = stderr.lower()
    return any(marker in low for marker in _UNSUPPORTED_MARKERS)


class AudioExtractor:
    def __init__(self, settings: Settings, executor: Optional[Executor] = None) -> None:
        self.settings = settings
        self.executor = executor
        self._procs: set[subprocess.Popen] = set()
        self._procs_lock = threading.Lock()

    # ------------------------------------------------------------------ planning
    def plan_windows(self, media_duration: Optional[float]) -> List[AudioWindow]:
        budget = float(max(60, self.settings.MAX_AUDIO_EXTRACT_SECONDS))
        if not media_duration or media_duration <= 0:
            return [AudioWindow(0.0, budget)]
        if media_duration <= budget + 120 or media_duration < 1200:
            return [AudioWindow(0.0, min(budget, media_duration))]
        # Long media: a head window (intros, recaps, opening ads) plus two
        # mid-stream anchors to detect framerate drift and mid-stream cuts.
        head = min(480.0, budget * 0.6)
        rest = budget - head
        windows = [AudioWindow(0.0, head)]
        if rest >= 120:
            each = min(180.0, rest / 2)
            for frac in (0.4, 0.7):
                start = round(media_duration * frac, 1)
                if start > head + 60 and start + each < media_duration:
                    windows.append(AudioWindow(start, each))
        return windows

    # ------------------------------------------------------------------ command
    def build_command(self, url: str, headers: Dict[str, str], window: AudioWindow) -> List[str]:
        ffmpeg = self.settings.resolved_ffmpeg_path
        if not ffmpeg:
            raise AudioExtractionError("FFmpeg binary is not available on the server.")
        cmd: List[str] = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error"]
        # Restrict FFmpeg to network protocols: blocks file://, concat:, etc. in playlists.
        cmd += ["-protocol_whitelist", "http,https,tcp,tls,crypto"]
        ua = headers.get("user-agent")
        if ua:
            cmd += ["-user_agent", ua]
        headers_arg = build_headers_arg(headers)
        if headers_arg:
            cmd += ["-headers", headers_arg]
        cmd += ["-rw_timeout", "15000000"]
        if detect_source_type(url) in {"direct", "direct_video", "direct_audio"}:
            cmd += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "4"]
        if window.start > 0:
            cmd += ["-ss", f"{window.start:.2f}"]
        cmd += [
            "-i", url,
            "-t", f"{window.duration:.2f}",
            "-map", "0:a:0",
            "-vn", "-sn", "-dn",
            "-af", "highpass=f=80,lowpass=f=4000",
            "-ac", "1",
            "-ar", str(SAMPLE_RATE),
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "pipe:1",
        ]
        return cmd

    # ------------------------------------------------------------------ execution
    def _run_blocking(self, cmd: Sequence[str], max_bytes: int, timeout: float) -> tuple[bytes, str, Optional[int], bool]:
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            proc = subprocess.Popen(
                list(cmd), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs
            )
        except OSError as exc:
            raise AudioExtractionError(f"Could not start FFmpeg: {exc}") from exc
        with self._procs_lock:
            self._procs.add(proc)

        stderr_chunks: List[bytes] = []

        def _drain_stderr() -> None:
            total = 0
            assert proc.stderr is not None
            for line in iter(proc.stderr.readline, b""):
                if total < 65536:
                    stderr_chunks.append(line)
                    total += len(line)

        timed_out = threading.Event()

        def _kill() -> None:
            timed_out.set()
            try:
                proc.kill()
            except OSError:
                pass

        err_thread = threading.Thread(target=_drain_stderr, daemon=True)
        err_thread.start()
        timer = threading.Timer(timeout, _kill)
        timer.daemon = True
        timer.start()
        buf = bytearray()
        try:
            assert proc.stdout is not None
            while True:
                chunk = proc.stdout.read(65536)
                if not chunk:
                    break
                buf.extend(chunk)
                if len(buf) >= max_bytes:
                    proc.kill()
                    break
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        finally:
            timer.cancel()
            err_thread.join(timeout=2)
            with self._procs_lock:
                self._procs.discard(proc)
        stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace").strip()
        return bytes(buf), stderr, proc.returncode, timed_out.is_set()

    async def _extract_window(self, url: str, headers: Dict[str, str], window: AudioWindow) -> ExtractedAudio:
        cmd = self.build_command(url, headers, window)
        max_bytes = min(self.settings.max_audio_extract_bytes, int(window.duration * BYTES_PER_SECOND) + 4096)
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        data, stderr, code, timed_out = await loop.run_in_executor(
            self.executor, self._run_blocking, cmd, max_bytes, float(self.settings.FFMPEG_TIMEOUT_SECONDS)
        )
        usable = len(data) - (len(data) % 2)
        pcm = np.frombuffer(data[:usable], dtype=np.int16).copy()
        seconds = len(pcm) / SAMPLE_RATE
        min_needed = min(MIN_USABLE_SECONDS, window.duration * 0.5)
        if seconds < min_needed:
            if timed_out:
                raise AudioExtractionError(
                    f"Audio extraction timed out after {self.settings.FFMPEG_TIMEOUT_SECONDS}s "
                    f"with only {seconds:.1f}s decoded."
                )
            msg = stderr.splitlines()[-1][:240] if stderr else f"exit code {code}"
            raise AudioExtractionError(f"FFmpeg could not decode audio: {msg}", unsupported=classify_ffmpeg_error(stderr))
        logger.info(
            "Extracted %.1fs audio @%.0fs from %s in %.2fs%s",
            seconds, window.start, detect_source_type(url), time.perf_counter() - t0,
            " (partial: timeout)" if timed_out else "",
        )
        return ExtractedAudio(window.start, pcm, partial=timed_out)

    async def extract(
        self,
        stream_url: str,
        headers: Optional[Dict[str, str]] = None,
        windows: Optional[List[AudioWindow]] = None,
    ) -> List[ExtractedAudio]:
        url = await validate_stream_url(stream_url, allow_private=self.settings.ALLOW_PRIVATE_STREAM_HOSTS)
        clean_headers = sanitize_playback_headers(headers)
        plan = windows or self.plan_windows(None)
        logger.info("Audio extraction from %s (%d window(s), headers=%s)",
                    redact_url(url), len(plan), mask_headers(clean_headers))
        results = await asyncio.gather(
            *(self._extract_window(url, clean_headers, w) for w in plan), return_exceptions=True
        )
        primary = results[0]
        if isinstance(primary, BaseException):
            if isinstance(primary, AudioExtractionError):
                raise primary
            raise AudioExtractionError(f"Unexpected audio extraction failure: {primary}") from primary
        out = [primary]
        for res in results[1:]:
            if isinstance(res, ExtractedAudio):
                out.append(res)
            else:
                logger.info("Secondary audio window skipped: %s", res)
        return out

    def kill_all(self) -> None:
        with self._procs_lock:
            procs = list(self._procs)
        for proc in procs:
            try:
                proc.kill()
            except OSError:
                pass


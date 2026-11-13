"""Typed application settings.

All legacy (v1) environment variable names are still honoured so existing
deployments (Render dashboard, ``.env`` files) keep working unchanged.
"""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app import __version__


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # --- Application -------------------------------------------------------
    APP_NAME: str = "CinemaHubSubtitleBackend"
    APP_VERSION: str = __version__
    ENVIRONMENT: str = "production"
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "text"  # "text" | "json"

    # --- OpenSubtitles REST API v1 ----------------------------------------
    OPENSUBTITLES_API_KEY: str = ""
    OPENSUBTITLES_USERNAME: str = ""
    OPENSUBTITLES_PASSWORD: str = ""
    OPENSUBTITLES_BASE_URL: str = "https://api.opensubtitles.com/api/v1"
    OPENSUBTITLES_USER_AGENT: str = f"CinemaHubSubtitleBackend v{__version__}"
    HTTP_TIMEOUT_SECONDS: float = 20.0
    MAX_CONCURRENT_REQUESTS: int = 5
    OPENSUBTITLES_MAX_RETRIES: int = 3
    LANGUAGES_MAX_PAGES: int = 3

    # --- Cache / persistence ----------------------------------------------
    CACHE_DIR: str = "cache"
    DATABASE_URL: str = ""  # legacy: sqlite+aiosqlite:///path.db
    CACHE_DB_PATH: str = ""  # preferred: explicit SQLite file path
    CACHE_TTL_HOURS: int = 168  # synchronized/served subtitles
    SYNC_CACHE_TTL_HOURS: int = 168
    SEARCH_CACHE_TTL_HOURS: int = 6
    FILE_CACHE_TTL_HOURS: int = 720  # raw OpenSubtitles files are immutable
    SPEECH_CACHE_TTL_HOURS: int = 168
    MEMORY_CACHE_MAX_MB: int = 64
    CACHE_MAX_SIZE_MB: int = 1024
    CACHE_CLEANUP_INTERVAL_MINUTES: int = 60

    # --- Subtitle processing ----------------------------------------------
    MAX_SUBTITLE_FILE_SIZE_MB: int = 10
    MAX_SUBTITLE_CUE_COUNT: int = 50000
    STRIP_SUBTITLE_ADS: bool = True

    # --- Synchronization ---------------------------------------------------
    MAX_CONCURRENT_SYNC_JOBS: int = 2
    SYNC_QUEUE_TIMEOUT_SECONDS: float = 60.0
    SYNC_MAX_CANDIDATES: int = 3
    SYNC_MAX_OFFSET_SECONDS: float = 300.0
    MAX_AUDIO_EXTRACT_SECONDS: int = 900
    MAX_AUDIO_EXTRACT_SIZE_MB: int = 50
    FFMPEG_TIMEOUT_SECONDS: int = 45
    FFMPEG_PATH: str = ""
    SYNC_WORKER_THREADS: int = 0  # 0 => MAX_CONCURRENT_SYNC_JOBS * 2
    ALLOW_PRIVATE_STREAM_HOSTS: bool = False  # development / testing only

    # --- HTTP surface ------------------------------------------------------
    CORS_ORIGINS: str = "*"
    API_ACCESS_KEY: str = ""  # optional shared secret (X-API-Key header)
    RATE_LIMIT_SYNC_PER_MINUTE: int = 20  # per client IP, 0 disables
    GZIP_MIN_SIZE: int = 1024

    # --- Derived helpers ---------------------------------------------------
    @property
    def cors_origins_list(self) -> List[str]:
        if not self.CORS_ORIGINS or self.CORS_ORIGINS.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_SUBTITLE_FILE_SIZE_MB * 1024 * 1024

    @property
    def max_audio_extract_bytes(self) -> int:
        return self.MAX_AUDIO_EXTRACT_SIZE_MB * 1024 * 1024

    @property
    def resolved_cache_dir(self) -> Path:
        path = Path(self.CACHE_DIR).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def resolved_db_path(self) -> str:
        """SQLite path for the persistent cache (``:memory:`` supported)."""
        if self.CACHE_DB_PATH:
            raw = self.CACHE_DB_PATH
        elif self.DATABASE_URL.startswith("sqlite"):
            raw = self.DATABASE_URL.split(":///", 1)[-1] or ":memory:"
        else:
            raw = str(self.resolved_cache_dir / "cinemahub_cache.db")
        if raw == ":memory:":
            return raw
        path = Path(raw).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    @property
    def resolved_ffmpeg_path(self) -> Optional[str]:
        if self.FFMPEG_PATH and os.path.isfile(self.FFMPEG_PATH):
            return self.FFMPEG_PATH
        system = shutil.which("ffmpeg")
        if system:
            return system
        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None

    @property
    def sync_worker_threads(self) -> int:
        return self.SYNC_WORKER_THREADS or max(2, self.MAX_CONCURRENT_SYNC_JOBS * 2)


@lru_cache()
def get_settings() -> Settings:
    return Settings()


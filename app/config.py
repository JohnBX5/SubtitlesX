"""Application configuration using Pydantic Settings."""

import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # OpenSubtitles REST API v1 Credentials
    OPENSUBTITLES_API_KEY: str = ""
    OPENSUBTITLES_USERNAME: str = ""
    OPENSUBTITLES_PASSWORD: str = ""

    # OpenSubtitles Endpoints & Identification
    OPENSUBTITLES_BASE_URL: str = "https://api.opensubtitles.com/api/v1"
    OPENSUBTITLES_USER_AGENT: str = "CinemaHubSubtitleBackend v1.0.0"

    # Application Information
    APP_NAME: str = "CinemaHubSubtitleBackend"
    APP_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./subtitles.db"

    # Cache Settings
    CACHE_DIR: str = "cache"
    CACHE_TTL_HOURS: int = 168  # 7 days
    SYNC_CACHE_TTL_HOURS: int = 168

    # Concurrency, Network, and Size Limits
    HTTP_TIMEOUT_SECONDS: float = 20.0
    MAX_SUBTITLE_FILE_SIZE_MB: int = 10
    MAX_SUBTITLE_CUE_COUNT: int = 50000
    MAX_CONCURRENT_REQUESTS: int = 5

    # Synchronization & Audio Processing Limits
    MAX_CONCURRENT_SYNC_JOBS: int = 2
    MAX_AUDIO_EXTRACT_SECONDS: int = 900  # 15 minutes max audio window
    MAX_AUDIO_EXTRACT_SIZE_MB: int = 50  # 50 MB max audio extract
    FFMPEG_TIMEOUT_SECONDS: int = 45  # 45 seconds max extraction time
    FFMPEG_PATH: str = ""

    # Security & CORS
    CORS_ORIGINS: str = "*"

    @property
    def cors_origins_list(self) -> List[str]:
        """Return CORS origins as a list of strings."""
        if not self.CORS_ORIGINS or self.CORS_ORIGINS.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def max_file_size_bytes(self) -> int:
        """Maximum allowed subtitle file size in bytes."""
        return self.MAX_SUBTITLE_FILE_SIZE_MB * 1024 * 1024

    @property
    def max_audio_extract_bytes(self) -> int:
        """Maximum allowed audio extraction size in bytes."""
        return self.MAX_AUDIO_EXTRACT_SIZE_MB * 1024 * 1024

    @property
    def resolved_cache_dir(self) -> Path:
        """Absolute path to the cache directory."""
        path = Path(self.CACHE_DIR).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def temp_sync_dir(self) -> Path:
        """Absolute path to the temporary synchronization scratch directory."""
        path = self.resolved_cache_dir / "sync_temp"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def resolved_ffmpeg_path(self) -> str:
        """Path to the ffmpeg executable."""
        if self.FFMPEG_PATH and os.path.isfile(self.FFMPEG_PATH):
            return self.FFMPEG_PATH

        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg:
            return system_ffmpeg

        # Fallback to imageio-ffmpeg bundled binary
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return "ffmpeg"


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()

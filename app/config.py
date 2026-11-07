"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import List

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

    # Concurrency, Network, and Size Limits
    HTTP_TIMEOUT_SECONDS: float = 20.0
    MAX_SUBTITLE_FILE_SIZE_MB: int = 10
    MAX_SUBTITLE_CUE_COUNT: int = 50000
    MAX_CONCURRENT_REQUESTS: int = 5

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
    def resolved_cache_dir(self) -> Path:
        """Absolute path to the cache directory."""
        path = Path(self.CACHE_DIR).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()


"""SQLAlchemy model for subtitle cache records."""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class SubtitleCacheRecord(Base):
    """Database model for cached subtitle files and metadata."""

    __tablename__ = "subtitle_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cache_key: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)

    media_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    imdb_id: Mapped[Optional[str]] = mapped_column(String(32), index=True, nullable=True)
    query: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    season_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    episode_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    language: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    format: Mapped[str] = mapped_column(String(16), nullable=False)  # srt, vtt, etc.

    subtitle_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    file_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    original_filename: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    stored_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)

    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    cue_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    encoding: Mapped[str] = mapped_column(String(32), nullable=False, default="utf-8")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        index=True,
        nullable=False,
    )

    def is_expired(self) -> bool:
        """Check whether the cache record is expired."""
        now = datetime.now(timezone.utc)
        if self.expires_at.tzinfo is None:
            # Handle naive datetime stored in SQLite
            return self.expires_at.replace(tzinfo=timezone.utc) < now
        return self.expires_at < now

    def __repr__(self) -> str:
        return (
            f"<SubtitleCacheRecord(id={self.id}, key='{self.cache_key}', "
            f"lang='{self.language}', format='{self.format}', "
            f"file='{self.stored_filename}')>"
        )

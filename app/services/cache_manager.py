"""Persistent subtitle caching manager using SQLite (SQLAlchemy/aiosqlite) and local disk storage.

Features:
- Deterministic cache key generation.
- Async in-flight request deduplication to prevent download stampedes.
- TTL expiration enforcement (default 7 days).
- Safe disk file storage with path traversal protection.
- Safe cleanup of expired records and orphan files.
"""

import asyncio
import hashlib
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_session_factory
from app.models.cache import SubtitleCacheRecord
from app.services.subtitle_manager import ProcessedSubtitle, subtitle_manager

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages persistent subtitle file storage and SQLite metadata records."""

    def __init__(self) -> None:
        self._inflight_locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    async def get_key_lock(self, cache_key: str) -> asyncio.Lock:
        """Get or create an asyncio.Lock for a specific cache key to deduplicate requests."""
        async with self._global_lock:
            if cache_key not in self._inflight_locks:
                self._inflight_locks[cache_key] = asyncio.Lock()
            return self._inflight_locks[cache_key]

    def build_cache_key(
        self,
        imdb_id: Optional[str] = None,
        query: Optional[str] = None,
        year: Optional[int] = None,
        language: str = "en",
        format_: str = "srt",
        media_type: Optional[str] = None,
        season_number: Optional[int] = None,
        episode_number: Optional[int] = None,
        file_id: Optional[int] = None,
    ) -> str:
        """Generate a deterministic cache key from subtitle identification parameters."""
        norm_imdb = (imdb_id or "").strip().lower()
        norm_query = (query or "").strip().lower()
        norm_lang = (language or "en").strip().lower()
        norm_fmt = (format_ or "srt").strip().lower()
        norm_type = (media_type or "movie").strip().lower()

        # Build raw key descriptor
        parts = [
            f"type={norm_type}",
            f"imdb={norm_imdb}",
            f"q={norm_query}",
            f"y={year or ''}",
            f"s={season_number or ''}",
            f"e={episode_number or ''}",
            f"lang={norm_lang}",
            f"fmt={norm_fmt}",
        ]
        if file_id:
            parts.append(f"fid={file_id}")

        raw_str = "|".join(parts)
        key_hash = hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:24]

        # Readable prefix + hash
        prefix = norm_imdb or norm_query[:16].replace(" ", "_") or "sub"
        prefix = "".join(c for c in prefix if c.isalnum() or c in ("-", "_"))
        return f"{prefix}_{norm_lang}_{norm_fmt}_{key_hash}"

    def build_sync_cache_key(
        self,
        imdb_id: Optional[str] = None,
        query: Optional[str] = None,
        year: Optional[int] = None,
        language: str = "en",
        format_: str = "srt",
        media_type: Optional[str] = None,
        season_number: Optional[int] = None,
        episode_number: Optional[int] = None,
        stream_url: Optional[str] = None,
    ) -> str:
        """Generate a deterministic cache key distinguishing the video audio source version."""
        stream_hash = ""
        if stream_url:
            clean_url = stream_url.split("?")[0].strip().lower()
            stream_hash = hashlib.sha256(clean_url.encode("utf-8")).hexdigest()[:16]

        base_key = self.build_cache_key(
            imdb_id=imdb_id,
            query=query,
            year=year,
            language=language,
            format_=format_,
            media_type=media_type,
            season_number=season_number,
            episode_number=episode_number,
        )
        return f"sync_{stream_hash}_{base_key}" if stream_hash else f"sync_{base_key}"

    async def get_cached_subtitle(
        self,
        cache_key: str,
        session: AsyncSession,
    ) -> Optional[Tuple[SubtitleCacheRecord, bytes]]:
        """Look up a subtitle in the database and disk cache.

        Returns:
            Tuple of (SubtitleCacheRecord, file_bytes) if valid cache hit, else None.
        """
        stmt = select(SubtitleCacheRecord).where(SubtitleCacheRecord.cache_key == cache_key)
        result = await session.execute(stmt)
        record = result.scalars().first()

        if not record:
            return None

        # Check TTL expiration
        if record.is_expired():
            logger.info("Cache hit expired for key '%s'. Invalidating record.", cache_key)
            await self.delete_record(record, session)
            return None

        # Check if physical file exists on disk
        file_path = Path(record.file_path).resolve()
        settings = get_settings()
        cache_root = settings.resolved_cache_dir

        # Path traversal guard: ensure file_path is strictly within cache_root
        if not str(file_path).startswith(str(cache_root)):
            logger.error("Security alert: cache file_path '%s' outside cache directory.", file_path)
            await self.delete_record(record, session)
            return None

        if not file_path.exists() or file_path.stat().st_size == 0:
            logger.warning("Cache record exists for '%s' but file is missing on disk.", cache_key)
            await self.delete_record(record, session)
            return None

        try:
            content_bytes = file_path.read_bytes()
            logger.info("Cache hit for key '%s' (%d bytes).", cache_key, len(content_bytes))
            return record, content_bytes
        except OSError as exc:
            logger.error("Failed to read cached file '%s': %s", file_path, exc)
            return None

    async def store_subtitle(
        self,
        cache_key: str,
        processed: ProcessedSubtitle,
        language: str,
        session: AsyncSession,
        media_type: Optional[str] = None,
        imdb_id: Optional[str] = None,
        query: Optional[str] = None,
        year: Optional[int] = None,
        season_number: Optional[int] = None,
        episode_number: Optional[int] = None,
        subtitle_id: Optional[str] = None,
        file_id: Optional[int] = None,
        original_filename: Optional[str] = None,
    ) -> SubtitleCacheRecord:
        """Store processed subtitle bytes to disk and insert/update the SQLite cache record."""
        settings = get_settings()
        cache_dir = settings.resolved_cache_dir

        # Generate a safe, unique filename for disk storage
        safe_base = subtitle_manager.sanitize_filename(
            original_filename or f"{cache_key}.{processed.format}",
            fallback_ext=processed.format,
        )
        stored_filename = f"{cache_key}_{safe_base}"
        target_path = (cache_dir / stored_filename).resolve()

        # Write file atomically
        temp_path = target_path.with_suffix(".tmp")
        temp_path.write_bytes(processed.content_bytes)
        temp_path.replace(target_path)

        # Calculate expiration time
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=settings.CACHE_TTL_HOURS)

        # Check if record already exists (e.g. from race condition)
        stmt = select(SubtitleCacheRecord).where(SubtitleCacheRecord.cache_key == cache_key)
        res = await session.execute(stmt)
        record = res.scalars().first()

        if record is None:
            record = SubtitleCacheRecord(
                cache_key=cache_key,
                media_type=media_type,
                imdb_id=imdb_id,
                query=query,
                year=year,
                season_number=season_number,
                episode_number=episode_number,
                language=language,
                format=processed.format,
                subtitle_id=subtitle_id,
                file_id=file_id,
                original_filename=original_filename,
                stored_filename=stored_filename,
                file_path=str(target_path),
                file_size=len(processed.content_bytes),
                cue_count=processed.cue_count,
                encoding=processed.encoding,
                created_at=now,
                expires_at=expires_at,
            )
            session.add(record)
        else:
            record.stored_filename = stored_filename
            record.file_path = str(target_path)
            record.file_size = len(processed.content_bytes)
            record.cue_count = processed.cue_count
            record.encoding = processed.encoding
            record.expires_at = expires_at

        await session.commit()
        await session.refresh(record)

        logger.info(
            "Cached subtitle '%s' successfully (size=%d, cues=%d, expires=%s).",
            cache_key,
            record.file_size,
            record.cue_count,
            expires_at.isoformat(),
        )
        return record

    async def delete_record(self, record: SubtitleCacheRecord, session: AsyncSession) -> None:
        """Safely delete a database record and its corresponding file on disk."""
        try:
            target_path = Path(record.file_path).resolve()
            if target_path.exists() and target_path.is_file():
                target_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not delete cached file '%s': %s", record.file_path, exc)

        await session.delete(record)
        await session.commit()

    async def cleanup_expired(self) -> int:
        """Delete all expired cache records and their on-disk files.

        Returns:
            Number of deleted records.
        """
        factory = get_session_factory()
        count = 0
        now = datetime.now(timezone.utc)

        async with factory() as session:
            stmt = select(SubtitleCacheRecord).where(SubtitleCacheRecord.expires_at <= now)
            result = await session.execute(stmt)
            expired_records = result.scalars().all()

            for rec in expired_records:
                try:
                    p = Path(rec.file_path).resolve()
                    if p.exists() and p.is_file():
                        p.unlink(missing_ok=True)
                except OSError:
                    pass
                await session.delete(rec)
                count += 1

            if count > 0:
                await session.commit()
                logger.info("Cleaned up %d expired subtitle cache records.", count)

        return count


# Global singleton instance
cache_manager = CacheManager()


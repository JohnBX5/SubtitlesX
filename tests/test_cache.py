"""Unit tests for SQLite persistent caching and disk storage."""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.cache import SubtitleCacheRecord
from app.services.cache_manager import cache_manager
from app.services.subtitle_manager import subtitle_manager
from tests.conftest import SAMPLE_VALID_SRT


def test_cache_key_generation_deterministic():
    """Verify cache keys are deterministic, case-insensitive, and distinct for different parameters."""
    key1 = cache_manager.build_cache_key(
        imdb_id="tt0111161",
        language="en",
        format_="srt",
    )
    key2 = cache_manager.build_cache_key(
        imdb_id="TT0111161",
        language="EN",
        format_="SRT",
    )
    # Different language
    key_ar = cache_manager.build_cache_key(
        imdb_id="tt0111161",
        language="ar",
        format_="srt",
    )
    # Different episode
    key_ep = cache_manager.build_cache_key(
        imdb_id="tt0903747",
        language="en",
        format_="srt",
        media_type="episode",
        season_number=1,
        episode_number=1,
    )

    assert key1 == key2
    assert key1 != key_ar
    assert key1 != key_ep


@pytest.mark.asyncio
async def test_cache_store_and_hit(test_db_session: AsyncSession):
    """Verify storing a processed subtitle persists it to disk and SQLite, and allows retrieval."""
    raw_bytes = SAMPLE_VALID_SRT.encode("utf-8")
    processed = subtitle_manager.process_subtitle(raw_bytes, filename_hint="shawshank.srt")

    cache_key = cache_manager.build_cache_key(
        imdb_id="tt0111161",
        language="en",
        format_="srt",
    )

    # 1. Store
    record = await cache_manager.store_subtitle(
        cache_key=cache_key,
        processed=processed,
        language="en",
        session=test_db_session,
        media_type="movie",
        imdb_id="tt0111161",
        query="The Shawshank Redemption",
        year=1994,
        subtitle_id="12345",
        file_id=98765,
        original_filename="shawshank.en.srt",
    )

    assert record.id is not None
    assert record.cache_key == cache_key
    assert Path(record.file_path).exists()
    assert Path(record.file_path).stat().st_size > 0

    # 2. Retrieve (HIT)
    hit = await cache_manager.get_cached_subtitle(cache_key, test_db_session)
    assert hit is not None
    cached_record, cached_bytes = hit

    assert cached_record.cache_key == cache_key
    assert cached_bytes == processed.content_bytes


@pytest.mark.asyncio
async def test_cache_expiration(test_db_session: AsyncSession):
    """Verify that expired cache records are invalidated and return None."""
    raw_bytes = SAMPLE_VALID_SRT.encode("utf-8")
    processed = subtitle_manager.process_subtitle(raw_bytes, filename_hint="expired.srt")

    cache_key = "test_expired_key"
    record = await cache_manager.store_subtitle(
        cache_key=cache_key,
        processed=processed,
        language="en",
        session=test_db_session,
        imdb_id="tt9999999",
    )

    # Manually backdate expiration to 1 hour ago
    record.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
    await test_db_session.commit()

    # Query: should detect expiration, delete record, and return None
    hit = await cache_manager.get_cached_subtitle(cache_key, test_db_session)
    assert hit is None

    # Verify deleted from DB
    stmt = select(SubtitleCacheRecord).where(SubtitleCacheRecord.cache_key == cache_key)
    res = await test_db_session.execute(stmt)
    assert res.scalars().first() is None


@pytest.mark.asyncio
async def test_cleanup_expired_records(test_db_session: AsyncSession):
    """Verify cleanup_expired removes expired records and deletes files from disk."""
    raw_bytes = SAMPLE_VALID_SRT.encode("utf-8")
    processed = subtitle_manager.process_subtitle(raw_bytes, filename_hint="cleanup.srt")

    # Store one active and one expired record
    active_key = "active_key"
    expired_key = "expired_key"

    rec_active = await cache_manager.store_subtitle(
        cache_key=active_key,
        processed=processed,
        language="en",
        session=test_db_session,
    )
    rec_expired = await cache_manager.store_subtitle(
        cache_key=expired_key,
        processed=processed,
        language="en",
        session=test_db_session,
    )

    rec_expired.expires_at = datetime.now(timezone.utc) - timedelta(days=2)
    await test_db_session.commit()

    expired_file_path = Path(rec_expired.file_path)
    assert expired_file_path.exists()

    # Run cleanup
    cleaned_count = await cache_manager.cleanup_expired()
    assert cleaned_count >= 1

    # Verify expired file deleted
    assert not expired_file_path.exists()

    # Verify active file still exists
    assert Path(rec_active.file_path).exists()


@pytest.mark.asyncio
async def test_inflight_request_deduplication():
    """Verify that multiple callers requesting the same cache key receive the exact same lock."""
    key = "concurrent_key_123"
    lock1 = await cache_manager.get_key_lock(key)
    lock2 = await cache_manager.get_key_lock(key)

    assert lock1 is lock2


"""Unit tests for the two-tier LRU + SQLite CacheStore and SingleFlight."""

import asyncio
import time

import pytest

from app.cache.store import CacheStore, make_key
from app.core.concurrency import SingleFlight


def test_make_key_deterministic():
    k1 = make_key("download", {"imdb": 111161, "q": None}, "en", "srt")
    k2 = make_key("download", {"q": None, "imdb": 111161}, "en", "srt")
    k_ar = make_key("download", {"imdb": 111161, "q": None}, "ar", "srt")
    assert k1 == k2
    assert k1 != k_ar


@pytest.mark.asyncio
async def test_cache_store_and_hit(cache_store: CacheStore):
    key = make_key("download", "tt0111161", "en", "srt")
    payload = b"1\n00:00:01,000 --> 00:00:04,000\nHello!\n"

    await cache_store.set("download", key, payload, ttl_seconds=3600, meta={"format": "srt", "cue_count": 1})

    hit = await cache_store.get("download", key)
    assert hit is not None
    assert hit.value == payload
    assert hit.meta["format"] == "srt"
    assert cache_store.hits >= 1

    # Evict from memory tier to confirm SQLite persistence read works
    cache_store._mem.clear()
    hit_db = await cache_store.get("download", key)
    assert hit_db is not None
    assert hit_db.value == payload


@pytest.mark.asyncio
async def test_cache_expiration(cache_store: CacheStore):
    key = "expired_key"
    await cache_store.set("download", key, b"data", ttl_seconds=0.01)
    await asyncio.sleep(0.05)

    assert await cache_store.get("download", key) is None


@pytest.mark.asyncio
async def test_cleanup_expired_records(cache_store: CacheStore):
    await cache_store.set("download", "active", b"active-data", ttl_seconds=3600)
    await cache_store.set("download", "expired", b"old-data", ttl_seconds=10)
    assert cache_store._db is not None
    await cache_store._db.execute(
        "UPDATE cache_entries SET expires_at = ? WHERE key = ?",
        (time.time() - 100.0, "expired"),
    )
    await cache_store._db.commit()
    cache_store._mem.clear()

    removed = await cache_store.cleanup()
    assert removed >= 1
    assert await cache_store.get("download", "expired") is None
    assert await cache_store.get("download", "active") is not None


@pytest.mark.asyncio
async def test_single_flight_coalesces_concurrent_calls():
    sf: SingleFlight[str] = SingleFlight()
    calls = 0

    async def work() -> str:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return "result_value"

    results = await asyncio.gather(*(sf.run("same_key", work) for _ in range(5)))
    assert all(r == "result_value" for r in results)
    assert calls == 1
    assert sf.inflight_count == 0


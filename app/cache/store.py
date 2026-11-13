"""Two-tier cache: bounded in-memory LRU in front of a persistent SQLite store.

Values are opaque bytes plus a small JSON ``meta`` dict. Entries live in
namespaces (``search``, ``file``, ``sync``, ``speech``...) with independent TTLs.
SQLite runs in WAL mode through a single aiosqlite connection, which serializes
writes safely without blocking the event loop.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import aiosqlite

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_entries (
    namespace   TEXT    NOT NULL,
    key         TEXT    NOT NULL,
    value       BLOB    NOT NULL,
    meta        TEXT    NOT NULL DEFAULT '{}',
    size        INTEGER NOT NULL,
    created_at  REAL    NOT NULL,
    expires_at  REAL    NOT NULL,
    accessed_at REAL    NOT NULL,
    PRIMARY KEY (namespace, key)
);
CREATE INDEX IF NOT EXISTS ix_cache_expires ON cache_entries (expires_at);
CREATE INDEX IF NOT EXISTS ix_cache_accessed ON cache_entries (accessed_at);
"""


def make_key(*parts: Any) -> str:
    """Deterministic, compact key from arbitrary JSON-serializable parts."""
    raw = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


@dataclass(slots=True)
class CacheEntry:
    value: bytes
    meta: Dict[str, Any]
    expires_at: float
    created_at: float = field(default_factory=time.time)

    @property
    def expired(self) -> bool:
        return time.time() >= self.expires_at


class _MemoryLRU:
    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self._data: "OrderedDict[Tuple[str, str], CacheEntry]" = OrderedDict()
        self._bytes = 0

    def get(self, k: Tuple[str, str]) -> Optional[CacheEntry]:
        entry = self._data.get(k)
        if entry is None:
            return None
        if entry.expired:
            self.pop(k)
            return None
        self._data.move_to_end(k)
        return entry

    def put(self, k: Tuple[str, str], entry: CacheEntry) -> None:
        size = len(entry.value)
        if size > self.max_bytes // 4:
            return  # never let one item dominate the memory tier
        self.pop(k)
        self._data[k] = entry
        self._bytes += size
        while self._bytes > self.max_bytes and self._data:
            _, old = self._data.popitem(last=False)
            self._bytes -= len(old.value)

    def pop(self, k: Tuple[str, str]) -> None:
        old = self._data.pop(k, None)
        if old is not None:
            self._bytes -= len(old.value)

    def clear(self) -> None:
        self._data.clear()
        self._bytes = 0

    @property
    def size_bytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._data)


class CacheStore:
    def __init__(self, db_path: str, *, memory_max_mb: int = 64, max_size_mb: int = 1024) -> None:
        self.db_path = db_path
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._mem = _MemoryLRU(memory_max_mb * 1024 * 1024)
        self._db: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------ lifecycle
    async def open(self) -> None:
        if self._db is not None:
            return
        self._db = await aiosqlite.connect(self.db_path)
        if self.db_path != ":memory:":
            await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        logger.info("Cache store opened at %s", self.db_path)

    async def close(self) -> None:
        if self._db is not None:
            try:
                await self._db.close()
            finally:
                self._db = None

    @property
    def is_open(self) -> bool:
        return self._db is not None

    # ------------------------------------------------------------------ operations
    async def get(self, namespace: str, key: str) -> Optional[CacheEntry]:
        k = (namespace, key)
        entry = self._mem.get(k)
        if entry is not None:
            self.hits += 1
            return entry
        if self._db is None:
            self.misses += 1
            return None
        now = time.time()
        try:
            async with self._db.execute(
                "SELECT value, meta, expires_at, created_at FROM cache_entries WHERE namespace=? AND key=?",
                (namespace, key),
            ) as cur:
                row = await cur.fetchone()
            if row is None:
                self.misses += 1
                return None
            if row[2] <= now:
                await self.delete(namespace, key)
                self.misses += 1
                return None
            await self._db.execute(
                "UPDATE cache_entries SET accessed_at=? WHERE namespace=? AND key=?", (now, namespace, key)
            )
            await self._db.commit()
        except Exception as exc:  # cache must never break request handling
            logger.warning("Cache read failed (%s/%s): %s", namespace, key, exc)
            self.misses += 1
            return None
        entry = CacheEntry(bytes(row[0]), json.loads(row[1] or "{}"), float(row[2]), float(row[3]))
        self._mem.put(k, entry)
        self.hits += 1
        return entry

    async def set(
        self,
        namespace: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: float,
        meta: Optional[Dict[str, Any]] = None,
        persist: bool = True,
    ) -> None:
        now = time.time()
        entry = CacheEntry(value, dict(meta or {}), now + ttl_seconds, now)
        self._mem.put((namespace, key), entry)
        if not persist or self._db is None:
            return
        try:
            await self._db.execute(
                "INSERT OR REPLACE INTO cache_entries "
                "(namespace, key, value, meta, size, created_at, expires_at, accessed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    namespace,
                    key,
                    value,
                    json.dumps(entry.meta, ensure_ascii=False, default=str),
                    len(value),
                    now,
                    entry.expires_at,
                    now,
                ),
            )
            await self._db.commit()
        except Exception as exc:
            logger.warning("Cache write failed (%s/%s): %s", namespace, key, exc)

    async def delete(self, namespace: str, key: str) -> None:
        self._mem.pop((namespace, key))
        if self._db is None:
            return
        try:
            await self._db.execute("DELETE FROM cache_entries WHERE namespace=? AND key=?", (namespace, key))
            await self._db.commit()
        except Exception as exc:
            logger.warning("Cache delete failed: %s", exc)

    async def get_json(self, namespace: str, key: str) -> Optional[Any]:
        entry = await self.get(namespace, key)
        if entry is None:
            return None
        try:
            return json.loads(entry.value.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            await self.delete(namespace, key)
            return None

    async def set_json(self, namespace: str, key: str, obj: Any, *, ttl_seconds: float) -> None:
        payload = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        await self.set(namespace, key, payload, ttl_seconds=ttl_seconds)

    # ------------------------------------------------------------------ maintenance
    async def cleanup(self) -> int:
        """Remove expired rows and enforce the total size budget (LRU eviction)."""
        if self._db is None:
            return 0
        async with self._lock:
            removed = 0
            try:
                cur = await self._db.execute("DELETE FROM cache_entries WHERE expires_at <= ?", (time.time(),))
                removed += cur.rowcount or 0
                async with self._db.execute("SELECT COALESCE(SUM(size), 0) FROM cache_entries") as c:
                    total = int((await c.fetchone())[0])
                if total > self.max_size_bytes:
                    target = int(self.max_size_bytes * 0.8)
                    async with self._db.execute(
                        "SELECT namespace, key, size FROM cache_entries ORDER BY accessed_at ASC"
                    ) as c:
                        victims = []
                        async for ns, key, size in c:
                            if total <= target:
                                break
                            victims.append((ns, key))
                            total -= int(size)
                    await self._db.executemany(
                        "DELETE FROM cache_entries WHERE namespace=? AND key=?", victims
                    )
                    for v in victims:
                        self._mem.pop(v)
                    removed += len(victims)
                await self._db.commit()
            except Exception as exc:
                logger.warning("Cache cleanup failed: %s", exc)
            if removed:
                logger.info("Cache cleanup removed %d entries.", removed)
            return removed

    async def stats(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "memory_entries": len(self._mem),
            "memory_bytes": self._mem.size_bytes,
            "hits": self.hits,
            "misses": self.misses,
        }
        if self._db is not None:
            try:
                async with self._db.execute(
                    "SELECT namespace, COUNT(*), COALESCE(SUM(size),0) FROM cache_entries GROUP BY namespace"
                ) as c:
                    info["namespaces"] = {ns: {"entries": n, "bytes": b} async for ns, n, b in c}
            except Exception:
                pass
        return info


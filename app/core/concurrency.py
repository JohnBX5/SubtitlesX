"""Async concurrency primitives."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from typing import Any, Awaitable, Callable, Deque, Dict, Generic, TypeVar

T = TypeVar("T")


class SingleFlight(Generic[T]):
    """Coalesce concurrent calls for the same key into a single execution.

    Unlike a dict of per-key locks, entries are removed as soon as the call
    finishes, so memory stays bounded no matter how many distinct keys are seen.
    """

    def __init__(self) -> None:
        self._inflight: Dict[str, asyncio.Future] = {}

    @property
    def inflight_count(self) -> int:
        return len(self._inflight)

    async def run(self, key: str, fn: Callable[[], Awaitable[T]]) -> T:
        existing = self._inflight.get(key)
        if existing is not None:
            return await asyncio.shield(existing)

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._inflight[key] = future
        try:
            result = await fn()
        except BaseException as exc:
            if not future.done():
                if isinstance(exc, asyncio.CancelledError):
                    future.cancel()
                else:
                    future.set_exception(exc)
                    future.exception()  # mark retrieved: avoids "never retrieved" warnings
            raise
        else:
            if not future.done():
                future.set_result(result)
            return result
        finally:
            self._inflight.pop(key, None)


class SlidingWindowRateLimiter:
    """Small in-memory per-key sliding-window limiter (per process)."""

    def __init__(self, limit: int, window_seconds: float = 60.0, max_keys: int = 10000) -> None:
        self.limit = limit
        self.window = window_seconds
        self.max_keys = max_keys
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def check(self, key: str) -> tuple[bool, float]:
        """Register a hit. Returns (allowed, retry_after_seconds)."""
        if self.limit <= 0:
            return True, 0.0
        now = time.monotonic()
        if len(self._hits) > self.max_keys:
            self._prune(now)
        bucket = self._hits[key]
        while bucket and now - bucket[0] > self.window:
            bucket.popleft()
        if len(bucket) >= self.limit:
            return False, max(0.0, self.window - (now - bucket[0]))
        bucket.append(now)
        return True, 0.0

    def _prune(self, now: float) -> None:
        for k in [k for k, b in self._hits.items() if not b or now - b[-1] > self.window]:
            self._hits.pop(k, None)


async def gather_limited(limit: int, coros: list[Awaitable[Any]]) -> list[Any]:
    sem = asyncio.Semaphore(max(1, limit))

    async def _wrap(c: Awaitable[Any]) -> Any:
        async with sem:
            return await c

    return await asyncio.gather(*(_wrap(c) for c in coros))

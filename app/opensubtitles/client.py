"""Async client for the OpenSubtitles REST API v1.

Features: pooled HTTP/1.1 connections, bounded concurrency, retries with
exponential backoff (honouring ``Retry-After``), automatic login/token renewal
with single-flight protection, VIP ``base_url`` support, sorted/lower-cased
query parameters (which the API requires to avoid redirects), and size-capped
streaming downloads.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.core.config import Settings
from app.core.errors import (
    AppError,
    RateLimitedError,
    ServiceUnavailableError,
    SubtitleValidationError,
    UpstreamError,
)
from app.opensubtitles.languages import language_name
from app.opensubtitles.models import (
    LanguageInfo,
    MovieInfo,
    SubtitleCandidate,
    SubtitleFile,
    TitleQuery,
    format_imdb_id,
    normalize_imdb_id,
)

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {500, 502, 503, 504, 520, 522, 524}
TOKEN_LIFETIME_SECONDS = 23 * 3600


class OpenSubtitlesClient:
    def __init__(self, settings: Settings, *, transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        self.settings = settings
        self._transport = transport
        self._client: Optional[httpx.AsyncClient] = None
        self._sem = asyncio.Semaphore(max(1, settings.MAX_CONCURRENT_REQUESTS))
        self._auth_lock = asyncio.Lock()
        self._token: Optional[str] = None
        self._token_expires_at = 0.0
        self._api_base = settings.OPENSUBTITLES_BASE_URL.rstrip("/")
        self._download_base: Optional[str] = None  # VIP accounts receive a dedicated host
        self.remaining_downloads: Optional[int] = None

    # ------------------------------------------------------------------ lifecycle
    @property
    def configured(self) -> bool:
        return bool(self.settings.OPENSUBTITLES_API_KEY)

    @property
    def has_credentials(self) -> bool:
        return bool(self.settings.OPENSUBTITLES_USERNAME and self.settings.OPENSUBTITLES_PASSWORD)

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            limit = max(2, self.settings.MAX_CONCURRENT_REQUESTS)
            self._client = httpx.AsyncClient(
                transport=self._transport,
                timeout=httpx.Timeout(self.settings.HTTP_TIMEOUT_SECONDS, connect=10.0),
                limits=httpx.Limits(max_connections=limit * 2, max_keepalive_connections=limit),
                follow_redirects=True,
                headers={"User-Agent": self.settings.OPENSUBTITLES_USER_AGENT},
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    def _headers(self, token: Optional[str] = None) -> Dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Api-Key": self.settings.OPENSUBTITLES_API_KEY,
            "User-Agent": self.settings.OPENSUBTITLES_USER_AGENT,
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _require_configured(self) -> None:
        if not self.configured:
            raise ServiceUnavailableError("OpenSubtitles API key is not configured on the server.")

    # ------------------------------------------------------------------ transport
    async def _request(
        self,
        method: str,
        url: str,
        *,
        params: Optional[List[Tuple[str, Any]]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        token: Optional[str] = None,
    ) -> httpx.Response:
        retries = max(1, self.settings.OPENSUBTITLES_MAX_RETRIES)
        last_exc: Optional[Exception] = None
        async with self._sem:
            for attempt in range(1, retries + 1):
                try:
                    resp = await self._http().request(
                        method, url, params=params, json=json_body, headers=self._headers(token)
                    )
                except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                    last_exc = exc
                    logger.warning("OpenSubtitles %s %s network error (%s), attempt %d/%d",
                                   method, url, exc.__class__.__name__, attempt, retries)
                    if attempt < retries:
                        await asyncio.sleep(self._backoff(attempt))
                        continue
                    raise UpstreamError("Network error communicating with OpenSubtitles.", status_code=504) from exc

                if resp.status_code == 429:
                    wait = self._retry_after(resp, attempt)
                    if attempt < retries and wait <= 10.0:
                        logger.warning("OpenSubtitles rate limited; retrying in %.1fs", wait)
                        await asyncio.sleep(wait)
                        continue
                    raise RateLimitedError("OpenSubtitles rate limit exceeded. Please retry shortly.", retry_after=wait)
                if resp.status_code in _RETRYABLE_STATUS:
                    if attempt < retries:
                        await asyncio.sleep(self._backoff(attempt))
                        continue
                    raise UpstreamError(f"OpenSubtitles service unavailable (HTTP {resp.status_code}).")
                return resp
        raise UpstreamError("OpenSubtitles request failed.") from last_exc

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(0.5 * (2 ** (attempt - 1)), 6.0) + random.uniform(0, 0.25)

    def _retry_after(self, resp: httpx.Response, attempt: int) -> float:
        header = resp.headers.get("Retry-After") or resp.headers.get("ratelimit-reset")
        try:
            return max(0.5, float(header)) if header else self._backoff(attempt) + 1.0
        except ValueError:
            return self._backoff(attempt) + 1.0

    @staticmethod
    def _json(resp: httpx.Response) -> Dict[str, Any]:
        try:
            data = resp.json()
        except ValueError as exc:
            raise UpstreamError("OpenSubtitles returned an invalid JSON response.") from exc
        if not isinstance(data, dict):
            raise UpstreamError("OpenSubtitles returned an unexpected response shape.")
        return data

    # ------------------------------------------------------------------ auth
    @property
    def token_valid(self) -> bool:
        return bool(self._token and time.time() < self._token_expires_at)

    async def _get_token(self, *, force: bool = False) -> Optional[str]:
        if not self.has_credentials:
            return None
        if not force and self.token_valid:
            return self._token
        async with self._auth_lock:
            if not force and self.token_valid:
                return self._token
            resp = await self._request(
                "POST",
                f"{self._api_base}/login",
                json_body={
                    "username": self.settings.OPENSUBTITLES_USERNAME,
                    "password": self.settings.OPENSUBTITLES_PASSWORD,
                },
            )
            if resp.status_code in (400, 401):
                raise UpstreamError("OpenSubtitles rejected the configured username/password.")
            if resp.status_code == 403:
                raise UpstreamError("OpenSubtitles rejected the configured API key.")
            if resp.status_code != 200:
                raise UpstreamError(f"OpenSubtitles login failed (HTTP {resp.status_code}).")
            data = self._json(resp)
            token = data.get("token")
            if not isinstance(token, str) or not token:
                raise UpstreamError("OpenSubtitles login response did not include a token.")
            base_url = data.get("base_url")
            if isinstance(base_url, str) and base_url.strip():
                host = base_url.strip().rstrip("/")
                if not host.startswith("http"):
                    host = f"https://{host}"
                self._download_base = host if host.endswith("/api/v1") else f"{host}/api/v1"
            self._token = token
            self._token_expires_at = time.time() + TOKEN_LIFETIME_SECONDS
            user = data.get("user") or {}
            logger.info("OpenSubtitles login OK (level=%s, allowed_downloads=%s)",
                        user.get("level"), user.get("allowed_downloads"))
            return token

    def invalidate_token(self) -> None:
        self._token = None
        self._token_expires_at = 0.0

    # ------------------------------------------------------------------ search
    @staticmethod
    def _uses_parent_id(q: TitleQuery) -> bool:
        return (
            normalize_imdb_id(q.imdb_id) is not None
            and q.media_type == "episode"
            and bool(q.season_number or q.episode_number)
        )

    @classmethod
    def _search_params(
        cls, q: TitleQuery, languages: Optional[str], page: int, *, parent: Optional[bool] = None
    ) -> List[Tuple[str, Any]]:
        params: Dict[str, Any] = {}
        imdb = normalize_imdb_id(q.imdb_id)
        use_parent = cls._uses_parent_id(q) if parent is None else parent
        if imdb is not None:
            if use_parent:
                params["parent_imdb_id"] = imdb
            else:
                params["imdb_id"] = imdb
        if q.query and q.query.strip() and imdb is None:
            params["query"] = q.query.strip().lower()
        if q.year:
            params["year"] = q.year
        if q.media_type in {"movie", "episode"}:
            params["type"] = q.media_type
        if q.season_number is not None:
            params["season_number"] = q.season_number
        if q.episode_number is not None:
            params["episode_number"] = q.episode_number
        if languages:
            params["languages"] = ",".join(sorted({x.strip().lower() for x in languages.split(",") if x.strip()}))
        if page > 1:
            params["page"] = page
        # The API asks for alphabetically sorted, lower-case parameters.
        return sorted(params.items())

    @staticmethod
    def _parse_candidate(item: Dict[str, Any]) -> Optional[SubtitleCandidate]:
        attrs = item.get("attributes") or {}
        lang = str(attrs.get("language") or "").lower()
        files = []
        for f in attrs.get("files") or []:
            fid = f.get("file_id")
            if fid is None:
                continue
            try:
                files.append(SubtitleFile(int(fid), str(f.get("file_name") or ""), int(f.get("cd_number") or 1)))
            except (TypeError, ValueError):
                continue
        if not lang or not files:
            return None
        uploader = attrs.get("uploader") or {}

        def num(key: str, cast=int, default=0):
            try:
                return cast(attrs.get(key) or default)
            except (TypeError, ValueError):
                return default

        fps = attrs.get("fps")
        return SubtitleCandidate(
            id=str(item.get("id", "")),
            subtitle_id=str(attrs.get("subtitle_id") or item.get("id", "")),
            language=lang,
            download_count=num("download_count"),
            new_download_count=num("new_download_count"),
            hearing_impaired=bool(attrs.get("hearing_impaired")),
            hd=bool(attrs.get("hd")),
            fps=float(fps) if isinstance(fps, (int, float)) and fps > 0 else None,
            votes=num("votes"),
            ratings=num("ratings", float, 0.0),
            from_trusted=bool(attrs.get("from_trusted")),
            foreign_parts_only=bool(attrs.get("foreign_parts_only")),
            ai_translated=bool(attrs.get("ai_translated")),
            machine_translated=bool(attrs.get("machine_translated")),
            release=attrs.get("release") or None,
            uploader_rank=uploader.get("rank") if isinstance(uploader, dict) else None,
            files=sorted(files, key=lambda f: f.cd_number),
        )

    @staticmethod
    def _parse_movie(data: List[Dict[str, Any]], q: TitleQuery) -> Optional[MovieInfo]:
        for item in data:
            feature = (item.get("attributes") or {}).get("feature_details") or {}
            if feature:
                title = feature.get("parent_title") if q.media_type == "episode" and feature.get("parent_title") else None
                title = title or feature.get("title") or feature.get("movie_name")
                imdb = feature.get("parent_imdb_id") if q.media_type == "episode" and feature.get("parent_imdb_id") else feature.get("imdb_id")
                year = feature.get("year")
                return MovieInfo(
                    imdb_id=format_imdb_id(imdb) or q.imdb_id,
                    title=title or q.query,
                    year=int(year) if isinstance(year, (int, float)) and year else None,
                )
        if q.imdb_id or q.query:
            return MovieInfo(imdb_id=format_imdb_id(normalize_imdb_id(q.imdb_id)) or q.imdb_id, title=q.query)
        return None

    async def search_page(
        self, q: TitleQuery, languages: Optional[str] = None, page: int = 1, *, parent: Optional[bool] = None
    ) -> Tuple[Optional[MovieInfo], List[SubtitleCandidate], int]:
        self._require_configured()
        params = self._search_params(q, languages, page, parent=parent)
        resp = await self._request("GET", f"{self._api_base}/subtitles", params=params)
        if resp.status_code in (401, 403):
            raise UpstreamError("OpenSubtitles API key is unauthorized or invalid.")
        if resp.status_code != 200:
            raise UpstreamError(f"OpenSubtitles search failed (HTTP {resp.status_code}).")
        body = self._json(resp)
        data = body.get("data") or []
        total_pages = int(body.get("total_pages") or 1)
        candidates = [c for c in (self._parse_candidate(i) for i in data) if c is not None]
        return self._parse_movie(data, q), candidates, total_pages

    async def search(
        self, q: TitleQuery, languages: Optional[str] = None, max_pages: int = 1
    ) -> Tuple[Optional[MovieInfo], List[SubtitleCandidate]]:
        parent: Optional[bool] = None
        movie, candidates, total = await self.search_page(q, languages, 1)
        if not candidates and self._uses_parent_id(q):
            # The client may have sent the episode's own IMDb id rather than the show's.
            parent = False
            movie, candidates, total = await self.search_page(q, languages, 1, parent=False)
        pages = min(max(1, max_pages), total)
        if pages > 1:
            results = await asyncio.gather(
                *(self.search_page(q, languages, p, parent=parent) for p in range(2, pages + 1)),
                return_exceptions=True,
            )
            for res in results:
                if isinstance(res, BaseException):
                    logger.warning("Extra search page failed: %s", res)
                    continue
                candidates.extend(res[1])
        # De-duplicate (pages can overlap when results shift between calls).
        unique: Dict[str, SubtitleCandidate] = {}
        for c in candidates:
            unique.setdefault(c.id or c.subtitle_id, c)
        return movie, list(unique.values())

    async def languages(self, q: TitleQuery) -> Tuple[Optional[MovieInfo], List[LanguageInfo]]:
        movie, candidates = await self.search(q, None, self.settings.LANGUAGES_MAX_PAGES)
        counts: Dict[str, int] = {}
        for c in candidates:
            counts[c.language] = counts.get(c.language, 0) + len(c.files)
        items = [
            LanguageInfo(code, language_name(code), n)
            for code, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        return movie, items

    # ------------------------------------------------------------------ download
    async def request_download(self, file_id: int) -> Tuple[str, str]:
        self._require_configured()
        token = await self._get_token()
        for attempt in (1, 2):
            base = self._download_base or self._api_base
            resp = await self._request("POST", f"{base}/download", json_body={"file_id": int(file_id)}, token=token)
            if resp.status_code == 401 and attempt == 1 and self.has_credentials:
                logger.info("OpenSubtitles token rejected; re-authenticating.")
                self.invalidate_token()
                token = await self._get_token(force=True)
                continue
            break
        if resp.status_code == 406:
            detail = ""
            try:
                body = resp.json()
                detail = f" Quota resets at {body.get('reset_time_utc') or body.get('reset_time')}."
            except ValueError:
                pass
            raise RateLimitedError("OpenSubtitles daily download quota exhausted." + detail)
        if resp.status_code == 404:
            raise AppError("Subtitle file no longer exists on OpenSubtitles.", status_code=404)
        if resp.status_code != 200:
            raise UpstreamError(f"OpenSubtitles download request rejected (HTTP {resp.status_code}).")
        data = self._json(resp)
        link = data.get("link")
        if not isinstance(link, str) or not link.startswith("http"):
            raise UpstreamError("OpenSubtitles download response is missing the file link.")
        remaining = data.get("remaining")
        if isinstance(remaining, int):
            self.remaining_downloads = remaining
        return link, str(data.get("file_name") or f"{file_id}.srt")

    async def fetch_file(self, link: str) -> bytes:
        limit = self.settings.max_file_size_bytes
        try:
            async with self._http().stream("GET", link, headers={"User-Agent": self.settings.OPENSUBTITLES_USER_AGENT}) as resp:
                if resp.status_code != 200:
                    raise UpstreamError(f"Subtitle file download failed (HTTP {resp.status_code}).")
                declared = resp.headers.get("Content-Length")
                if declared and declared.isdigit() and int(declared) > limit:
                    raise SubtitleValidationError("Subtitle file exceeds the maximum permitted size.")
                chunks: List[bytes] = []
                size = 0
                async for chunk in resp.aiter_bytes():
                    size += len(chunk)
                    if size > limit:
                        raise SubtitleValidationError("Subtitle file exceeds the maximum permitted size.")
                    chunks.append(chunk)
        except httpx.TimeoutException as exc:
            raise UpstreamError("Timed out downloading the subtitle file.", status_code=504) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError("Network error downloading the subtitle file.") from exc
        data = b"".join(chunks)
        if not data:
            raise UpstreamError("Received an empty subtitle file from OpenSubtitles.")
        return data

    async def download_file(self, file_id: int) -> Tuple[bytes, str]:
        link, name = await self.request_download(file_id)
        return await self.fetch_file(link), name

"""Application orchestration service tying together OpenSubtitles, subtitle processing, caching, and sync."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from fastapi import Response

from app.cache.store import CacheStore, make_key
from app.core.concurrency import SingleFlight, SlidingWindowRateLimiter, gather_limited
from app.core.config import Settings
from app.core.errors import (
    AppError,
    BadRequestError,
    NotFoundError,
    RateLimitedError,
)
from app.core.security import stream_fingerprint
from app.opensubtitles.client import OpenSubtitlesClient
from app.opensubtitles.models import (
    LanguageInfo as DomainLanguageInfo,
    MovieInfo as DomainMovieInfo,
    SubtitleCandidate as DomainCandidate,
    TitleQuery,
)
from app.opensubtitles.ranking import select_top
from app.schemas.subtitle import (
    LanguageInfo,
    LanguagesResponse,
    MovieInfo,
    SyncRequest,
    SyncResponse,
)
from app.subtitles.processing import (
    build_download_filename,
    load_subtitle_bytes,
    render_document,
    sanitize_filename,
)
from app.sync.engine import CandidateSyncInput, SyncEngine, SyncResult

logger = logging.getLogger(__name__)

LANGUAGE_CODE_REGEX = re.compile(r"^[a-zA-Z]{2,3}(-[a-zA-Z0-9]{2,4})?$")


@dataclass(slots=True)
class _PreparedDownload:
    content_bytes: bytes
    media_type: str
    headers: Dict[str, str]


class SubtitleBackendService:
    """High-level application service for discovery, download, and synchronization."""

    def __init__(
        self,
        settings: Settings,
        cache: CacheStore,
        opensubtitles: OpenSubtitlesClient,
        sync_engine: SyncEngine,
    ) -> None:
        self.settings = settings
        self.cache = cache
        self.opensubtitles = opensubtitles
        self.sync_engine = sync_engine
        self._flight: SingleFlight[Any] = SingleFlight()
        self._sync_limiter = SlidingWindowRateLimiter(
            limit=settings.RATE_LIMIT_SYNC_PER_MINUTE,
            window_seconds=60.0,
        )

    @staticmethod
    def validate_title_identification(imdb_id: Optional[str], query: Optional[str]) -> None:
        if not (imdb_id and imdb_id.strip()) and not (query and query.strip()):
            raise BadRequestError(
                "At least one valid identification method is required: 'imdb_id' or 'query'."
            )

    @staticmethod
    def validate_language_code(language: str) -> str:
        clean = (language or "").strip().lower()
        if not LANGUAGE_CODE_REGEX.match(clean):
            raise BadRequestError(
                f"Invalid language code '{language}'. Must be a valid 2-3 letter ISO code (e.g. 'en', 'ar')."
            )
        return clean

    def check_sync_rate_limit(self, client_ip: str) -> None:
        allowed, retry_after = self._sync_limiter.check(client_ip or "unknown")
        if not allowed:
            raise RateLimitedError(
                "Synchronization rate limit exceeded. Please wait before retrying.",
                retry_after=retry_after,
            )

    # ------------------------------------------------------------------ search & file helpers
    async def _search_cached(
        self, tq: TitleQuery, clean_lang: Optional[str]
    ) -> Tuple[Optional[DomainMovieInfo], List[DomainCandidate]]:
        search_key = make_key("search", tq.cache_parts(), clean_lang)
        cached = await self.cache.get_json("search", search_key)
        if isinstance(cached, dict):
            movie_raw = cached.get("movie")
            movie = DomainMovieInfo(**movie_raw) if isinstance(movie_raw, dict) else None
            cands = [DomainCandidate.from_dict(item) for item in cached.get("candidates", [])]
            return movie, cands

        async def _fetch() -> Tuple[Optional[DomainMovieInfo], List[DomainCandidate]]:
            cached_inner = await self.cache.get_json("search", search_key)
            if isinstance(cached_inner, dict):
                movie_raw = cached_inner.get("movie")
                movie = DomainMovieInfo(**movie_raw) if isinstance(movie_raw, dict) else None
                cands = [DomainCandidate.from_dict(item) for item in cached_inner.get("candidates", [])]
                return movie, cands

            movie, cands = await self.opensubtitles.search(tq, languages=clean_lang, max_pages=1)
            if cands:
                await self.cache.set_json(
                    "search",
                    search_key,
                    {
                        "movie": (
                            {"imdb_id": movie.imdb_id, "title": movie.title, "year": movie.year}
                            if movie
                            else None
                        ),
                        "candidates": [c.to_dict() for c in cands],
                    },
                    ttl_seconds=float(self.settings.SEARCH_CACHE_TTL_HOURS) * 3600.0,
                )
            return movie, cands

        return await self._flight.run(f"search:{search_key}", _fetch)

    async def _fetch_raw_file(self, file_id: int) -> Tuple[bytes, str]:
        file_key = str(file_id)
        cached = await self.cache.get("file", file_key)
        if cached is not None:
            return cached.value, str(cached.meta.get("file_name") or f"{file_id}.srt")

        async def _download() -> Tuple[bytes, str]:
            cached_inner = await self.cache.get("file", file_key)
            if cached_inner is not None:
                return cached_inner.value, str(cached_inner.meta.get("file_name") or f"{file_id}.srt")

            raw_bytes, remote_name = await self.opensubtitles.download_file(file_id)
            await self.cache.set(
                "file",
                file_key,
                raw_bytes,
                ttl_seconds=float(self.settings.FILE_CACHE_TTL_HOURS) * 3600.0,
                meta={"file_name": remote_name},
            )
            return raw_bytes, remote_name

        return await self._flight.run(f"file:{file_key}", _download)

    # ------------------------------------------------------------------ /languages
    async def discover_languages(
        self,
        *,
        imdb_id: Optional[str] = None,
        query: Optional[str] = None,
        year: Optional[int] = None,
        media_type: Optional[str] = None,
        season_number: Optional[int] = None,
        episode_number: Optional[int] = None,
    ) -> LanguagesResponse:
        self.validate_title_identification(imdb_id, query)
        tq = TitleQuery(
            imdb_id=imdb_id,
            query=query,
            year=year,
            media_type=media_type,
            season_number=season_number,
            episode_number=episode_number,
        )
        lang_key = make_key("languages", tq.cache_parts())

        cached = await self.cache.get_json("languages", lang_key)
        if isinstance(cached, dict):
            movie_raw = cached.get("movie")
            movie_schema = MovieInfo(**movie_raw) if isinstance(movie_raw, dict) else None
            langs_schema = [LanguageInfo(**item) for item in cached.get("languages", [])]
            return LanguagesResponse(success=True, movie=movie_schema, languages=langs_schema)

        async def _compute() -> Dict[str, Any]:
            cached_inner = await self.cache.get_json("languages", lang_key)
            if isinstance(cached_inner, dict):
                return cached_inner
            movie, langs = await self.opensubtitles.languages(tq)
            payload = {
                "movie": (
                    {"imdb_id": movie.imdb_id, "title": movie.title}
                    if movie is not None
                    else None
                ),
                "languages": [
                    {
                        "language_code": item.language_code,
                        "language_name": item.language_name,
                        "available_files": item.available_files,
                    }
                    for item in langs
                ],
            }
            if langs:
                await self.cache.set_json(
                    "languages",
                    lang_key,
                    payload,
                    ttl_seconds=float(self.settings.SEARCH_CACHE_TTL_HOURS) * 3600.0,
                )
            return payload

        payload = await self._flight.run(f"languages:{lang_key}", _compute)
        movie_raw = payload.get("movie")
        movie_schema = MovieInfo(**movie_raw) if isinstance(movie_raw, dict) else None
        langs_schema = [LanguageInfo(**item) for item in payload.get("languages", [])]
        return LanguagesResponse(success=True, movie=movie_schema, languages=langs_schema)

    # ------------------------------------------------------------------ /download
    async def download_subtitle(
        self,
        *,
        language: str,
        imdb_id: Optional[str] = None,
        query: Optional[str] = None,
        year: Optional[int] = None,
        media_type: Optional[str] = "movie",
        season_number: Optional[int] = None,
        episode_number: Optional[int] = None,
        target_format: str = "srt",
        hearing_impaired: bool = False,
        sync: bool = False,
        stream_url: Optional[str] = None,
        client_ip: str = "unknown",
    ) -> Response:
        self.validate_title_identification(imdb_id, query)
        clean_lang = self.validate_language_code(language)
        fmt = (target_format or "srt").lower()

        if sync and stream_url:
            self.check_sync_rate_limit(client_ip)

        tq = TitleQuery(
            imdb_id=imdb_id,
            query=query,
            year=year,
            media_type=media_type,
            season_number=season_number,
            episode_number=episode_number,
        )

        is_sync_req = bool(sync and stream_url)
        namespace = "sync" if is_sync_req else "download"
        cache_key = (
            make_key("sync", tq.cache_parts(), clean_lang, fmt, hearing_impaired, stream_fingerprint(stream_url or ""))
            if is_sync_req
            else make_key("download", tq.cache_parts(), clean_lang, fmt, hearing_impaired)
        )

        cached = await self.cache.get(namespace, cache_key)
        if cached is not None:
            meta = cached.meta
            rec_fmt = str(meta.get("format") or fmt)
            cue_count = int(meta.get("cue_count") or 0)
            filename = sanitize_filename(str(meta.get("filename") or f"subtitle.{rec_fmt}"), fallback_ext=rec_fmt)
            media_type_header = "text/vtt; charset=utf-8" if rec_fmt == "vtt" else "text/plain; charset=utf-8"
            headers = {
                "Content-Disposition": f"attachment; filename=\"{filename}\"; filename*=UTF-8''{quote(filename)}",
                "X-Cache": "HIT",
                "X-Subtitle-Format": rec_fmt,
                "X-Subtitle-Cue-Count": str(cue_count),
                "Cache-Control": "public, max-age=86400",
            }
            if is_sync_req:
                headers["X-Sync-Status"] = str(meta.get("sync_status") or "synchronized")
                if meta.get("offset_seconds") is not None:
                    headers["X-Sync-Offset"] = f"{float(meta['offset_seconds']):.3f}"
                headers["X-Audio-Analyzed"] = str(bool(meta.get("audio_analyzed", True))).lower()
            return Response(content=cached.value, media_type=media_type_header, headers=headers)

        async def _build() -> _PreparedDownload:
            cached_inner = await self.cache.get(namespace, cache_key)
            if cached_inner is not None:
                meta = cached_inner.meta
                rec_fmt = str(meta.get("format") or fmt)
                cue_count = int(meta.get("cue_count") or 0)
                filename = sanitize_filename(str(meta.get("filename") or f"subtitle.{rec_fmt}"), fallback_ext=rec_fmt)
                media_type_header = "text/vtt; charset=utf-8" if rec_fmt == "vtt" else "text/plain; charset=utf-8"
                headers = {
                    "Content-Disposition": f"attachment; filename=\"{filename}\"; filename*=UTF-8''{quote(filename)}",
                    "X-Cache": "HIT",
                    "X-Subtitle-Format": rec_fmt,
                    "X-Subtitle-Cue-Count": str(cue_count),
                    "Cache-Control": "public, max-age=86400",
                }
                if is_sync_req:
                    headers["X-Sync-Status"] = str(meta.get("sync_status") or "synchronized")
                    if meta.get("offset_seconds") is not None:
                        headers["X-Sync-Offset"] = f"{float(meta['offset_seconds']):.3f}"
                    headers["X-Audio-Analyzed"] = str(bool(meta.get("audio_analyzed", True))).lower()
                return _PreparedDownload(cached_inner.value, media_type_header, headers)

            movie_info, candidates = await self._search_cached(tq, clean_lang)
            if not candidates:
                raise NotFoundError(
                    f"No subtitles found for the requested title in language '{clean_lang}'."
                )

            movie_title = (movie_info.title if movie_info and movie_info.title else None) or query
            top = select_top(
                candidates,
                limit=1,
                release_hint=movie_title,
                prefer_hearing_impaired=hearing_impaired,
            )
            if not top or not top[0].primary_file:
                raise NotFoundError("No suitable subtitle files available for download.")

            target_file = top[0].primary_file
            raw_bytes, remote_name = await self._fetch_raw_file(target_file.file_id)

            doc = load_subtitle_bytes(
                raw_bytes,
                language=clean_lang,
                filename_hint=remote_name,
                max_bytes=self.settings.max_file_size_bytes,
                max_cues=self.settings.MAX_SUBTITLE_CUE_COUNT,
                strip_ads=self.settings.STRIP_SUBTITLE_ADS,
            )
            rendered = render_document(doc, fmt)

            sync_status = "none"
            sync_offset: Optional[float] = 0.0
            audio_analyzed = False

            if is_sync_req:
                sync_res = await self.sync_engine.synchronize_subtitle(
                    subtitle_content=rendered.content,
                    stream_url=stream_url,
                    target_format=fmt,
                    language=clean_lang,
                )
                sync_status = sync_res.status
                sync_offset = sync_res.offset_seconds if sync_res.offset_seconds is not None else 0.0
                audio_analyzed = sync_res.audio_analyzed
                synced_doc = load_subtitle_bytes(
                    sync_res.subtitle_content.encode("utf-8"),
                    language=clean_lang,
                    filename_hint=f"synced.{fmt}",
                    max_bytes=self.settings.max_file_size_bytes,
                    max_cues=self.settings.MAX_SUBTITLE_CUE_COUNT,
                    strip_ads=False,
                )
                rendered = render_document(synced_doc, fmt)

            safe_dl_filename = build_download_filename(movie_title, clean_lang, rendered.format)
            meta = {
                "format": rendered.format,
                "cue_count": rendered.cue_count,
                "filename": safe_dl_filename,
                "subtitle_id": top[0].subtitle_id,
                "file_id": target_file.file_id,
            }
            if is_sync_req:
                meta["sync_status"] = sync_status
                meta["offset_seconds"] = sync_offset
                meta["audio_analyzed"] = audio_analyzed

            ttl_hours = self.settings.SYNC_CACHE_TTL_HOURS if is_sync_req else self.settings.CACHE_TTL_HOURS
            await self.cache.set(
                namespace,
                cache_key,
                rendered.content_bytes,
                ttl_seconds=float(ttl_hours) * 3600.0,
                meta=meta,
            )

            resp_headers = {
                "Content-Disposition": (
                    f"attachment; filename=\"{safe_dl_filename}\"; filename*=UTF-8''{quote(safe_dl_filename)}"
                ),
                "X-Cache": "MISS",
                "X-Subtitle-Format": rendered.format,
                "X-Subtitle-Cue-Count": str(rendered.cue_count),
                "Cache-Control": "public, max-age=86400",
            }
            if is_sync_req:
                resp_headers["X-Sync-Status"] = sync_status
                resp_headers["X-Sync-Offset"] = f"{(sync_offset or 0.0):.3f}"
                resp_headers["X-Audio-Analyzed"] = str(audio_analyzed).lower()

            return _PreparedDownload(rendered.content_bytes, rendered.media_type, resp_headers)

        prepared = await self._flight.run(f"{namespace}:{cache_key}", _build)
        return Response(
            content=prepared.content_bytes,
            media_type=prepared.media_type,
            headers=prepared.headers,
        )

    # ------------------------------------------------------------------ /synchronize
    async def synchronize_subtitle(
        self,
        req: SyncRequest,
        *,
        client_ip: str = "unknown",
    ) -> SyncResponse:
        if not req.subtitle_content and not req.imdb_id and not req.query:
            raise BadRequestError(
                "At least one subtitle source is required: 'subtitle_content', 'imdb_id', or 'query'."
            )

        clean_lang = self.validate_language_code(req.language)
        target_format = (req.format or "srt").lower()

        if req.stream_url:
            self.check_sync_rate_limit(client_ip)

        tq = TitleQuery(
            imdb_id=req.imdb_id,
            query=req.query,
            year=req.year,
            media_type=req.media_type,
            season_number=req.season_number,
            episode_number=req.episode_number,
        )

        # 1. Check sync cache when stream_url is present and not user-supplied subtitle
        if req.stream_url and not req.subtitle_content:
            sync_cache_key = make_key(
                "sync",
                tq.cache_parts(),
                clean_lang,
                target_format,
                req.hearing_impaired,
                stream_fingerprint(req.stream_url),
            )
            cached = await self.cache.get("sync", sync_cache_key)
            if cached is not None:
                meta = cached.meta
                return SyncResponse(
                    success=True,
                    sync_status=str(meta.get("sync_status") or "synchronized"),
                    language=clean_lang,
                    format=str(meta.get("format") or target_format),
                    audio_analyzed=bool(meta.get("audio_analyzed", True)),
                    offset_seconds=meta.get("offset_seconds", 0.0),
                    scale_factor=float(meta.get("scale_factor", 1.0)),
                    alignment_method=str(meta.get("alignment_method", "global_fft")),
                    segments_count=int(meta.get("segments_count", 1)),
                    candidates_evaluated=int(meta.get("candidates_evaluated", 1)),
                    selected_candidate_id=meta.get("selected_candidate_id"),
                    confidence_score=meta.get("confidence_score", 1.0),
                    processing_time_seconds=0.005,
                    cues_count=int(meta.get("cue_count") or 0),
                    subtitle_content=cached.value.decode("utf-8", errors="replace"),
                    detail="Retrieved from persistent synchronization cache.",
                )

        # 2. Gather candidate inputs
        candidate_inputs: List[CandidateSyncInput] = []
        if req.subtitle_content:
            candidate_inputs.append(
                CandidateSyncInput(
                    candidate_id="custom_supplied",
                    subtitle_content=req.subtitle_content,
                    metadata_score=100.0,
                )
            )
        else:
            movie_info, candidates = await self._search_cached(tq, clean_lang)
            if not candidates:
                raise NotFoundError(
                    f"No subtitles found for requested title in language '{clean_lang}'."
                )

            max_cands = self.settings.SYNC_MAX_CANDIDATES if req.stream_url else 1
            hint = req.query or (movie_info.title if movie_info else None)
            top_candidates = select_top(
                candidates,
                limit=max_cands,
                release_hint=hint,
                prefer_hearing_impaired=req.hearing_impaired,
            )
            if not top_candidates:
                raise NotFoundError("No suitable subtitle files available for download.")

            async def _prepare_candidate(cand: DomainCandidate) -> Optional[CandidateSyncInput]:
                assert cand.primary_file is not None
                raw_bytes, remote_name = await self._fetch_raw_file(cand.primary_file.file_id)
                doc = load_subtitle_bytes(
                    raw_bytes,
                    language=clean_lang,
                    filename_hint=remote_name,
                    max_bytes=self.settings.max_file_size_bytes,
                    max_cues=self.settings.MAX_SUBTITLE_CUE_COUNT,
                    strip_ads=self.settings.STRIP_SUBTITLE_ADS,
                )
                rendered = render_document(doc, "srt")
                return CandidateSyncInput(
                    candidate_id=str(cand.subtitle_id or cand.id),
                    subtitle_content=rendered.content,
                    metadata_score=cand.ranking_score,
                    metadata=cand,
                )

            coros = [_prepare_candidate(c) for c in top_candidates]
            results = await self._gather_safe(coros)
            for idx, item in enumerate(results):
                if isinstance(item, CandidateSyncInput):
                    candidate_inputs.append(item)
                elif isinstance(item, BaseException):
                    if idx == 0 and len(top_candidates) == 1 and isinstance(item, AppError):
                        raise item
                    logger.warning("Skipping candidate %s due to fetch/parse error: %s", top_candidates[idx].subtitle_id, item)

            if not candidate_inputs:
                raise AppError("Failed to download any valid subtitle candidates from OpenSubtitles.", status_code=502)

        # 3. Synchronize
        if len(candidate_inputs) == 1:
            sync_result: SyncResult = await self.sync_engine.synchronize_subtitle(
                subtitle_content=candidate_inputs[0].subtitle_content,
                stream_url=req.stream_url,
                stream_headers=req.stream_headers,
                target_format=target_format,
                language=clean_lang,
                media_duration=req.media_duration,
            )
            sync_result.selected_candidate_id = candidate_inputs[0].candidate_id
            sync_result.candidates_evaluated = 1
        else:
            sync_result, _ = await self.sync_engine.synchronize_candidates(
                candidates_inputs=candidate_inputs,
                stream_url=req.stream_url,
                stream_headers=req.stream_headers,
                target_format=target_format,
                language=clean_lang,
                media_duration=req.media_duration,
            )

        # 4. Cache if successful
        if req.stream_url and not req.subtitle_content and sync_result.success:
            try:
                sync_cache_key = make_key(
                    "sync",
                    tq.cache_parts(),
                    clean_lang,
                    target_format,
                    req.hearing_impaired,
                    stream_fingerprint(req.stream_url),
                )
                await self.cache.set(
                    "sync",
                    sync_cache_key,
                    sync_result.subtitle_content.encode("utf-8"),
                    ttl_seconds=float(self.settings.SYNC_CACHE_TTL_HOURS) * 3600.0,
                    meta={
                        "format": sync_result.format,
                        "cue_count": sync_result.cue_count,
                        "filename": build_download_filename(req.query or req.imdb_id, clean_lang, sync_result.format),
                        "sync_status": sync_result.status,
                        "offset_seconds": sync_result.offset_seconds,
                        "audio_analyzed": sync_result.audio_analyzed,
                        "scale_factor": sync_result.scale_factor,
                        "alignment_method": sync_result.alignment_method,
                        "segments_count": sync_result.segments_count,
                        "candidates_evaluated": sync_result.candidates_evaluated,
                        "selected_candidate_id": sync_result.selected_candidate_id,
                        "confidence_score": sync_result.confidence_score,
                    },
                )
            except Exception as exc:
                logger.warning("Failed to store synchronized subtitle in cache: %s", exc)

        return SyncResponse(
            success=sync_result.success,
            sync_status=sync_result.status,
            language=clean_lang,
            format=sync_result.format,
            audio_analyzed=sync_result.audio_analyzed,
            offset_seconds=sync_result.offset_seconds,
            scale_factor=sync_result.scale_factor,
            alignment_method=sync_result.alignment_method,
            segments_count=sync_result.segments_count,
            candidates_evaluated=sync_result.candidates_evaluated,
            selected_candidate_id=sync_result.selected_candidate_id,
            confidence_score=sync_result.confidence_score,
            processing_time_seconds=sync_result.processing_time_seconds,
            cues_count=sync_result.cue_count,
            subtitle_content=sync_result.subtitle_content,
            detail=sync_result.detail,
        )

    async def _gather_safe(self, coros: list[Any]) -> list[Any]:
        async def _wrap(coro: Any) -> Any:
            try:
                return await coro
            except BaseException as exc:
                return exc

        return await gather_limited(self.settings.MAX_CONCURRENT_REQUESTS, [_wrap(c) for c in coros])


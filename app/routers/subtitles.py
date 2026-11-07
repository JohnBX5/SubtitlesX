"""Subtitle discovery, download, and audio synchronization router for Cinema Hub."""

import logging
import re
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.subtitle import LanguagesResponse, MovieInfo, SyncRequest, SyncResponse
from app.services.cache_manager import cache_manager
from app.services.opensubtitles import OpenSubtitlesAPIError, opensubtitles_client
from app.services.ranking import select_best_candidate, select_top_candidates
from app.services.subtitle_manager import (
    SubtitleConversionError,
    SubtitleValidationError,
    subtitle_manager,
)
from app.services.sync_service import CandidateSyncInput, SyncResult, subtitle_sync_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Subtitles"])

# ISO language code regex (e.g. 'en', 'ar', 'pt-br', 'zh-cn')
LANGUAGE_CODE_REGEX = re.compile(r"^[a-zA-Z]{2,3}(-[a-zA-Z0-9]{2,4})?$")


def validate_title_identification(
    imdb_id: Optional[str],
    query: Optional[str],
) -> None:
    """Ensure at least one valid identification method is supplied."""
    if not (imdb_id and imdb_id.strip()) and not (query and query.strip()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one valid identification method is required: 'imdb_id' or 'query'.",
        )


@router.get(
    "/languages",
    response_model=LanguagesResponse,
    summary="Discover available subtitle languages",
)
async def discover_languages(
    imdb_id: Optional[str] = Query(None, description="IMDb identifier (e.g. tt0111161 or 0111161)"),
    query: Optional[str] = Query(None, description="Movie or episode title"),
    year: Optional[int] = Query(None, ge=1880, le=2100, description="Release year"),
    type: Optional[str] = Query(None, pattern="^(movie|episode|all)$", description="Media type"),
    season_number: Optional[int] = Query(None, ge=1, description="TV show season number"),
    episode_number: Optional[int] = Query(None, ge=1, description="TV show episode number"),
) -> LanguagesResponse:
    """Retrieve all languages that have verified subtitle files for the given movie or show."""
    validate_title_identification(imdb_id, query)

    try:
        movie_info, languages = await opensubtitles_client.get_available_languages(
            imdb_id=imdb_id,
            query=query,
            year=year,
            media_type=type,
            season_number=season_number,
            episode_number=episode_number,
        )
    except OpenSubtitlesAPIError as exc:
        logger.error("OpenSubtitles API error in /languages: %s", exc)
        raise HTTPException(
            status_code=exc.status_code if exc.status_code in {429, 502, 503, 504} else 502,
            detail=str(exc),
        )

    return LanguagesResponse(
        success=True,
        movie=movie_info,
        languages=languages,
    )


@router.get(
    "/download",
    summary="Download the best matching subtitle file (optionally synchronized)",
    responses={
        200: {
            "description": "The normalized subtitle file in UTF-8 encoding",
            "content": {
                "text/plain": {},
                "text/vtt": {},
            },
        },
        400: {"description": "Invalid query parameters"},
        404: {"description": "No subtitles found"},
        502: {"description": "OpenSubtitles service error"},
    },
)
async def download_subtitle(
    language: str = Query(..., description="ISO language code (e.g. 'ar', 'en')"),
    imdb_id: Optional[str] = Query(None, description="IMDb identifier (e.g. tt0111161)"),
    query: Optional[str] = Query(None, description="Movie or episode title"),
    year: Optional[int] = Query(None, ge=1880, le=2100, description="Release year"),
    type: Optional[str] = Query("movie", pattern="^(movie|episode|all)$", description="Media type"),
    season_number: Optional[int] = Query(None, ge=1, description="TV show season number"),
    episode_number: Optional[int] = Query(None, ge=1, description="TV show episode number"),
    format: Optional[str] = Query("srt", pattern="^(srt|vtt)$", description="Desired format (srt or vtt)"),
    hearing_impaired: bool = Query(False, description="Prefer hearing impaired (SDH) subtitles"),
    sync: bool = Query(False, description="Whether to automatically synchronize subtitle to stream audio"),
    stream_url: Optional[str] = Query(None, description="Optional playback stream URL for audio synchronization"),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Search, rank, download, validate, cache, and deliver the best matching subtitle file.

    When sync=true and stream_url is provided, automatically performs server-side audio
    synchronization and attaches X-Sync-* response headers.
    """
    validate_title_identification(imdb_id, query)

    # Validate language code format
    clean_lang = language.strip().lower()
    if not LANGUAGE_CODE_REGEX.match(clean_lang):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid language code '{language}'. Must be a valid 2-3 letter ISO code (e.g. 'en', 'ar').",
        )

    target_format = (format or "srt").lower()

    # 1. Determine cache key
    if sync and stream_url:
        cache_key = cache_manager.build_sync_cache_key(
            imdb_id=imdb_id,
            query=query,
            year=year,
            language=clean_lang,
            format_=target_format,
            media_type=type,
            season_number=season_number,
            episode_number=episode_number,
            stream_url=stream_url,
        )
    else:
        cache_key = cache_manager.build_cache_key(
            imdb_id=imdb_id,
            query=query,
            year=year,
            language=clean_lang,
            format_=target_format,
            media_type=type,
            season_number=season_number,
            episode_number=episode_number,
        )

    # 2. Acquire per-key lock to prevent simultaneous duplicate processing
    key_lock = await cache_manager.get_key_lock(cache_key)

    async with key_lock:
        # Check cache inside the lock
        cached_entry = await cache_manager.get_cached_subtitle(cache_key, db)
        if cached_entry is not None:
            record, file_bytes = cached_entry
            mime_type = "text/vtt; charset=utf-8" if record.format == "vtt" else "text/plain; charset=utf-8"
            safe_dl_filename = record.original_filename or f"{cache_key}.{record.format}"
            encoded_fn = quote(safe_dl_filename)

            headers = {
                "Content-Disposition": f'attachment; filename="{safe_dl_filename}"; filename*=UTF-8\'\'{encoded_fn}',
                "X-Cache": "HIT",
                "X-Subtitle-Format": record.format,
                "X-Subtitle-Cue-Count": str(record.cue_count),
                "Cache-Control": "public, max-age=86400",
            }
            if sync and stream_url:
                headers["X-Sync-Status"] = "synchronized"
                headers["X-Audio-Analyzed"] = "true"

            return Response(content=file_bytes, media_type=mime_type, headers=headers)

        # 3. Base Subtitle Retrieval (Check Base Cache or Query OpenSubtitles)
        base_cache_key = cache_manager.build_cache_key(
            imdb_id=imdb_id,
            query=query,
            year=year,
            language=clean_lang,
            format_="srt",
            media_type=type,
            season_number=season_number,
            episode_number=episode_number,
        )

        base_sub_bytes: Optional[bytes] = None
        base_cached = await cache_manager.get_cached_subtitle(base_cache_key, db)
        movie_title = query or ""
        best_candidate = None
        file_id = None

        if base_cached is not None:
            _, base_sub_bytes = base_cached
        else:
            try:
                movie_info, candidates = await opensubtitles_client.search_subtitles(
                    imdb_id=imdb_id,
                    query=query,
                    year=year,
                    media_type=type,
                    languages=clean_lang,
                    season_number=season_number,
                    episode_number=episode_number,
                )
            except OpenSubtitlesAPIError as exc:
                logger.error("OpenSubtitles API error during subtitle search: %s", exc)
                raise HTTPException(
                    status_code=exc.status_code if exc.status_code in {429, 502, 503, 504} else 502,
                    detail=str(exc),
                )

            if not candidates:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No subtitles found for the requested title in language '{clean_lang}'.",
                )

            if movie_info and movie_info.title:
                movie_title = movie_info.title

            best_candidate = select_best_candidate(
                candidates,
                query_hint=movie_title,
                prefer_hearing_impaired=hearing_impaired,
            )

            if not best_candidate or not best_candidate.files:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No suitable subtitle files available for download.",
                )

            target_file = best_candidate.files[0]
            file_id = target_file.file_id

            try:
                download_url, remote_filename = await opensubtitles_client.request_download_link(file_id)
                base_sub_bytes = await opensubtitles_client.fetch_file_bytes(download_url)
            except OpenSubtitlesAPIError as exc:
                logger.error("Failed to acquire OpenSubtitles file %s: %s", file_id, exc)
                raise HTTPException(
                    status_code=exc.status_code if exc.status_code in {429, 502, 503, 504} else 502,
                    detail=str(exc),
                )

        # 4. Process base subtitle
        processed = subtitle_manager.process_subtitle(
            raw_bytes=base_sub_bytes,
            filename_hint="base.srt",
            target_format=target_format,
        )

        sync_status = "none"
        sync_offset = 0.0
        audio_analyzed = False

        # 5. Audio Synchronization (if requested)
        if sync and stream_url:
            sync_res = await subtitle_sync_service.synchronize_subtitle(
                subtitle_content=processed.content_str,
                stream_url=stream_url,
                target_format=target_format,
                language=clean_lang,
            )
            sync_status = sync_res.status
            sync_offset = sync_res.offset_seconds or 0.0
            audio_analyzed = sync_res.audio_analyzed

            # Re-wrap processed subtitle if synchronized or aligned
            processed = subtitle_manager.process_subtitle(
                raw_bytes=sync_res.subtitle_content.encode("utf-8"),
                filename_hint=f"synced.{target_format}",
                target_format=target_format,
            )

        # 6. Generate safe filename
        title_slug = "subtitle"
        if movie_title:
            title_slug = re.sub(r"[^A-Za-z0-9]+", ".", movie_title.strip())

        safe_dl_filename = f"{title_slug}.{clean_lang}.{processed.format}"
        safe_dl_filename = subtitle_manager.sanitize_filename(safe_dl_filename, fallback_ext=processed.format)

        # 7. Store into persistent SQLite and disk cache
        try:
            await cache_manager.store_subtitle(
                cache_key=cache_key,
                processed=processed,
                language=clean_lang,
                session=db,
                media_type=type,
                imdb_id=imdb_id,
                query=query,
                year=year,
                season_number=season_number,
                episode_number=episode_number,
                subtitle_id=best_candidate.subtitle_id if best_candidate else None,
                file_id=file_id,
                original_filename=safe_dl_filename,
            )
        except Exception as exc:
            logger.warning("Failed to store subtitle in persistent cache: %s", exc)

        encoded_fn = quote(safe_dl_filename)
        resp_headers = {
            "Content-Disposition": f'attachment; filename="{safe_dl_filename}"; filename*=UTF-8\'\'{encoded_fn}',
            "X-Cache": "MISS",
            "X-Subtitle-Format": processed.format,
            "X-Subtitle-Cue-Count": str(processed.cue_count),
            "Cache-Control": "public, max-age=86400",
        }
        if sync and stream_url:
            resp_headers["X-Sync-Status"] = sync_status
            resp_headers["X-Sync-Offset"] = f"{sync_offset:.3f}"
            resp_headers["X-Audio-Analyzed"] = str(audio_analyzed).lower()

        return Response(
            content=processed.content_bytes,
            media_type=processed.media_type_header,
            headers=resp_headers,
        )


@router.post(
    "/synchronize",
    response_model=SyncResponse,
    summary="Synchronize subtitle with video audio",
    responses={
        200: {"description": "Structured synchronization result with aligned or fallback subtitle"},
        400: {"description": "Invalid input parameters"},
        404: {"description": "Subtitle not found"},
    },
)
async def synchronize_subtitle_endpoint(
    req: SyncRequest,
    db: AsyncSession = Depends(get_db),
) -> SyncResponse:
    """Analyze video stream audio, compute timing corrections, and return synchronized subtitle content."""
    # Ensure at least title/IMDb or direct subtitle_content is provided
    if not req.subtitle_content and not req.imdb_id and not req.query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one subtitle source is required: 'subtitle_content', 'imdb_id', or 'query'.",
        )

    clean_lang = req.language.strip().lower()
    if not LANGUAGE_CODE_REGEX.match(clean_lang):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid language code '{req.language}'.",
        )

    target_format = (req.format or "srt").lower()

    # 1. Check sync cache if stream_url is present and not user-supplied raw text
    if req.stream_url and not req.subtitle_content:
        sync_cache_key = cache_manager.build_sync_cache_key(
            imdb_id=req.imdb_id,
            query=req.query,
            year=req.year,
            language=clean_lang,
            format_=target_format,
            media_type=req.media_type,
            season_number=req.season_number,
            episode_number=req.episode_number,
            stream_url=req.stream_url,
        )

        cached_entry = await cache_manager.get_cached_subtitle(sync_cache_key, db)
        if cached_entry is not None:
            record, file_bytes = cached_entry
            content_str = file_bytes.decode("utf-8", errors="replace")
            logger.info("Serving synchronized subtitle from cache (key=%s)", sync_cache_key)
            return SyncResponse(
                success=True,
                sync_status="synchronized",
                language=clean_lang,
                format=record.format,
                audio_analyzed=True,
                offset_seconds=0.0,
                confidence_score=1.0,
                processing_time_seconds=0.005,
                cues_count=record.cue_count,
                subtitle_content=content_str,
                detail="Retrieved from persistent synchronization cache.",
            )

    # 2. Retrieve base subtitle content (single or multi-candidate acoustic selection)
    candidate_inputs: list[CandidateSyncInput] = []

    if req.subtitle_content:
        candidate_inputs.append(
            CandidateSyncInput(
                candidate_id="custom_supplied",
                subtitle_content=req.subtitle_content,
                metadata_score=100.0,
            )
        )
    else:
        base_cache_key = cache_manager.build_cache_key(
            imdb_id=req.imdb_id,
            query=req.query,
            year=req.year,
            language=clean_lang,
            format_="srt",
            media_type=req.media_type,
            season_number=req.season_number,
            episode_number=req.episode_number,
        )

        base_cached = await cache_manager.get_cached_subtitle(base_cache_key, db)
        if base_cached is not None and not req.stream_url:
            record, file_bytes = base_cached
            candidate_inputs.append(
                CandidateSyncInput(
                    candidate_id=record.subtitle_id or "cached_base",
                    subtitle_content=file_bytes.decode("utf-8", errors="replace"),
                    metadata_score=100.0,
                )
            )
        else:
            try:
                movie_info, candidates = await opensubtitles_client.search_subtitles(
                    imdb_id=req.imdb_id,
                    query=req.query,
                    year=req.year,
                    media_type=req.media_type,
                    languages=clean_lang,
                    season_number=req.season_number,
                    episode_number=req.episode_number,
                )
            except OpenSubtitlesAPIError as exc:
                logger.error("OpenSubtitles API error during sync search: %s", exc)
                raise HTTPException(
                    status_code=exc.status_code if exc.status_code in {429, 502, 503, 504} else 502,
                    detail=str(exc),
                )

            if not candidates:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No subtitles found for requested title in language '{clean_lang}'.",
                )

            max_cands = 3 if req.stream_url else 1
            top_candidates = select_top_candidates(
                candidates,
                query_hint=req.query or (movie_info.title if movie_info else None),
                prefer_hearing_impaired=req.hearing_impaired,
                max_candidates=max_cands,
            )

            if not top_candidates:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No suitable subtitle files available for download.",
                )

            for idx, cand in enumerate(top_candidates):
                try:
                    download_url, _ = await opensubtitles_client.request_download_link(cand.files[0].file_id)
                    raw_bytes = await opensubtitles_client.fetch_file_bytes(download_url)
                    processed_base = subtitle_manager.process_subtitle(raw_bytes, target_format="srt")
                    candidate_inputs.append(
                        CandidateSyncInput(
                            candidate_id=str(cand.subtitle_id or cand.id),
                            subtitle_content=processed_base.content_str,
                            metadata_score=cand.ranking_score,
                            metadata=cand,
                        )
                    )
                except Exception as exc:
                    if idx == 0 and len(top_candidates) == 1:
                        if isinstance(exc, OpenSubtitlesAPIError):
                            raise HTTPException(
                                status_code=exc.status_code if exc.status_code in {429, 502, 503, 504} else 502,
                                detail=str(exc),
                            )
                        raise
                    logger.warning("Skipping candidate %s due to fetch/parse error: %s", cand.subtitle_id, exc)

            if not candidate_inputs:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Failed to download any valid subtitle candidates from OpenSubtitles.",
                )

    # 3. Execute audio synchronization (single or multi-candidate acoustic selection)
    if len(candidate_inputs) == 1:
        sync_result: SyncResult = await subtitle_sync_service.synchronize_subtitle(
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
        sync_result, _ = await subtitle_sync_service.synchronize_candidates(
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
            sync_cache_key = cache_manager.build_sync_cache_key(
                imdb_id=req.imdb_id,
                query=req.query,
                year=req.year,
                language=clean_lang,
                format_=target_format,
                media_type=req.media_type,
                season_number=req.season_number,
                episode_number=req.episode_number,
                stream_url=req.stream_url,
            )
            processed_sync = subtitle_manager.process_subtitle(
                sync_result.subtitle_content.encode("utf-8"),
                filename_hint=f"synced.{target_format}",
                target_format=target_format,
            )
            await cache_manager.store_subtitle(
                cache_key=sync_cache_key,
                processed=processed_sync,
                language=clean_lang,
                session=db,
                media_type=req.media_type,
                imdb_id=req.imdb_id,
                query=req.query,
                year=req.year,
                season_number=req.season_number,
                episode_number=req.episode_number,
                original_filename=f"synced.{target_format}",
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

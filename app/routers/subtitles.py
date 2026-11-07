"""Subtitle discovery and download router for Cinema Hub."""

import logging
import re
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.subtitle import LanguagesResponse, MovieInfo
from app.services.cache_manager import cache_manager
from app.services.opensubtitles import OpenSubtitlesAPIError, opensubtitles_client
from app.services.ranking import select_best_candidate
from app.services.subtitle_manager import (
    SubtitleConversionError,
    SubtitleValidationError,
    subtitle_manager,
)

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
    summary="Download the best matching subtitle file",
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
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Search, rank, download, validate, cache, and deliver the best matching subtitle file."""
    validate_title_identification(imdb_id, query)

    # Validate language code format
    clean_lang = language.strip().lower()
    if not LANGUAGE_CODE_REGEX.match(clean_lang):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid language code '{language}'. Must be a valid 2-3 letter ISO code (e.g. 'en', 'ar').",
        )

    target_format = (format or "srt").lower()

    # 1. Generate cache key
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

    # 2. Acquire per-key lock to prevent simultaneous duplicate downloads
    key_lock = await cache_manager.get_key_lock(cache_key)

    async with key_lock:
        # Check cache inside the lock
        cached_entry = await cache_manager.get_cached_subtitle(cache_key, db)
        if cached_entry is not None:
            record, file_bytes = cached_entry
            mime_type = "text/vtt; charset=utf-8" if record.format == "vtt" else "text/plain; charset=utf-8"
            safe_dl_filename = record.original_filename or f"{cache_key}.{record.format}"
            encoded_fn = quote(safe_dl_filename)

            return Response(
                content=file_bytes,
                media_type=mime_type,
                headers={
                    "Content-Disposition": f'attachment; filename="{safe_dl_filename}"; filename*=UTF-8\'\'{encoded_fn}',
                    "X-Cache": "HIT",
                    "X-Subtitle-Format": record.format,
                    "X-Subtitle-Cue-Count": str(record.cue_count),
                    "Cache-Control": "public, max-age=86400",
                },
            )

        # 3. Cache MISS: Query OpenSubtitles for subtitles matching the requested language
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

        # 4. Rank candidates deterministically
        best_candidate = select_best_candidate(
            candidates,
            query_hint=query or (movie_info.title if movie_info else None),
            prefer_hearing_impaired=hearing_impaired,
        )

        if not best_candidate or not best_candidate.files:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No suitable subtitle files available for download.",
            )

        target_file = best_candidate.files[0]
        file_id = target_file.file_id

        # 5. Request official download link
        try:
            download_url, remote_filename = await opensubtitles_client.request_download_link(file_id)
        except OpenSubtitlesAPIError as exc:
            logger.error("Failed to acquire OpenSubtitles download link for file %d: %s", file_id, exc)
            raise HTTPException(
                status_code=exc.status_code if exc.status_code in {429, 502, 503, 504} else 502,
                detail=str(exc),
            )

        # 6. Fetch raw subtitle bytes
        try:
            raw_bytes = await opensubtitles_client.fetch_file_bytes(download_url)
        except OpenSubtitlesAPIError as exc:
            logger.error("Failed to download subtitle file bytes from %s: %s", download_url, exc)
            raise HTTPException(
                status_code=exc.status_code if exc.status_code in {429, 502, 503, 504} else 502,
                detail="Failed to retrieve subtitle file from storage provider.",
            )

        # 7. Validate, normalize UTF-8, and convert format if requested
        try:
            processed = subtitle_manager.process_subtitle(
                raw_bytes=raw_bytes,
                filename_hint=remote_filename,
                target_format=target_format,
            )
        except (SubtitleValidationError, SubtitleConversionError) as exc:
            logger.error("Subtitle processing validation failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Downloaded subtitle failed validation: {exc}",
            )

        # 8. Generate safe filename
        title_slug = "subtitle"
        if movie_info and movie_info.title:
            title_slug = re.sub(r"[^A-Za-z0-9]+", ".", movie_info.title.strip())
        elif query:
            title_slug = re.sub(r"[^A-Za-z0-9]+", ".", query.strip())

        safe_dl_filename = f"{title_slug}.{clean_lang}.{processed.format}"
        safe_dl_filename = subtitle_manager.sanitize_filename(safe_dl_filename, fallback_ext=processed.format)

        # 9. Store into persistent SQLite and disk cache
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
                subtitle_id=best_candidate.subtitle_id,
                file_id=file_id,
                original_filename=safe_dl_filename,
            )
        except Exception as exc:
            logger.warning("Failed to store subtitle in persistent cache: %s", exc)

        encoded_fn = quote(safe_dl_filename)

        return Response(
            content=processed.content_bytes,
            media_type=processed.media_type_header,
            headers={
                "Content-Disposition": f'attachment; filename="{safe_dl_filename}"; filename*=UTF-8\'\'{encoded_fn}',
                "X-Cache": "MISS",
                "X-Subtitle-Format": processed.format,
                "X-Subtitle-Cue-Count": str(processed.cue_count),
                "X-Ranking-Score": str(best_candidate.ranking_score),
                "Cache-Control": "public, max-age=86400",
            },
        )


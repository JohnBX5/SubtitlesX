"""Subtitle discovery, download, and audio synchronization router for Cinema Hub."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query, Request, Response

from app.schemas.subtitle import LanguagesResponse, SyncRequest, SyncResponse
from app.service import SubtitleBackendService

router = APIRouter(tags=["Subtitles"])


def get_service(request: Request) -> SubtitleBackendService:
    return request.app.state.service


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    return request.client.host if request.client else "unknown"


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
    service: SubtitleBackendService = Depends(get_service),
) -> LanguagesResponse:
    return await service.discover_languages(
        imdb_id=imdb_id,
        query=query,
        year=year,
        media_type=type,
        season_number=season_number,
        episode_number=episode_number,
    )


@router.get(
    "/download",
    summary="Download the best matching subtitle file (optionally synchronized)",
    responses={
        200: {
            "description": "The normalized subtitle file in UTF-8 encoding",
            "content": {"text/plain": {}, "text/vtt": {}},
        },
        400: {"description": "Invalid query parameters"},
        404: {"description": "No subtitles found"},
        502: {"description": "OpenSubtitles service error"},
    },
)
async def download_subtitle(
    request: Request,
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
    service: SubtitleBackendService = Depends(get_service),
) -> Response:
    return await service.download_subtitle(
        language=language,
        imdb_id=imdb_id,
        query=query,
        year=year,
        media_type=type,
        season_number=season_number,
        episode_number=episode_number,
        target_format=(format or "srt").lower(),
        hearing_impaired=hearing_impaired,
        sync=sync,
        stream_url=stream_url,
        client_ip=_client_ip(request),
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
    request: Request,
    req: SyncRequest,
    service: SubtitleBackendService = Depends(get_service),
) -> SyncResponse:
    return await service.synchronize_subtitle(req, client_ip=_client_ip(request))


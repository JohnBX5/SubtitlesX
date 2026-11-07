"""Pydantic schemas for request validation and API responses."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check endpoint response schema."""

    status: str = Field(..., description="Service status indicator", examples=["healthy"])
    service: str = Field(..., description="Service identifier", examples=["cinema-hub-subtitles"])
    version: str = Field(..., description="Service version string", examples=["1.0.0"])


class MovieInfo(BaseModel):
    """Movie or show metadata associated with subtitle results."""

    imdb_id: Optional[str] = Field(None, description="IMDb identifier (e.g. tt0111161)")
    title: Optional[str] = Field(None, description="Movie or show title")


class LanguageInfo(BaseModel):
    """Language item with available subtitle file count."""

    language_code: str = Field(..., description="ISO 639-1 / 639-2 language code", examples=["en", "ar"])
    language_name: str = Field(..., description="Human-readable language name", examples=["English", "Arabic"])
    available_files: int = Field(..., description="Number of available subtitle files", examples=[25])


class LanguagesResponse(BaseModel):
    """Response schema for the /api/subtitles/languages discovery endpoint."""

    success: bool = Field(True, description="Indicates request success")
    movie: Optional[MovieInfo] = Field(None, description="Associated movie or show details")
    languages: List[LanguageInfo] = Field(default_factory=list, description="List of available languages")


class SubtitleFileInfo(BaseModel):
    """Internal representation of a subtitle file from OpenSubtitles."""

    file_id: int
    cd_number: Optional[int] = 1
    file_name: str


class SubtitleCandidate(BaseModel):
    """Internal model representing an OpenSubtitles search result candidate for ranking."""

    id: str
    subtitle_id: str
    language: str
    download_count: int = 0
    new_download_count: int = 0
    hearing_impaired: bool = False
    hd: bool = False
    fps: Optional[float] = None
    votes: int = 0
    ratings: float = 0.0
    from_trusted: bool = False
    foreign_parts_only: bool = False
    ai_translated: bool = False
    machine_translated: bool = False
    release: Optional[str] = None
    files: List[SubtitleFileInfo] = Field(default_factory=list)
    ranking_score: float = 0.0
    ranking_reason: str = ""


class SyncRequest(BaseModel):
    """Request payload for subtitle audio synchronization."""

    language: str = Field(..., description="ISO 639-1 / 639-2 language code (e.g. 'en', 'ar')")
    imdb_id: Optional[str] = Field(None, description="IMDb identifier (e.g. tt0111161)")
    query: Optional[str] = Field(None, description="Movie or show title")
    year: Optional[int] = Field(None, description="Release year")
    media_type: Optional[str] = Field("movie", description="Media type ('movie' or 'episode')")
    season_number: Optional[int] = Field(None, description="TV season number")
    episode_number: Optional[int] = Field(None, description="TV episode number")
    format: Optional[str] = Field("srt", description="Target subtitle format ('srt' or 'vtt')")
    stream_url: Optional[str] = Field(None, description="Playback video/audio stream URL (MP4, HLS m3u8, DASH mpd)")
    stream_headers: Optional[Dict[str, str]] = Field(None, description="HTTP request headers (Referer, User-Agent, etc.)")
    media_duration: Optional[float] = Field(None, description="Total media duration in seconds")
    subtitle_content: Optional[str] = Field(None, description="Optional raw subtitle text if syncing user-supplied subtitle")
    hearing_impaired: bool = Field(False, description="Prefer hearing impaired / SDH subtitles")


class SyncResponse(BaseModel):
    """Response payload for subtitle audio synchronization."""

    success: bool = Field(True, description="Whether request was handled")
    sync_status: str = Field(
        ...,
        description="Sync status: 'synchronized', 'already_aligned', 'partial', 'unavailable', 'unsupported', or 'failed'",
    )
    language: str = Field(..., description="Language code")
    format: str = Field(..., description="Subtitle format ('srt' or 'vtt')")
    audio_analyzed: bool = Field(False, description="Whether audio was accessed and analyzed")
    offset_seconds: Optional[float] = Field(None, description="Applied timing correction in seconds")
    scale_factor: float = Field(1.0, description="Applied framerate drift scale factor (1.0 = no drift)")
    alignment_method: str = Field(
        "global_fft",
        description="Alignment algorithm used: 'global_fft', 'drift_fft', or 'piecewise_split'",
    )
    segments_count: int = Field(1, description="Number of distinct timing segments aligned")
    candidates_evaluated: int = Field(1, description="Number of subtitle candidates acoustically evaluated")
    selected_candidate_id: Optional[str] = Field(None, description="ID of the acoustically selected subtitle candidate")
    confidence_score: Optional[float] = Field(None, description="Synchronization confidence score (0.0 to 1.0)")
    processing_time_seconds: float = Field(..., description="Total processing time in seconds")
    cues_count: int = Field(0, description="Total dialogue cues in subtitle")
    subtitle_content: Optional[str] = Field(None, description="Synchronized or fallback UTF-8 subtitle text")
    download_url: Optional[str] = Field(None, description="Direct download URL for the synchronized file")
    detail: Optional[str] = Field(None, description="Human-readable processing notes")

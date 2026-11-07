"""Pydantic schemas for request validation and API responses."""

from typing import List, Optional
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


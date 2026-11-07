"""Services package."""

from app.services.audio_extractor import AudioExtractionError, audio_extractor
from app.services.authentication import OpenSubtitlesAuthError, auth_manager
from app.services.cache_manager import cache_manager
from app.services.opensubtitles import (
    OpenSubtitlesAPIError,
    opensubtitles_client,
)
from app.services.ranking import (
    calculate_candidate_score,
    rank_subtitle_candidates,
    select_best_candidate,
)
from app.services.ssrf_protector import SSRFValidationError, validate_stream_url
from app.services.subtitle_manager import (
    ProcessedSubtitle,
    SubtitleConversionError,
    SubtitleValidationError,
    subtitle_manager,
)
from app.services.sync_service import (
    SubtitleCue,
    SubtitleSynchronizationService,
    SyncResult,
    subtitle_sync_service,
)

__all__ = [
    "auth_manager",
    "OpenSubtitlesAuthError",
    "opensubtitles_client",
    "OpenSubtitlesAPIError",
    "subtitle_manager",
    "SubtitleValidationError",
    "SubtitleConversionError",
    "ProcessedSubtitle",
    "cache_manager",
    "calculate_candidate_score",
    "rank_subtitle_candidates",
    "select_best_candidate",
    "audio_extractor",
    "AudioExtractionError",
    "validate_stream_url",
    "SSRFValidationError",
    "subtitle_sync_service",
    "SubtitleSynchronizationService",
    "SubtitleCue",
    "SyncResult",
]

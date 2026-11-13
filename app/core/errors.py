"""Domain error hierarchy.

Services raise these errors; the HTTP layer maps them to status codes in a
single place (``app.main``). Messages are safe to show to clients.
"""

from __future__ import annotations


class AppError(Exception):
    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code


class BadRequestError(AppError):
    status_code = 400
    code = "bad_request"


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class UnauthorizedError(AppError):
    status_code = 401
    code = "unauthorized"


class RateLimitedError(AppError):
    status_code = 429
    code = "rate_limited"

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class UpstreamError(AppError):
    """OpenSubtitles (or another upstream) failed or misbehaved."""

    status_code = 502
    code = "upstream_error"


class ServiceUnavailableError(AppError):
    status_code = 503
    code = "service_unavailable"


class SubtitleValidationError(AppError):
    """Subtitle payload is not a valid/safe subtitle document."""

    status_code = 422
    code = "invalid_subtitle"


class AudioExtractionError(AppError):
    """Audio could not be obtained from the media stream.

    ``unsupported`` distinguishes permanent conditions (DRM, no audio track,
    unknown container) from transient ones (network, timeouts).
    """

    status_code = 502
    code = "audio_unavailable"

    def __init__(self, message: str, *, unsupported: bool = False) -> None:
        super().__init__(message)
        self.unsupported = unsupported


class StreamURLRejected(AudioExtractionError):
    """Stream URL failed SSRF validation."""

    def __init__(self, message: str) -> None:
        super().__init__(message, unsupported=True)


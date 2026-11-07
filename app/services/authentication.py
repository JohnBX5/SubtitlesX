"""Authentication service for OpenSubtitles REST API v1.

Manages Bearer Tokens in memory, thread/task-safely prevents duplicate logins,
refreshes tokens before expiration, and ensures credentials are never logged.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


class OpenSubtitlesAuthError(Exception):
    """Raised when authentication with OpenSubtitles fails."""


class OpenSubtitlesAuthManager:
    """Manages OpenSubtitles authentication state, Bearer token caching, and renewals."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._user_info: Dict[str, Any] = {}
        # Refresh tokens 1 hour before 24-hour expiration (82800 seconds = 23 hours)
        self._token_lifetime_seconds: float = 23 * 3600

    @property
    def is_authenticated(self) -> bool:
        """Return True if a non-expired token is present in memory."""
        return bool(self._token and time.time() < self._token_expires_at)

    async def get_valid_token(self, client: httpx.AsyncClient) -> str:
        """Get an existing valid Bearer token or authenticate to obtain a new one.

        Uses an async lock to ensure multiple concurrent requests do not trigger
        redundant login calls.
        """
        # Fast path without lock
        if self.is_authenticated and self._token:
            return self._token

        async with self._lock:
            # Double-check inside lock
            if self.is_authenticated and self._token:
                return self._token

            token = await self._authenticate(client)
            return token

    async def invalidate_token(self) -> None:
        """Invalidate the current cached token, forcing re-authentication on the next call."""
        async with self._lock:
            self._token = None
            self._token_expires_at = 0.0
            self._user_info.clear()
            logger.info("OpenSubtitles token invalidated.")

    async def _authenticate(self, client: httpx.AsyncClient) -> str:
        """Perform the login request to OpenSubtitles REST API v1."""
        settings = get_settings()

        if not settings.OPENSUBTITLES_API_KEY:
            raise OpenSubtitlesAuthError(
                "OPENSUBTITLES_API_KEY is not configured in environment."
            )
        if not settings.OPENSUBTITLES_USERNAME or not settings.OPENSUBTITLES_PASSWORD:
            raise OpenSubtitlesAuthError(
                "OPENSUBTITLES_USERNAME or OPENSUBTITLES_PASSWORD is not configured in environment."
            )

        login_url = f"{settings.OPENSUBTITLES_BASE_URL.rstrip('/')}/login"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Api-Key": settings.OPENSUBTITLES_API_KEY,
            "User-Agent": settings.OPENSUBTITLES_USER_AGENT,
        }
        payload = {
            "username": settings.OPENSUBTITLES_USERNAME,
            "password": settings.OPENSUBTITLES_PASSWORD,
        }

        logger.info("Authenticating with OpenSubtitles REST API v1...")

        try:
            response = await client.post(
                login_url,
                json=payload,
                headers=headers,
                timeout=settings.HTTP_TIMEOUT_SECONDS,
            )
        except httpx.TimeoutException as exc:
            logger.error("Authentication request timed out.")
            raise OpenSubtitlesAuthError("OpenSubtitles login request timed out.") from exc
        except httpx.RequestError as exc:
            logger.error("Authentication network error: %s", exc.__class__.__name__)
            raise OpenSubtitlesAuthError("Network error while authenticating with OpenSubtitles.") from exc

        if response.status_code == 401:
            logger.error("Authentication failed: invalid username or password (HTTP 401).")
            raise OpenSubtitlesAuthError("Invalid OpenSubtitles credentials.")
        elif response.status_code == 403:
            logger.error("Authentication failed: invalid API key or forbidden (HTTP 403).")
            raise OpenSubtitlesAuthError("OpenSubtitles API key is invalid or forbidden.")
        elif response.status_code == 429:
            logger.error("Authentication rate limit exceeded (HTTP 429).")
            raise OpenSubtitlesAuthError("OpenSubtitles rate limit exceeded. Please try again later.")
        elif response.status_code != 200:
            logger.error("Authentication failed with HTTP status %d.", response.status_code)
            raise OpenSubtitlesAuthError(
                f"OpenSubtitles login failed with status code {response.status_code}."
            )

        try:
            data = response.json()
        except Exception as exc:
            logger.error("Authentication response is not valid JSON.")
            raise OpenSubtitlesAuthError("Invalid JSON response from OpenSubtitles login endpoint.") from exc

        token = data.get("token")
        if not token or not isinstance(token, str):
            logger.error("Authentication response did not contain a valid token string.")
            raise OpenSubtitlesAuthError("OpenSubtitles login response missing token.")

        self._token = token
        self._token_expires_at = time.time() + self._token_lifetime_seconds
        self._user_info = data.get("user", {})

        logger.info(
            "Authentication successful. Token cached (user_id=%s, level=%s).",
            self._user_info.get("user_id", "unknown"),
            self._user_info.get("level", "user"),
        )
        return token


# Global singleton instance
auth_manager = OpenSubtitlesAuthManager()


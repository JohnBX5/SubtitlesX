"""Official OpenSubtitles REST API v1 Client.

Implements asynchronous search, authentication integration, deterministic candidate
extraction, quota management, bounded retries with exponential backoff, and rate-limit handling.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.config import get_settings
from app.schemas.subtitle import LanguageInfo, MovieInfo, SubtitleCandidate, SubtitleFileInfo
from app.services.authentication import OpenSubtitlesAuthError, auth_manager

logger = logging.getLogger(__name__)


# ISO 639-1 / 639-2 fallback lookup dictionary
ISO_LANGUAGE_NAMES: Dict[str, str] = {
    "ar": "Arabic",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "pt-br": "Portuguese (Brazil)",
    "ru": "Russian",
    "zh": "Chinese",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "ja": "Japanese",
    "ko": "Korean",
    "hi": "Hindi",
    "tr": "Turkish",
    "pl": "Polish",
    "nl": "Dutch",
    "sv": "Swedish",
    "no": "Norwegian",
    "da": "Danish",
    "fi": "Finnish",
    "el": "Greek",
    "he": "Hebrew",
    "id": "Indonesian",
    "ms": "Malay",
    "th": "Thai",
    "vi": "Vietnamese",
    "cs": "Czech",
    "hu": "Hungarian",
    "ro": "Romanian",
    "bg": "Bulgarian",
    "uk": "Ukrainian",
    "fa": "Persian",
    "ur": "Urdu",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "mr": "Marathi",
    "tl": "Tagalog",
    "hr": "Croatian",
    "sr": "Serbian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "et": "Estonian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "sq": "Albanian",
    "mk": "Macedonian",
    "bs": "Bosnian",
    "is": "Icelandic",
}


class OpenSubtitlesAPIError(Exception):
    """Base exception for OpenSubtitles API interactions."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class OpenSubtitlesClient:
    """Async client for OpenSubtitles REST API v1."""

    def __init__(self) -> None:
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._languages_cache: Dict[str, str] = {}
        self._languages_lock = asyncio.Lock()

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            settings = get_settings()
            self._semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)
        return self._semaphore

    def get_http_client(self) -> httpx.AsyncClient:
        """Get or create the shared AsyncClient with connection pooling."""
        if self._client is None or self._client.is_closed:
            settings = get_settings()
            limits = httpx.Limits(
                max_connections=settings.MAX_CONCURRENT_REQUESTS * 2,
                max_keepalive_connections=settings.MAX_CONCURRENT_REQUESTS,
            )
            self._client = httpx.AsyncClient(
                limits=limits,
                timeout=httpx.Timeout(settings.HTTP_TIMEOUT_SECONDS),
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client session."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
            logger.info("OpenSubtitles HTTP client closed.")

    def _get_default_headers(self) -> Dict[str, str]:
        settings = get_settings()
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Api-Key": settings.OPENSUBTITLES_API_KEY,
            "User-Agent": settings.OPENSUBTITLES_USER_AGENT,
        }

    async def _send_request_with_retry(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
    ) -> httpx.Response:
        """Execute HTTP request with exponential backoff on retryable status codes and rate limits."""
        client = self.get_http_client()
        sem = self._get_semaphore()

        async with sem:
            for attempt in range(1, max_retries + 1):
                try:
                    response = await client.request(
                        method=method,
                        url=url,
                        headers=headers,
                        params=params,
                        json=json_data,
                    )

                    # Success or non-retryable client error
                    if response.status_code in {200, 201, 204}:
                        return response

                    if response.status_code == 429:
                        # Rate limited: check Retry-After header
                        retry_after = 2.0
                        header_val = response.headers.get("Retry-After")
                        if header_val and header_val.isdigit():
                            retry_after = float(header_val)
                        else:
                            retry_after = min(2.0 ** attempt, 10.0)

                        logger.warning(
                            "OpenSubtitles 429 Rate Limit (attempt %d/%d). Sleeping %.1fs...",
                            attempt,
                            max_retries,
                            retry_after,
                        )
                        if attempt < max_retries:
                            await asyncio.sleep(retry_after)
                            continue
                        raise OpenSubtitlesAPIError(
                            "OpenSubtitles rate limit exceeded. Please wait before retrying.",
                            status_code=429,
                        )

                    if response.status_code in {500, 502, 503, 504}:
                        delay = min(1.0 * (2 ** (attempt - 1)), 6.0)
                        logger.warning(
                            "OpenSubtitles server error HTTP %d (attempt %d/%d). Retrying in %.1fs...",
                            response.status_code,
                            attempt,
                            max_retries,
                            delay,
                        )
                        if attempt < max_retries:
                            await asyncio.sleep(delay)
                            continue
                        raise OpenSubtitlesAPIError(
                            f"OpenSubtitles service unavailable (HTTP {response.status_code}).",
                            status_code=502,
                        )

                    # Non-retryable errors (400, 401, 403, 404)
                    return response

                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    delay = min(1.0 * (2 ** (attempt - 1)), 6.0)
                    logger.warning(
                        "Network error '%s' on %s %s (attempt %d/%d). Retrying in %.1fs...",
                        exc.__class__.__name__,
                        method,
                        url,
                        attempt,
                        max_retries,
                        delay,
                    )
                    if attempt < max_retries:
                        await asyncio.sleep(delay)
                        continue
                    raise OpenSubtitlesAPIError(
                        "Network timeout or error communicating with OpenSubtitles service.",
                        status_code=504,
                    ) from exc

            raise OpenSubtitlesAPIError("Max retries exceeded communicating with OpenSubtitles.")

    @staticmethod
    def normalize_imdb_id(imdb_id: Optional[str]) -> Optional[Any]:
        """Normalize IMDb identifier (accepts tt0111161 or 0111161 / 111161).

        OpenSubtitles accepts integer or string, but integers without 'tt' are most reliable.
        """
        if not imdb_id:
            return None
        cleaned = imdb_id.strip()
        if cleaned.lower().startswith("tt"):
            numeric = cleaned[2:]
            if numeric.isdigit():
                return int(numeric)
        elif cleaned.isdigit():
            return int(cleaned)
        return cleaned

    async def search_subtitles(
        self,
        imdb_id: Optional[str] = None,
        query: Optional[str] = None,
        year: Optional[int] = None,
        media_type: Optional[str] = None,
        languages: Optional[str] = None,
        season_number: Optional[int] = None,
        episode_number: Optional[int] = None,
    ) -> Tuple[Optional[MovieInfo], List[SubtitleCandidate]]:
        """Query OpenSubtitles GET /api/v1/subtitles.

        Returns:
            Tuple of (MovieInfo, List[SubtitleCandidate])
        """
        settings = get_settings()
        url = f"{settings.OPENSUBTITLES_BASE_URL.rstrip('/')}/subtitles"
        headers = self._get_default_headers()

        params: Dict[str, Any] = {}
        norm_id = self.normalize_imdb_id(imdb_id)
        if norm_id is not None:
            params["imdb_id"] = norm_id
        if query:
            params["query"] = query.strip()
        if year:
            params["year"] = year
        if media_type in {"movie", "episode", "all"}:
            params["type"] = media_type
        if languages:
            params["languages"] = languages.strip()
        if season_number is not None:
            params["season_number"] = season_number
        if episode_number is not None:
            params["episode_number"] = episode_number

        logger.info("Searching OpenSubtitles with params: %s", {k: v for k, v in params.items() if k != "query"})

        resp = await self._send_request_with_retry(
            method="GET",
            url=url,
            headers=headers,
            params=params,
        )

        if resp.status_code == 401 or resp.status_code == 403:
            logger.error("OpenSubtitles API key rejected (HTTP %d).", resp.status_code)
            raise OpenSubtitlesAPIError(
                "OpenSubtitles API key is unauthorized or invalid.",
                status_code=502,
            )
        elif resp.status_code != 200:
            logger.error("Search failed with HTTP %d: %s", resp.status_code, resp.text[:200])
            raise OpenSubtitlesAPIError(
                f"OpenSubtitles search request failed with status {resp.status_code}.",
                status_code=502,
            )

        try:
            body = resp.json()
        except Exception as exc:
            raise OpenSubtitlesAPIError("Invalid JSON returned by OpenSubtitles search.") from exc

        data = body.get("data", [])
        if not data:
            return None, []

        # Extract movie metadata from the first valid result's feature_details
        movie_info: Optional[MovieInfo] = None
        for item in data:
            attrs = item.get("attributes", {})
            feature = attrs.get("feature_details", {})
            if feature:
                title = feature.get("title") or feature.get("movie_name")
                f_imdb = feature.get("imdb_id")
                formatted_imdb = f"tt{f_imdb:07d}" if isinstance(f_imdb, int) else (f"tt{f_imdb}" if f_imdb else imdb_id)
                movie_info = MovieInfo(imdb_id=formatted_imdb, title=title)
                break

        if not movie_info and (imdb_id or query):
            movie_info = MovieInfo(imdb_id=imdb_id, title=query)

        candidates: List[SubtitleCandidate] = []
        for item in data:
            item_id = str(item.get("id", ""))
            attrs = item.get("attributes", {})
            lang = attrs.get("language", "").lower()
            if not lang:
                continue

            raw_files = attrs.get("files", [])
            files: List[SubtitleFileInfo] = []
            for f in raw_files:
                fid = f.get("file_id")
                if fid is not None:
                    files.append(
                        SubtitleFileInfo(
                            file_id=int(fid),
                            cd_number=f.get("cd_number", 1),
                            file_name=f.get("file_name", f"subtitle_{fid}.srt"),
                        )
                    )

            if not files:
                continue

            cand = SubtitleCandidate(
                id=item_id,
                subtitle_id=str(attrs.get("subtitle_id", item_id)),
                language=lang,
                download_count=attrs.get("download_count", 0) or 0,
                new_download_count=attrs.get("new_download_count", 0) or 0,
                hearing_impaired=bool(attrs.get("hearing_impaired", False)),
                hd=bool(attrs.get("hd", False)),
                fps=attrs.get("fps"),
                votes=attrs.get("votes", 0) or 0,
                ratings=float(attrs.get("ratings", 0.0) or 0.0),
                from_trusted=bool(attrs.get("from_trusted", False)),
                foreign_parts_only=bool(attrs.get("foreign_parts_only", False)),
                ai_translated=bool(attrs.get("ai_translated", False)),
                machine_translated=bool(attrs.get("machine_translated", False)),
                release=attrs.get("release"),
                files=files,
            )
            candidates.append(cand)

        return movie_info, candidates

    async def get_available_languages(
        self,
        imdb_id: Optional[str] = None,
        query: Optional[str] = None,
        year: Optional[int] = None,
        media_type: Optional[str] = None,
        season_number: Optional[int] = None,
        episode_number: Optional[int] = None,
    ) -> Tuple[Optional[MovieInfo], List[LanguageInfo]]:
        """Search OpenSubtitles and aggregate available language options with file counts."""
        movie_info, candidates = await self.search_subtitles(
            imdb_id=imdb_id,
            query=query,
            year=year,
            media_type=media_type,
            season_number=season_number,
            episode_number=episode_number,
        )

        if not candidates:
            return movie_info, []

        # Count available files per language
        lang_counts: Dict[str, int] = {}
        for cand in candidates:
            lang = cand.language.lower()
            file_count = len(cand.files)
            lang_counts[lang] = lang_counts.get(lang, 0) + file_count

        # Build language info list
        language_items: List[LanguageInfo] = []
        for code, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
            name = await self.resolve_language_name(code)
            language_items.append(
                LanguageInfo(
                    language_code=code,
                    language_name=name,
                    available_files=count,
                )
            )

        return movie_info, language_items

    async def resolve_language_name(self, code: str) -> str:
        """Resolve ISO language code to human-readable language name."""
        code_clean = code.strip().lower()

        # Check static ISO lookup first
        if code_clean in ISO_LANGUAGE_NAMES:
            return ISO_LANGUAGE_NAMES[code_clean]

        # Check runtime cache
        async with self._languages_lock:
            if code_clean in self._languages_cache:
                return self._languages_cache[code_clean]

        # Fallback to uppercase code or capital name
        return code_clean.upper()

    async def request_download_link(self, file_id: int) -> Tuple[str, str]:
        """Request official download link from OpenSubtitles POST /api/v1/download.

        Returns:
            Tuple of (download_link, remote_filename)
        """
        settings = get_settings()
        url = f"{settings.OPENSUBTITLES_BASE_URL.rstrip('/')}/download"
        client = self.get_http_client()

        # Download requires Bearer token
        token = await auth_manager.get_valid_token(client)
        headers = self._get_default_headers()
        headers["Authorization"] = f"Bearer {token}"

        payload = {"file_id": file_id, "sub_format": "srt"}

        logger.info("Requesting download link for file_id %d...", file_id)

        resp = await self._send_request_with_retry(
            method="POST",
            url=url,
            headers=headers,
            json_data=payload,
        )

        # Handle token expiration (HTTP 401)
        if resp.status_code == 401:
            logger.warning("Token expired during download request; re-authenticating and retrying once...")
            await auth_manager.invalidate_token()
            token = await auth_manager.get_valid_token(client)
            headers["Authorization"] = f"Bearer {token}"
            resp = await self._send_request_with_retry(
                method="POST",
                url=url,
                headers=headers,
                json_data=payload,
            )

        if resp.status_code != 200:
            logger.error("Download link request failed with HTTP %d: %s", resp.status_code, resp.text[:200])
            raise OpenSubtitlesAPIError(
                f"OpenSubtitles download request rejected (HTTP {resp.status_code}). Quota or permission error.",
                status_code=502,
            )

        try:
            data = resp.json()
        except Exception as exc:
            raise OpenSubtitlesAPIError("Invalid JSON returned by download endpoint.") from exc

        link = data.get("link")
        if not link:
            raise OpenSubtitlesAPIError("OpenSubtitles download response missing 'link' attribute.")

        file_name = data.get("file_name") or f"subtitle_{file_id}.srt"
        remaining = data.get("remaining")
        logger.info("Download link received for file %d (remaining quota: %s).", file_id, remaining)

        return link, file_name

    async def fetch_file_bytes(self, download_url: str) -> bytes:
        """Fetch raw subtitle file bytes from the temporary download link."""
        client = self.get_http_client()
        settings = get_settings()

        logger.info("Fetching subtitle bytes from download link...")

        try:
            resp = await client.get(
                download_url,
                timeout=settings.HTTP_TIMEOUT_SECONDS,
                headers={"User-Agent": settings.OPENSUBTITLES_USER_AGENT},
            )
        except httpx.TimeoutException as exc:
            raise OpenSubtitlesAPIError("Timeout while downloading subtitle file bytes.", status_code=504) from exc
        except httpx.RequestError as exc:
            raise OpenSubtitlesAPIError("Network error while downloading subtitle file bytes.", status_code=502) from exc

        if resp.status_code != 200:
            raise OpenSubtitlesAPIError(
                f"Failed to fetch subtitle file from link (HTTP {resp.status_code}).",
                status_code=502,
            )

        raw_bytes = resp.content
        if not raw_bytes:
            raise OpenSubtitlesAPIError("Received 0 bytes from subtitle download link.", status_code=502)

        return raw_bytes


# Global singleton instance
opensubtitles_client = OpenSubtitlesClient()


"""FastAPI application entry point for Cinema Hub Subtitle Backend."""

from __future__ import annotations

import asyncio
import hmac
import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.cache.store import CacheStore
from app.core.config import Settings, get_settings
from app.core.errors import AppError, RateLimitedError
from app.core.logging import configure_logging, request_id_var
from app.opensubtitles.client import OpenSubtitlesClient
from app.routers import health_router, subtitles_router
from app.service import SubtitleBackendService
from app.sync.engine import SyncEngine

logger = logging.getLogger("cinema_hub")

_PUBLIC_PATHS = frozenset({"/", "/api/subtitles/health", "/docs", "/redoc", "/openapi.json"})


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    cfg = settings or get_settings()
    configure_logging(level=cfg.LOG_LEVEL, fmt=cfg.LOG_FORMAT)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
        active_cfg = getattr(application.state, "settings", None) or get_settings()
        logger.info("Starting %s v%s...", active_cfg.APP_NAME, active_cfg.APP_VERSION)

        cache = CacheStore(
            active_cfg.resolved_db_path,
            memory_max_mb=active_cfg.MEMORY_CACHE_MAX_MB,
            max_size_mb=active_cfg.CACHE_MAX_SIZE_MB,
        )
        await cache.open()
        await cache.cleanup()

        os_client = OpenSubtitlesClient(active_cfg)
        sync_engine = SyncEngine(settings=active_cfg, cache=cache)
        service = SubtitleBackendService(
            settings=active_cfg,
            cache=cache,
            opensubtitles=os_client,
            sync_engine=sync_engine,
        )

        application.state.settings = active_cfg
        application.state.cache = cache
        application.state.opensubtitles = os_client
        application.state.sync_engine = sync_engine
        application.state.service = service

        async def _periodic_cleanup() -> None:
            interval = max(60.0, float(active_cfg.CACHE_CLEANUP_INTERVAL_MINUTES) * 60.0)
            while True:
                await asyncio.sleep(interval)
                await cache.cleanup()

        cleanup_task = asyncio.create_task(_periodic_cleanup())

        try:
            yield
        finally:
            logger.info("Shutting down %s...", active_cfg.APP_NAME)
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass
            sync_engine.close()
            await os_client.close()
            await cache.close()
            logger.info("Shutdown complete.")

    application = FastAPI(
        title=cfg.APP_NAME,
        version=cfg.APP_VERSION,
        description="Automated subtitle backend for Cinema Hub powered by OpenSubtitles REST API v1.",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    application.state.settings = cfg

    # ------------------------------------------------------------------ Middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    application.add_middleware(GZipMiddleware, minimum_size=cfg.GZIP_MIN_SIZE)

    @application.middleware("http")
    async def request_context_and_auth_middleware(request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        token = request_id_var.set(req_id)
        try:
            active_cfg: Settings = getattr(request.app.state, "settings", cfg)
            expected_key = active_cfg.API_ACCESS_KEY
            if expected_key and request.method != "OPTIONS" and request.url.path not in _PUBLIC_PATHS:
                provided_key = request.headers.get("X-API-Key") or ""
                if not hmac.compare_digest(provided_key, expected_key):
                    resp = JSONResponse(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        content={"detail": "Invalid or missing API key."},
                    )
                    resp.headers["X-Request-ID"] = req_id
                    return resp

            response = await call_next(request)
            response.headers["X-Request-ID"] = req_id
            return response
        finally:
            request_id_var.reset(token)

    # ------------------------------------------------------------------ Exception Handlers
    @application.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        headers = None
        if isinstance(exc, RateLimitedError) and exc.retry_after is not None:
            headers = {"Retry-After": str(max(1, int(round(exc.retry_after))))}
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message, "code": exc.code},
            headers=headers,
        )

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = []
        for err in exc.errors():
            field = " -> ".join(str(loc) for loc in err.get("loc", []))
            msg = err.get("msg", "Invalid value")
            errors.append(f"{field}: {msg}")
        return JSONResponse(
            status_code=422,
            content={"detail": "Request validation failed", "errors": errors},
        )

    @application.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=getattr(exc, "headers", None),
        )

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled server exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error. Please try again later."},
        )

    # ------------------------------------------------------------------ Routes
    application.include_router(health_router, prefix="/api/subtitles")
    application.include_router(subtitles_router, prefix="/api/subtitles")

    @application.get("/", include_in_schema=False)
    async def root_overview():
        active_cfg: Settings = getattr(application.state, "settings", cfg)
        return {
            "service": active_cfg.APP_NAME,
            "version": active_cfg.APP_VERSION,
            "health_check": "/api/subtitles/health",
            "documentation": "/docs",
        }

    return application


app = create_app()


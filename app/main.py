"""FastAPI application entry point for Cinema Hub Subtitle Backend."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.database import close_db, init_db
from app.routers import health_router, subtitles_router
from app.services.cache_manager import cache_manager
from app.services.opensubtitles import opensubtitles_client

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cinema_hub")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: manages startup database setup, cache verification, and graceful shutdown."""
    settings = get_settings()
    logger.info("Starting %s v%s...", settings.APP_NAME, settings.APP_VERSION)

    # 1. Ensure cache directory exists
    cache_dir = settings.resolved_cache_dir
    logger.info("Subtitle cache directory configured at: %s", cache_dir)

    # 2. Initialize database schema
    await init_db()

    # 3. Clean up expired cache items on startup
    try:
        cleaned = await cache_manager.cleanup_expired()
        if cleaned > 0:
            logger.info("Initial cache cleanup removed %d expired records.", cleaned)
    except Exception as exc:
        logger.warning("Initial cache cleanup encountered an issue: %s", exc)

    yield

    # Shutdown logic
    logger.info("Shutting down %s...", settings.APP_NAME)
    await opensubtitles_client.close()
    await close_db()
    logger.info("Shutdown complete.")


settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Automated subtitle backend for Cinema Hub powered by OpenSubtitles REST API v1.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# -----------------------------------------------------------------------------
# Security & Middleware
# -----------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Safe Error Handling
# -----------------------------------------------------------------------------


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle FastAPI request validation errors cleanly."""
    errors = []
    for err in exc.errors():
        field = " -> ".join(str(loc) for loc in err.get("loc", []))
        msg = err.get("msg", "Invalid value")
        errors.append(f"{field}: {msg}")

    return JSONResponse(
        status_code=422,
        content={"detail": "Request validation failed", "errors": errors},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle standard HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all unhandled exception handler ensuring no internal stack traces or secrets leak."""
    logger.error("Unhandled server exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Please try again later."},
    )


# -----------------------------------------------------------------------------
# Route Registration
# -----------------------------------------------------------------------------

# Main API router under /api/subtitles
app.include_router(health_router, prefix="/api/subtitles")
app.include_router(subtitles_router, prefix="/api/subtitles")


@app.get("/", include_in_schema=False)
async def root_redirect():
    """Root redirect to health check."""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "health_check": "/api/subtitles/health",
        "documentation": "/docs",
    }

"""API Routers package."""

from app.routers.health import router as health_router
from app.routers.subtitles import router as subtitles_router

__all__ = ["health_router", "subtitles_router"]


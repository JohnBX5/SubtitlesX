"""API routers package."""

from app.routers.health import router as health_router
from app.routers.subtitles import get_service, router as subtitles_router

__all__ = ["get_service", "health_router", "subtitles_router"]


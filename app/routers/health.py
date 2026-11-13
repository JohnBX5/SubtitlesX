"""Health check router."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.core.config import get_settings
from app.schemas.subtitle import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description="Returns service health status and version metadata for deployment monitoring.",
)
async def health_check(request: Request) -> HealthResponse:
    settings = getattr(request.app.state, "settings", None) or get_settings()
    return HealthResponse(
        status="healthy",
        service="cinema-hub-subtitles",
        version=settings.APP_VERSION,
    )


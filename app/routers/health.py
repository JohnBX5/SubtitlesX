"""Health check router."""

from fastapi import APIRouter
from app.config import get_settings
from app.schemas.subtitle import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Service health status")
async def health_check() -> HealthResponse:
    """Return backend operational status, service name, and version without exposing secrets."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        service="cinema-hub-subtitles",
        version=settings.APP_VERSION,
    )


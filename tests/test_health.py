"""Tests for health check and service status endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check_endpoint(async_client: AsyncClient):
    """Verify GET /api/subtitles/health returns healthy status, service name, and version."""
    response = await async_client.get("/api/subtitles/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "cinema-hub-subtitles"
    assert "version" in data
    # Ensure no internal secrets leak
    assert "OPENSUBTITLES_API_KEY" not in str(data)
    assert "password" not in str(data).lower()
    assert "token" not in str(data).lower()


@pytest.mark.asyncio
async def test_root_endpoint(async_client: AsyncClient):
    """Verify root GET / returns basic service overview and docs pointer."""
    response = await async_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["health_check"] == "/api/subtitles/health"


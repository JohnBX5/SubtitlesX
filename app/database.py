"""Database configuration and async session management with SQLAlchemy and SQLite."""

import logging
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

from app.config import get_settings

logger = logging.getLogger(__name__)

Base = declarative_base()

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Get or create the SQLAlchemy AsyncEngine instance."""
    global _engine
    if _engine is None:
        settings = get_settings()
        db_url = settings.DATABASE_URL

        # If using local SQLite file, ensure directory exists
        if db_url.startswith("sqlite+aiosqlite:///"):
            raw_path = db_url.replace("sqlite+aiosqlite:///", "")
            if raw_path and not raw_path.startswith(":memory:"):
                db_path = Path(raw_path).resolve()
                db_path.parent.mkdir(parents=True, exist_ok=True)

        _engine = create_async_engine(
            db_url,
            echo=False,
            future=True,
            connect_args={"check_same_thread": False} if "sqlite" in db_url else {},
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the async session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            expire_on_commit=False,
            class_=AsyncSession,
        )
    return _session_factory


async def init_db() -> None:
    """Initialize database tables on application startup."""
    engine = get_engine()
    async with engine.begin() as conn:
        # Import models here to ensure metadata is registered
        from app.models.cache import SubtitleCacheRecord  # noqa: F401
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized successfully.")


async def close_db() -> None:
    """Close and dispose the database engine on shutdown."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database connection closed.")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for obtaining an async database session."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


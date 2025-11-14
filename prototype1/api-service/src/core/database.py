"""
Database configuration and initialization.
"""

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector

from .config import get_settings

logger = structlog.get_logger(__name__)

# Database base class
Base = declarative_base()

# Global database engine and session
engine = None
async_session = None


async def init_db():
    """Initialize database connection and create tables."""
    global engine, async_session
    
    settings = get_settings()
    
    # Create async engine
    engine = create_async_engine(
        settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"),
        echo=settings.DATABASE_ECHO,
        pool_pre_ping=True,
        pool_recycle=300,
    )
    
    # Create async session factory
    async_session = sessionmaker(
        engine, 
        class_=AsyncSession, 
        expire_on_commit=False
    )
    
    # Create tables
    async with engine.begin() as conn:
        # Enable pgvector extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("Database initialized successfully")


async def get_db() -> AsyncSession:
    """Get database session dependency."""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()


async def close_db():
    """Close database connections."""
    global engine
    if engine:
        await engine.dispose()
        logger.info("Database connections closed")
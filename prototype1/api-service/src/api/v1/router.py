"""
Main API router for v1 endpoints.
"""

from fastapi import APIRouter

from .endpoints import songs, recommendations, mixing, health

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(songs.router, prefix="/songs", tags=["songs"])
api_router.include_router(recommendations.router, prefix="/recommendations", tags=["recommendations"])
api_router.include_router(mixing.router, prefix="/mixing", tags=["mixing"])
api_router.include_router(health.router, prefix="/health", tags=["health"])
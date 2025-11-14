"""
Health check endpoints.
"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def health_check():
    """API service health check."""
    return {"status": "healthy", "service": "api-service"}
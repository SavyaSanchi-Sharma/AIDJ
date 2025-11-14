"""
Mixing API endpoints - placeholder for now.
"""

from fastapi import APIRouter

router = APIRouter()

@router.post("/sessions")
async def create_mixing_session():
    """Create mixing session - placeholder."""
    return {"message": "Mixing session endpoint - coming soon"}
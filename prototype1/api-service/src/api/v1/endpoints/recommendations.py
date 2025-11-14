"""
Recommendations API endpoints - placeholder for now.
"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/similar/{song_id}")
async def get_similar_songs(song_id: str):
    """Get similar songs - placeholder."""
    return {"message": "Recommendations endpoint - coming soon"}
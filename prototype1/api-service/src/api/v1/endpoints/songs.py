"""
Songs API endpoints - upload, processing, and retrieval.
"""

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from uuid import UUID

from ....core.database import get_db
from ....models.song import Song, ProcessingStatus
from ....services.song_service import SongService
from ....schemas.song import SongResponse, SongDetailResponse, MusicalDNAResponse

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.post("/", response_model=SongResponse, status_code=status.HTTP_201_CREATED)
async def upload_song(
    file: UploadFile = File(...),
    title: str = Form(...),
    artist: str = Form(...),
    album: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a new song for processing.
    
    Accepts audio files (mp3, wav, flac) up to 100MB.
    Triggers asynchronous processing pipeline for stem separation and Musical DNA analysis.
    """
    logger.info("Song upload requested", filename=file.filename, title=title, artist=artist)
    
    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File name is required"
        )
    
    # Check file format
    allowed_formats = {".mp3", ".wav", ".flac", ".aac"}
    file_extension = file.filename.lower().split(".")[-1]
    if f".{file_extension}" not in allowed_formats:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file format. Allowed: {', '.join(allowed_formats)}"
        )
    
    try:
        song_service = SongService(db)
        song = await song_service.create_song(
            file=file,
            title=title,
            artist=artist,
            album=album
        )
        
        logger.info("Song uploaded successfully", song_id=song.id, title=title)
        return SongResponse.from_orm(song)
        
    except ValueError as e:
        logger.error("Song upload validation error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Song upload failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload song"
        )


@router.get("/", response_model=dict)
async def list_songs(
    limit: int = 20,
    offset: int = 0,
    key_signature: Optional[str] = None,
    bpm_min: Optional[float] = None,
    bpm_max: Optional[float] = None,
    processing_status: Optional[ProcessingStatus] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    List songs with filtering and pagination.
    
    Supports filtering by key signature, BPM range, and processing status.
    """
    try:
        song_service = SongService(db)
        result = await song_service.list_songs(
            limit=limit,
            offset=offset,
            key_signature=key_signature,
            bpm_min=bpm_min,
            bpm_max=bpm_max,
            processing_status=processing_status
        )
        
        return {
            "songs": [SongResponse.from_orm(song) for song in result["songs"]],
            "total": result["total"],
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error("Failed to list songs", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve songs"
        )


@router.get("/{song_id}", response_model=SongDetailResponse)
async def get_song(
    song_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed song information including Musical DNA and stems.
    """
    try:
        song_service = SongService(db)
        song = await song_service.get_song_by_id(song_id)
        
        if not song:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Song not found"
            )
        
        return SongDetailResponse.from_orm(song)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get song", song_id=song_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve song"
        )


@router.delete("/{song_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_song(
    song_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a song and all associated data.
    """
    try:
        song_service = SongService(db)
        success = await song_service.delete_song(song_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Song not found"
            )
        
        logger.info("Song deleted successfully", song_id=song_id)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete song", song_id=song_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete song"
        )


@router.get("/{song_id}/musical-dna", response_model=MusicalDNAResponse)
async def get_musical_dna(
    song_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get Musical DNA analysis for a song.
    
    Returns 202 if analysis is still in progress.
    """
    try:
        song_service = SongService(db)
        song = await song_service.get_song_by_id(song_id)
        
        if not song:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Song not found"
            )
        
        if song.processing_status == ProcessingStatus.PROCESSING:
            return JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content={"message": "Analysis in progress"}
            )
        
        if song.processing_status == ProcessingStatus.FAILED:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Song processing failed"
            )
        
        if not song.musical_dna:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Musical DNA analysis not found"
            )
        
        return MusicalDNAResponse.from_orm(song.musical_dna)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get Musical DNA", song_id=song_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve Musical DNA"
        )


@router.get("/{song_id}/stems")
async def get_stems(
    song_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get separated stems for a song.
    
    Returns 202 if stem separation is still in progress.
    """
    try:
        song_service = SongService(db)
        song = await song_service.get_song_by_id(song_id)
        
        if not song:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Song not found"
            )
        
        if song.processing_status == ProcessingStatus.PROCESSING:
            return JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content={"message": "Stem separation in progress"}
            )
        
        stems = await song_service.get_stems_by_song_id(song_id)
        return stems
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get stems", song_id=song_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve stems"
        )
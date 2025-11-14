"""
Song service for business logic and database operations.
"""

import os
import shutil
import structlog
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from uuid import UUID

from fastapi import UploadFile
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from ..core.config import get_settings
from ..models.song import Song, MusicalDNA, Stem, StemFeatures, ProcessingStatus
from .task_service import TaskService

logger = structlog.get_logger(__name__)


class SongService:
    """Service for song-related operations."""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.settings = get_settings()
        self.task_service = TaskService()

    async def create_song(
        self,
        file: UploadFile,
        title: str,
        artist: str,
        album: Optional[str] = None
    ) -> Song:
        """
        Create a new song record and save the audio file.
        
        Triggers asynchronous processing pipeline.
        """
        logger.info("Creating song", title=title, artist=artist, filename=file.filename)
        
        # Validate file size
        file_content = await file.read()
        file_size = len(file_content)
        
        if file_size > self.settings.MAX_FILE_SIZE:
            raise ValueError(f"File too large. Maximum size: {self.settings.MAX_FILE_SIZE} bytes")
        
        # Reset file position
        await file.seek(0)
        
        # Create song record
        song = Song(
            title=title,
            artist=artist,
            album=album,
            file_format=file.filename.split(".")[-1].lower(),
            file_size=file_size,
            duration=0.0,  # Will be updated during processing
            file_path="",  # Will be set after saving file
            processing_status=ProcessingStatus.PENDING
        )
        
        # Save to database first to get ID
        self.db.add(song)
        await self.db.commit()
        await self.db.refresh(song)
        
        try:
            # Save audio file
            file_path = await self._save_audio_file(file, song.id, song.file_format)
            song.file_path = file_path
            
            await self.db.commit()
            
            # Trigger async processing
            await self.task_service.start_song_processing(song.id)
            
            logger.info("Song created successfully", song_id=song.id, file_path=file_path)
            return song
            
        except Exception as e:
            # Cleanup on error
            await self.db.rollback()
            logger.error("Failed to create song", error=str(e))
            raise
    
    async def _save_audio_file(self, file: UploadFile, song_id: UUID, file_format: str) -> str:
        """Save uploaded audio file to disk."""
        # Create storage directory
        storage_path = Path(self.settings.AUDIO_STORAGE_PATH)
        storage_path.mkdir(parents=True, exist_ok=True)
        
        # Generate file path
        filename = f"{song_id}.{file_format}"
        file_path = storage_path / filename
        
        # Save file
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        return str(file_path)

    async def get_song_by_id(self, song_id: UUID) -> Optional[Song]:
        """Get song by ID with related data."""
        query = select(Song).options(
            joinedload(Song.musical_dna),
            joinedload(Song.stems).joinedload(Stem.features)
        ).where(Song.id == song_id)
        
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def list_songs(
        self,
        limit: int = 20,
        offset: int = 0,
        key_signature: Optional[str] = None,
        bpm_min: Optional[float] = None,
        bpm_max: Optional[float] = None,
        processing_status: Optional[ProcessingStatus] = None
    ) -> Dict:
        """List songs with filtering and pagination."""
        
        # Build base query
        query = select(Song)
        count_query = select(func.count(Song.id))
        
        # Apply filters
        if key_signature:
            query = query.join(MusicalDNA).where(MusicalDNA.key_signature == key_signature)
            count_query = count_query.join(MusicalDNA).where(MusicalDNA.key_signature == key_signature)
        
        if bpm_min:
            query = query.join(MusicalDNA).where(MusicalDNA.bpm >= bpm_min)
            count_query = count_query.join(MusicalDNA).where(MusicalDNA.bpm >= bpm_min)
        
        if bpm_max:
            query = query.join(MusicalDNA).where(MusicalDNA.bpm <= bmp_max)
            count_query = count_query.join(MusicalDNA).where(MusicalDNA.bpm <= bpm_max)
        
        if processing_status:
            query = query.where(Song.processing_status == processing_status)
            count_query = count_query.where(Song.processing_status == processing_status)
        
        # Apply pagination
        query = query.offset(offset).limit(limit)
        
        # Execute queries
        songs_result = await self.db.execute(query)
        count_result = await self.db.execute(count_query)
        
        songs = songs_result.scalars().all()
        total = count_result.scalar()
        
        return {
            "songs": songs,
            "total": total
        }

    async def delete_song(self, song_id: UUID) -> bool:
        """Delete song and associated files."""
        song = await self.get_song_by_id(song_id)
        
        if not song:
            return False
        
        try:
            # Delete audio files
            if os.path.exists(song.file_path):
                os.remove(song.file_path)
            
            # Delete stem files
            for stem in song.stems:
                if os.path.exists(stem.file_path):
                    os.remove(stem.file_path)
            
            # Delete database record (cascades to related records)
            await self.db.delete(song)
            await self.db.commit()
            
            logger.info("Song deleted successfully", song_id=song_id)
            return True
            
        except Exception as e:
            await self.db.rollback()
            logger.error("Failed to delete song", song_id=song_id, error=str(e))
            raise

    async def get_stems_by_song_id(self, song_id: UUID) -> List[Stem]:
        """Get all stems for a song."""
        query = select(Stem).options(
            joinedload(Stem.features)
        ).where(Stem.song_id == song_id)
        
        result = await self.db.execute(query)
        return result.scalars().all()

    async def update_processing_status(
        self,
        song_id: UUID,
        status: ProcessingStatus,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None
    ) -> None:
        """Update song processing status."""
        query = select(Song).where(Song.id == song_id)
        result = await self.db.execute(query)
        song = result.scalar_one_or_none()
        
        if song:
            song.processing_status = status
            if started_at:
                song.processing_started_at = started_at
            if completed_at:
                song.processing_completed_at = completed_at
            
            await self.db.commit()
            logger.info("Processing status updated", song_id=song_id, status=status)
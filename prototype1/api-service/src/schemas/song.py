"""
Pydantic schemas for song-related API responses.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from ..models.song import ProcessingStatus, StemType


class SongResponse(BaseModel):
    """Basic song response schema."""
    id: UUID
    title: str
    artist: str
    album: Optional[str]
    duration: float
    file_format: str
    file_size: int
    upload_timestamp: datetime
    processing_status: ProcessingStatus
    user_id: Optional[UUID]

    class Config:
        from_attributes = True


class MusicalDNAResponse(BaseModel):
    """Musical DNA analysis response schema."""
    id: UUID
    song_id: UUID
    key_signature: str
    camelot_key: str
    bpm: float
    time_signature: str
    energy_curve: List[float]
    loudness_lufs: float
    dynamic_range: float
    spectral_centroid: List[float]
    created_timestamp: datetime

    class Config:
        from_attributes = True


class StemFeaturesResponse(BaseModel):
    """Stem features response schema."""
    id: UUID
    stem_id: UUID
    rhythmic_pattern: List[float]
    harmonic_content: List[float]
    timbre_features: List[float]
    tempo_stability: float
    percussive_content: float
    pitch_content: Optional[List[float]]

    class Config:
        from_attributes = True


class StemResponse(BaseModel):
    """Stem response schema."""
    id: UUID
    song_id: UUID
    stem_type: StemType
    file_path: str
    confidence_score: float
    rms_energy: float
    spectral_bandwidth: float
    features: Optional[StemFeaturesResponse]

    class Config:
        from_attributes = True


class SongDetailResponse(BaseModel):
    """Detailed song response with Musical DNA and stems."""
    id: UUID
    title: str
    artist: str
    album: Optional[str]
    duration: float
    file_format: str
    file_size: int
    upload_timestamp: datetime
    processing_status: ProcessingStatus
    user_id: Optional[UUID]
    musical_dna: Optional[MusicalDNAResponse]
    stems: List[StemResponse]

    class Config:
        from_attributes = True
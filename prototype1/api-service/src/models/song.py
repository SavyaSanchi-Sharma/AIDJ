"""
Song database model with Musical DNA.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import UUID, uuid4

from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean, Text, JSON
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID, ARRAY
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from ..core.database import Base


class ProcessingStatus(str, Enum):
    """Song processing status options."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Song(Base):
    """Song model representing an uploaded audio file."""
    
    __tablename__ = "songs"
    
    # Primary key
    id: UUID = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Basic metadata
    title: str = Column(String(200), nullable=False, index=True)
    artist: str = Column(String(200), nullable=False, index=True)
    album: Optional[str] = Column(String(200), nullable=True)
    
    # File information
    file_path: str = Column(String(500), nullable=False)
    file_format: str = Column(String(10), nullable=False)
    file_size: int = Column(Integer, nullable=False)
    duration: float = Column(Float, nullable=False)
    
    # Processing status
    processing_status: ProcessingStatus = Column(
        String(20), 
        nullable=False, 
        default=ProcessingStatus.PENDING,
        index=True
    )
    
    # Timestamps
    upload_timestamp: datetime = Column(DateTime, nullable=False, default=datetime.utcnow)
    processing_started_at: Optional[datetime] = Column(DateTime, nullable=True)
    processing_completed_at: Optional[datetime] = Column(DateTime, nullable=True)
    
    # Optional user association
    user_id: Optional[UUID] = Column(PostgresUUID(as_uuid=True), nullable=True)
    
    # Relationships
    musical_dna = relationship("MusicalDNA", back_populates="song", uselist=False)
    stems = relationship("Stem", back_populates="song")
    
    def __repr__(self):
        return f"<Song(id={self.id}, title='{self.title}', artist='{self.artist}')>"


class MusicalDNA(Base):
    """Musical DNA model with vector embeddings for similarity search."""
    
    __tablename__ = "musical_dna"
    
    # Primary key
    id: UUID = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    song_id: UUID = Column(PostgresUUID(as_uuid=True), nullable=False, unique=True, index=True)
    
    # Basic musical features
    key_signature: str = Column(String(20), nullable=False, index=True)
    camelot_key: str = Column(String(5), nullable=False, index=True)
    bpm: float = Column(Float, nullable=False, index=True)
    time_signature: str = Column(String(10), nullable=False, default="4/4")
    
    # Audio analysis features
    energy_curve: List[float] = Column(ARRAY(Float), nullable=False)
    loudness_lufs: float = Column(Float, nullable=False)
    dynamic_range: float = Column(Float, nullable=False)
    
    # Spectral features (arrays)
    spectral_centroid: List[float] = Column(ARRAY(Float), nullable=False)
    spectral_rolloff: List[float] = Column(ARRAY(Float), nullable=False)
    zero_crossing_rate: List[float] = Column(ARRAY(Float), nullable=False)
    
    # Feature matrices (stored as JSON for flexibility)
    mfcc_features: dict = Column(JSON, nullable=False)
    chroma_features: dict = Column(JSON, nullable=False)
    
    # Vector embeddings for similarity search (pgvector)
    global_embedding: List[float] = Column(Vector(512), nullable=False)
    harmonic_embedding: List[float] = Column(Vector(256), nullable=False)
    rhythmic_embedding: List[float] = Column(Vector(256), nullable=False)
    timbral_embedding: List[float] = Column(Vector(256), nullable=False)
    
    # Timestamp
    created_timestamp: datetime = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    song = relationship("Song", back_populates="musical_dna")
    
    def __repr__(self):
        return f"<MusicalDNA(song_id={self.song_id}, key={self.key_signature}, bpm={self.bpm})>"


class StemType(str, Enum):
    """Stem type options."""
    VOCALS = "vocals"
    DRUMS = "drums"
    BASS = "bass"
    OTHER = "other"
    HARMONY = "harmony"


class Stem(Base):
    """Individual separated audio stem."""
    
    __tablename__ = "stems"
    
    # Primary key
    id: UUID = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    song_id: UUID = Column(PostgresUUID(as_uuid=True), nullable=False, index=True)
    
    # Stem information
    stem_type: StemType = Column(String(20), nullable=False, index=True)
    file_path: str = Column(String(500), nullable=False)
    confidence_score: float = Column(Float, nullable=False)
    
    # Audio characteristics
    rms_energy: float = Column(Float, nullable=False)
    spectral_bandwidth: float = Column(Float, nullable=False)
    
    # Relationships
    song = relationship("Song", back_populates="stems")
    features = relationship("StemFeatures", back_populates="stem", uselist=False)
    
    def __repr__(self):
        return f"<Stem(id={self.id}, type={self.stem_type}, confidence={self.confidence_score:.2f})>"


class StemFeatures(Base):
    """Musical analysis specific to each stem."""
    
    __tablename__ = "stem_features"
    
    # Primary key
    id: UUID = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    stem_id: UUID = Column(PostgresUUID(as_uuid=True), nullable=False, unique=True, index=True)
    
    # Rhythmic and harmonic analysis
    rhythmic_pattern: List[float] = Column(ARRAY(Float), nullable=False)
    harmonic_content: List[float] = Column(ARRAY(Float), nullable=False)
    timbre_features: List[float] = Column(ARRAY(Float), nullable=False)
    tempo_stability: float = Column(Float, nullable=False)
    percussive_content: float = Column(Float, nullable=False)
    
    # Pitch content (for melodic stems)
    pitch_content: Optional[List[float]] = Column(ARRAY(Float), nullable=True)
    
    # Vector embedding for stem similarity (pgvector)
    stem_embedding: List[float] = Column(Vector(256), nullable=False)
    
    # Relationships
    stem = relationship("Stem", back_populates="features")
    
    def __repr__(self):
        return f"<StemFeatures(stem_id={self.stem_id}, tempo_stability={self.tempo_stability:.2f})>"
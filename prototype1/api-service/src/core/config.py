"""
Application configuration management.
"""

from functools import lru_cache
from typing import Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    HOST: str = Field(default="0.0.0.0", description="API host")
    PORT: int = Field(default=8000, description="API port")
    DEBUG: bool = Field(default=False, description="Debug mode")
    SECRET_KEY: str = Field(description="Secret key for JWT tokens")
    
    # Database Configuration
    DATABASE_URL: str = Field(description="PostgreSQL database URL")
    DATABASE_ECHO: bool = Field(default=False, description="Echo SQL queries")
    
    # Redis Configuration
    REDIS_URL: str = Field(description="Redis URL for caching")
    
    # RabbitMQ Configuration
    RABBITMQ_URL: str = Field(description="RabbitMQ URL for task queue")
    
    # Storage Configuration
    AUDIO_STORAGE_PATH: str = Field(default="./data/audio", description="Local audio storage path")
    S3_BUCKET: Optional[str] = Field(default=None, description="S3 bucket for audio files")
    S3_REGION: Optional[str] = Field(default=None, description="S3 region")
    CDN_BASE_URL: Optional[str] = Field(default=None, description="CDN base URL")
    
    # ML Configuration
    GPU_ENABLED: bool = Field(default=False, description="Enable GPU acceleration")
    DEMUCS_MODEL_PATH: str = Field(default="./models/demucs", description="Path to Demucs models")
    FAISS_INDEX_PATH: str = Field(default="./data/faiss_indices", description="Path to FAISS indices")
    
    # Processing Limits
    MAX_FILE_SIZE: int = Field(default=100 * 1024 * 1024, description="Max file size in bytes (100MB)")
    MAX_SONG_DURATION: int = Field(default=3600, description="Max song duration in seconds (1 hour)")
    PROCESSING_TIMEOUT: int = Field(default=300, description="Processing timeout in seconds")
    
    # Performance Settings
    MAX_CONCURRENT_PROCESSING: int = Field(default=4, description="Max concurrent processing tasks")
    SIMILARITY_SEARCH_LIMIT: int = Field(default=1000, description="Max similarity search results")
    
    @validator("DATABASE_URL")
    def validate_database_url(cls, v):
        """Ensure database URL is provided."""
        if not v:
            raise ValueError("DATABASE_URL is required")
        return v
    
    @validator("SECRET_KEY")
    def validate_secret_key(cls, v):
        """Ensure secret key is provided."""
        if not v:
            raise ValueError("SECRET_KEY is required")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
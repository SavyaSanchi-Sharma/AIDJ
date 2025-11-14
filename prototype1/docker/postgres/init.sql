-- Initialize PostgreSQL database with pgvector extension
-- This script runs when the database container starts for the first time

-- Create the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create initial database structure (basic setup)
-- Full schema will be created by Alembic migrations

-- Enable pgvector extension
SELECT extname FROM pg_extension WHERE extname = 'vector';

-- Set up initial permissions
GRANT ALL PRIVILEGES ON DATABASE music_mixer TO postgres;

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'PostgreSQL with pgvector extension initialized successfully';
END $$;
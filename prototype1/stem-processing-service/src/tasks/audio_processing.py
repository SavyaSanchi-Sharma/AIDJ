"""
Celery tasks for audio processing with Demucs stem separation.
"""

import os
import structlog
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from uuid import UUID

from ..celery_app import app
from ..audio.demucs_processor import DemucsProcessor
from ..audio.feature_extractor import FeatureExtractor
from ..ml.musical_dna_generator import MusicalDNAGenerator
from ..database.connection import get_db_session, update_song_status

logger = structlog.get_logger(__name__)


@app.task(bind=True, max_retries=3)
def process_song(self, song_id: str) -> Dict[str, Any]:
    """
    Complete audio processing pipeline for a song.
    
    Steps:
    1. Load audio file
    2. Separate stems using Demucs
    3. Extract features from each stem
    4. Generate Musical DNA vectors
    5. Update database with results
    """
    song_uuid = UUID(song_id)
    logger.info("Starting song processing", song_id=song_id, task_id=self.request.id)
    
    try:
        # Update status to processing
        with get_db_session() as db:
            update_song_status(db, song_uuid, "processing", started_at=datetime.utcnow())
        
        # Check GPU availability
        device = "cuda" if torch.cuda.is_available() and os.getenv("GPU_ENABLED", "false").lower() == "true" else "cpu"
        logger.info("Using device for processing", device=device, song_id=song_id)
        
        # Step 1: Initialize processors
        demucs_processor = DemucsProcessor(device=device)
        feature_extractor = FeatureExtractor()
        dna_generator = MusicalDNAGenerator()
        
        # Step 2: Get song file path from database
        with get_db_session() as db:
            song = db.query(Song).filter(Song.id == song_uuid).first()
            if not song:
                raise ValueError(f"Song {song_id} not found in database")
            
            audio_file_path = song.file_path
        
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        logger.info("Processing audio file", file_path=audio_file_path, song_id=song_id)
        
        # Step 3: Separate stems using Demucs
        self.update_state(
            state='PROGRESS',
            meta={'current': 1, 'total': 5, 'status': 'Separating stems with Demucs'}
        )
        
        stems_result = demucs_processor.separate_stems(audio_file_path)
        logger.info("Stem separation completed", song_id=song_id, stems_count=len(stems_result))
        
        # Step 4: Extract features from original and stems
        self.update_state(
            state='PROGRESS',
            meta={'current': 2, 'total': 5, 'status': 'Extracting audio features'}
        )
        
        # Extract features from original track
        global_features = feature_extractor.extract_global_features(audio_file_path)
        
        # Extract features from each stem
        stem_features = {}
        for stem_type, stem_path in stems_result.items():
            stem_features[stem_type] = feature_extractor.extract_stem_features(stem_path, stem_type)
        
        logger.info("Feature extraction completed", song_id=song_id)
        
        # Step 5: Generate Musical DNA vectors
        self.update_state(
            state='PROGRESS',
            meta={'current': 3, 'total': 5, 'status': 'Generating Musical DNA vectors'}
        )
        
        musical_dna = dna_generator.generate_musical_dna(
            global_features=global_features,
            stem_features=stem_features,
            audio_file_path=audio_file_path
        )
        
        logger.info("Musical DNA generation completed", song_id=song_id)
        
        # Step 6: Update database with results
        self.update_state(
            state='PROGRESS',
            meta={'current': 4, 'total': 5, 'status': 'Saving results to database'}
        )
        
        with get_db_session() as db:
            # Save stems
            saved_stems = []
            for stem_type, stem_path in stems_result.items():
                stem_record = Stem(
                    song_id=song_uuid,
                    stem_type=stem_type,
                    file_path=stem_path,
                    confidence_score=stems_result.get(f"{stem_type}_confidence", 0.95),
                    rms_energy=stem_features[stem_type].get("rms_energy", 0.0),
                    spectral_bandwidth=stem_features[stem_type].get("spectral_bandwidth", 0.0)
                )
                db.add(stem_record)
                db.flush()  # Get ID
                
                # Save stem features
                stem_features_record = StemFeatures(
                    stem_id=stem_record.id,
                    rhythmic_pattern=stem_features[stem_type]["rhythmic_pattern"],
                    harmonic_content=stem_features[stem_type]["harmonic_content"],
                    timbre_features=stem_features[stem_type]["timbre_features"],
                    tempo_stability=stem_features[stem_type]["tempo_stability"],
                    percussive_content=stem_features[stem_type]["percussive_content"],
                    pitch_content=stem_features[stem_type].get("pitch_content"),
                    stem_embedding=stem_features[stem_type]["embedding"]
                )
                db.add(stem_features_record)
                saved_stems.append(stem_record.id)
            
            # Save Musical DNA
            musical_dna_record = MusicalDNA(
                song_id=song_uuid,
                key_signature=musical_dna["key_signature"],
                camelot_key=musical_dna["camelot_key"],
                bpm=musical_dna["bpm"],
                time_signature=musical_dna["time_signature"],
                energy_curve=musical_dna["energy_curve"],
                loudness_lufs=musical_dna["loudness_lufs"],
                dynamic_range=musical_dna["dynamic_range"],
                spectral_centroid=global_features["spectral_centroid"],
                spectral_rolloff=global_features["spectral_rolloff"],
                zero_crossing_rate=global_features["zero_crossing_rate"],
                mfcc_features=global_features["mfcc_features"],
                chroma_features=global_features["chroma_features"],
                global_embedding=musical_dna["global_embedding"],
                harmonic_embedding=musical_dna["harmonic_embedding"],
                rhythmic_embedding=musical_dna["rhythmic_embedding"],
                timbral_embedding=musical_dna["timbral_embedding"]
            )
            db.add(musical_dna_record)
            
            # Update song with duration and completion status
            song.duration = global_features["duration"]
            song.processing_status = "completed"
            song.processing_completed_at = datetime.utcnow()
            
            db.commit()
        
        logger.info("Song processing completed successfully", song_id=song_id, task_id=self.request.id)
        
        return {
            "status": "completed",
            "song_id": song_id,
            "stems_processed": list(stems_result.keys()),
            "duration": global_features["duration"],
            "bpm": musical_dna["bpm"],
            "key": musical_dna["key_signature"]
        }
        
    except Exception as exc:
        logger.error("Song processing failed", song_id=song_id, error=str(exc), task_id=self.request.id)
        
        # Update status to failed
        try:
            with get_db_session() as db:
                update_song_status(db, song_uuid, "failed")
        except Exception as db_exc:
            logger.error("Failed to update song status to failed", song_id=song_id, error=str(db_exc))
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info("Retrying song processing", song_id=song_id, retry_count=self.request.retries + 1)
            raise self.retry(countdown=60 * (self.request.retries + 1))
        
        raise exc


# Import models for the task (these would be in shared models)
# For now, we'll use placeholder imports
class Song:
    pass

class Stem:
    pass

class StemFeatures:
    pass

class MusicalDNA:
    pass
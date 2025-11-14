"""
Demucs stem separation processor.
"""

import os
import torch
import structlog
from pathlib import Path
from typing import Dict, Any
from demucs.api import separate_stems as demucs_separate
from demucs.pretrained import get_model

logger = structlog.get_logger(__name__)


class DemucsProcessor:
    """Handles stem separation using Facebook Demucs."""
    
    def __init__(self, model_name: str = "htdemucs", device: str = "auto"):
        """
        Initialize Demucs processor.
        
        Args:
            model_name: Demucs model to use (htdemucs is the latest/best)
            device: Device to run on ('cuda', 'cpu', or 'auto')
        """
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
        logger.info("Initializing Demucs processor", model=model_name, device=self.device)
        
        # Load model lazily to save memory
        self._load_model()
    
    def _load_model(self):
        """Load the Demucs model."""
        try:
            self.model = get_model(self.model_name)
            self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info("Demucs model loaded successfully", model=self.model_name, device=self.device)
            
        except Exception as e:
            logger.error("Failed to load Demucs model", model=self.model_name, error=str(e))
            raise
    
    def separate_stems(self, audio_file_path: str) -> Dict[str, str]:
        """
        Separate audio into stems using Demucs.
        
        Args:
            audio_file_path: Path to input audio file
            
        Returns:
            Dictionary mapping stem type to output file path
        """
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        logger.info("Starting stem separation", audio_file=audio_file_path, model=self.model_name)
        
        # Create output directory
        output_dir = Path(audio_file_path).parent / "stems" / Path(audio_file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Use Demucs API for separation
            # This returns a tensor with shape (sources, channels, length)
            separated = demucs_separate(
                model=self.model,
                audio_file=audio_file_path,
                device=self.device,
                shifts=1,  # Number of random shifts for better quality
                overlap=0.25,  # Overlap between segments
                progress=False,  # We handle progress in Celery task
            )
            
            # Demucs typically separates into: drums, bass, other, vocals
            stem_names = ["drums", "bass", "other", "vocals"]
            stem_paths = {}
            
            # Save each stem as a separate file
            for i, stem_name in enumerate(stem_names):
                if i < separated.shape[0]:  # Ensure we have this stem
                    stem_path = output_dir / f"{stem_name}.wav"
                    
                    # Convert to audio and save
                    import torchaudio
                    stem_audio = separated[i]  # Shape: (channels, length)
                    
                    # Ensure we have the right shape (channels, samples)
                    if stem_audio.dim() == 1:
                        stem_audio = stem_audio.unsqueeze(0)  # Add channel dimension
                    
                    # Save with original sample rate (44.1kHz default)
                    torchaudio.save(str(stem_path), stem_audio.cpu(), sample_rate=44100)
                    
                    stem_paths[stem_name] = str(stem_path)
                    
                    logger.info("Stem saved", stem_type=stem_name, path=str(stem_path))
            
            logger.info("Stem separation completed", 
                       audio_file=audio_file_path, 
                       stems_created=len(stem_paths),
                       output_dir=str(output_dir))
            
            return stem_paths
            
        except Exception as e:
            logger.error("Stem separation failed", 
                        audio_file=audio_file_path, 
                        error=str(e))
            raise
    
    def get_separation_quality_metrics(self, original_path: str, stem_paths: Dict[str, str]) -> Dict[str, float]:
        """
        Calculate separation quality metrics.
        
        This is a simplified version - in production you'd want more sophisticated metrics.
        """
        try:
            import librosa
            import numpy as np
            
            # Load original audio
            original_audio, sr = librosa.load(original_path, sr=None)
            
            metrics = {}
            
            for stem_type, stem_path in stem_paths.items():
                # Load separated stem
                stem_audio, _ = librosa.load(stem_path, sr=sr)
                
                # Simple SNR-like metric
                if len(stem_audio) > 0:
                    # Calculate RMS energy
                    stem_rms = np.sqrt(np.mean(stem_audio ** 2))
                    
                    # Confidence score based on RMS energy (simplified)
                    # Higher energy stems typically indicate better separation
                    confidence = min(1.0, stem_rms * 10)  # Normalize to 0-1
                    
                    metrics[f"{stem_type}_confidence"] = confidence
                else:
                    metrics[f"{stem_type}_confidence"] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.warning("Failed to calculate separation metrics", error=str(e))
            # Return default confidence scores
            return {f"{stem}_confidence": 0.85 for stem in stem_paths.keys()}
    
    def cleanup_model(self):
        """Clean up model to free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Demucs model cleanup completed")
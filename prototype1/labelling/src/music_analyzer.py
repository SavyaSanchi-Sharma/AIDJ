"""
Music Analysis Module for AI Labeling.
Extracts audio features and prepares data for GPT analysis.
"""

import librosa
import numpy as np
import structlog
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

logger = structlog.get_logger(__name__)


@dataclass
class TrackAnalysis:
    """Basic track analysis data."""
    
    # File info
    file_path: str
    title: str
    artist: str
    duration: float
    
    # Audio features  
    tempo: float
    key: str
    energy_curve: List[float]
    onset_times: List[float]
    
    # Spectral features
    spectral_centroids: List[float]
    rms_energy: List[float]
    
    # Structure hints
    beat_times: List[float]
    estimated_sections: int


class BasicMusicAnalyzer:
    """
    Simplified music analyzer that extracts essential features for GPT analysis.
    Focuses on features that help identify song sections and mix points.
    """
    
    def __init__(self, sample_rate: int = 22050):
        """Initialize with lower sample rate for faster processing."""
        
        self.sample_rate = sample_rate
        logger.info("BasicMusicAnalyzer initialized", sample_rate=sample_rate)
    
    def analyze_track(self, audio_file_path: str) -> TrackAnalysis:
        """
        Analyze a single track and extract essential features.
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            TrackAnalysis object with extracted features
        """
        
        logger.info("Analyzing track", file_path=audio_file_path)
        
        try:
            # Load audio
            y, sr = librosa.load(audio_file_path, sr=self.sample_rate)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Basic file info
            file_path = Path(audio_file_path)
            title = file_path.stem.replace("_", " ").replace("-", " ")
            artist = "Unknown Artist"  # Could extract from metadata
            
            # Tempo and beats
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beats, sr=sr).tolist()
            
            # Key detection (simplified)
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            key_profile = np.mean(chroma, axis=1)
            key_idx = np.argmax(key_profile)
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key = key_names[key_idx]
            
            # Energy analysis
            rms = librosa.feature.rms(y=y, hop_length=512)[0]
            rms_energy = rms.tolist()
            
            # Create smoothed energy curve (for visualization/analysis)
            hop_length = len(y) // 100  # 100 points across the song
            energy_curve = []
            
            for i in range(0, len(y) - hop_length, hop_length):
                segment_energy = np.mean(rms[i//512:(i+hop_length)//512]) if i//512 < len(rms) else 0
                energy_curve.append(float(segment_energy))
            
            # Onset detection
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, wait=1)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0].tolist()
            
            # Estimate number of sections (rough heuristic)
            estimated_sections = max(3, min(8, int(duration / 30)))  # ~30sec per section
            
            analysis = TrackAnalysis(
                file_path=str(file_path),
                title=title,
                artist=artist,
                duration=float(duration),
                tempo=float(tempo),
                key=key,
                energy_curve=energy_curve,
                onset_times=onset_times,
                spectral_centroids=spectral_centroids,
                rms_energy=rms_energy,
                beat_times=beat_times,
                estimated_sections=estimated_sections
            )
            
            logger.info("Track analysis completed", 
                       title=title,
                       duration=f"{duration:.1f}s",
                       tempo=f"{tempo:.1f} BPM",
                       key=key)
            
            return analysis
            
        except Exception as e:
            logger.error("Track analysis failed", 
                        file_path=audio_file_path, 
                        error=str(e))
            raise
    
    def create_visualization_data(self, analysis: TrackAnalysis) -> Dict[str, Any]:
        """Create data for visualizing the track analysis."""
        
        duration = analysis.duration
        
        # Create time axis
        time_points = np.linspace(0, duration, len(analysis.energy_curve))
        
        visualization_data = {
            "title": analysis.title,
            "duration": duration,
            "tempo": analysis.tempo,
            "key": analysis.key,
            "time_axis": time_points.tolist(),
            "energy_curve": analysis.energy_curve,
            "beat_times": analysis.beat_times,
            "onset_times": analysis.onset_times,
            "estimated_sections": analysis.estimated_sections
        }
        
        return visualization_data
    
    def save_analysis(self, analysis: TrackAnalysis, output_file: str):
        """Save analysis to JSON file."""
        
        try:
            analysis_dict = asdict(analysis)
            
            with open(output_file, 'w') as f:
                json.dump(analysis_dict, f, indent=2, default=str)
            
            logger.info("Analysis saved", output_file=output_file)
            
        except Exception as e:
            logger.error("Failed to save analysis", 
                        output_file=output_file, 
                        error=str(e))
    
    def load_analysis(self, input_file: str) -> TrackAnalysis:
        """Load analysis from JSON file."""
        
        try:
            with open(input_file, 'r') as f:
                analysis_dict = json.load(f)
            
            analysis = TrackAnalysis(**analysis_dict)
            
            logger.info("Analysis loaded", input_file=input_file)
            return analysis
            
        except Exception as e:
            logger.error("Failed to load analysis", 
                        input_file=input_file, 
                        error=str(e))
            raise


def analyze_multiple_tracks(audio_files: List[str], 
                          output_dir: str = "analysis_output") -> List[TrackAnalysis]:
    """
    Analyze multiple tracks and save results.
    
    Args:
        audio_files: List of audio file paths
        output_dir: Directory to save analysis results
        
    Returns:
        List of TrackAnalysis objects
    """
    
    analyzer = BasicMusicAnalyzer()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    analyses = []
    
    for i, audio_file in enumerate(audio_files):
        try:
            logger.info("Processing file", 
                       file=audio_file, 
                       progress=f"{i+1}/{len(audio_files)}")
            
            # Analyze track
            analysis = analyzer.analyze_track(audio_file)
            analyses.append(analysis)
            
            # Save individual analysis
            output_file = output_path / f"{analysis.title.replace(' ', '_')}_analysis.json"
            analyzer.save_analysis(analysis, str(output_file))
            
        except Exception as e:
            logger.error("Failed to process file", 
                        file=audio_file, 
                        error=str(e))
            continue
    
    # Save combined analysis
    combined_file = output_path / "combined_analysis.json"
    combined_data = {
        "timestamp": datetime.now().isoformat(),
        "total_tracks": len(analyses),
        "tracks": [asdict(analysis) for analysis in analyses]
    }
    
    with open(combined_file, 'w') as f:
        json.dump(combined_data, f, indent=2, default=str)
    
    logger.info("Batch analysis completed", 
               total_processed=len(analyses),
               output_dir=output_dir)
    
    return analyses
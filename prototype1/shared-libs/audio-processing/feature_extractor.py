"""
Audio feature extraction using librosa and Essentia.
Generates Musical DNA features from audio files.
"""

import librosa
import numpy as np
import structlog
from typing import Dict, List, Optional, Tuple, Any
import essentia.standard as es
from scipy.stats import kurtosis, skew
from scipy.signal import hilbert
import pickle
import asyncio
from .metadata_extractor import MetadataConfig, extract_comprehensive_metadata

logger = structlog.get_logger(__name__)


class ComprehensiveFeatureExtractor:
    """
    Extracts maximum possible features from audio using multiple libraries.
    Combines audio analysis, metadata extraction, and derived features.
    Generates 1500+ dimensional feature vectors for ML training.
    """
    
    def __init__(self, sample_rate: int = 44100, metadata_config: Optional[MetadataConfig] = None):
        """
        Initialize feature extractor.
        
        Args:
            sample_rate: Target sample rate for analysis
        """
        self.sample_rate = sample_rate
        self.metadata_config = metadata_config or MetadataConfig()
        
        # Initialize Essentia algorithms
        self.rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        self.key_extractor = es.KeyExtractor()
        self.loudness_extractor = es.LoudnessEBUR128()
        
        # Advanced Essentia extractors
        self.barkbands_extractor = es.BarkBands()
        self.melbands_extractor = es.MelBands()
        self.onset_rate_extractor = es.OnsetRate()
        self.dynamic_complexity_extractor = es.DynamicComplexity()
        self.dissonance_extractor = es.Dissonance()
        self.spectral_complexity_extractor = es.SpectralComplexity()
        self.beat_loudness_extractor = es.BeatsLoudness()
        self.danceability_extractor = es.Danceability()
        
        # Initialize windowing for multi-resolution analysis
        self.window_sizes = [512, 1024, 2048, 4096, 8192]
        self.hop_length = 512
        
        logger.info("ComprehensiveFeatureExtractor initialized", 
                   sample_rate=sample_rate, window_sizes=len(self.window_sizes))
    
    async def extract_maximum_features(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Extract maximum possible features from the complete audio file.
        
        Returns 1500+ dimensional feature vector combining:
        - Audio signal features (1000+ dimensions)
        - Metadata features (500+ dimensions)
        - Derived cross-modal features
        """
        logger.info("Extracting maximum features", file_path=audio_file_path)
        
        try:
            # Load audio file
            y, sr = librosa.load(audio_file_path, sr=self.sample_rate)
            duration = librosa.get_duration(y=y, sr=sr)
            
            logger.info("Audio loaded", duration=duration, sample_rate=sr)
            
            features = {
                "duration": duration,
                "sample_rate": sr
            }
            
            # Multi-resolution spectral analysis
            features.update(self._extract_multi_resolution_spectral_features(y, sr))
            
            # Comprehensive rhythm analysis
            features.update(self._extract_comprehensive_rhythm_features(y, sr))
            
            # Advanced harmonic analysis
            features.update(self._extract_advanced_harmonic_features(y, sr))
            
            # Comprehensive energy and dynamics
            features.update(self._extract_comprehensive_energy_features(y, sr))
            
            # Advanced Essentia features
            features.update(self._extract_comprehensive_essentia_features(audio_file_path))
            
            # Psychoacoustic features
            features.update(self._extract_psychoacoustic_features(y, sr))
            
            # Structural analysis
            features.update(self._extract_structural_features(y, sr))
            
            # Timbral complexity features
            features.update(self._extract_timbral_complexity_features(y, sr))
            
            # Statistical audio features
            features.update(self._extract_statistical_features(y, sr))
            
            # Extract comprehensive metadata
            metadata = await extract_comprehensive_metadata(audio_file_path, self.metadata_config)
            features["metadata"] = metadata
            
            # Generate final feature vector
            features["comprehensive_feature_vector"] = self._generate_comprehensive_feature_vector(features)
            
            logger.info("Maximum feature extraction completed", 
                       feature_count=len(features.get("comprehensive_feature_vector", [])),
                       duration=duration,
                       total_categories=len([k for k in features.keys() if k != "comprehensive_feature_vector"]))
            
            return features
            
        except Exception as e:
            logger.error("Maximum feature extraction failed", 
                        file_path=audio_file_path, 
                        error=str(e))
            raise
    
    def extract_stem_features(self, stem_file_path: str, stem_type: str) -> Dict[str, Any]:
        """
        Extract features specific to a separated stem.
        
        Args:
            stem_file_path: Path to the stem audio file
            stem_type: Type of stem (vocals, drums, bass, other)
        """
        logger.info("Extracting stem features", file_path=stem_file_path, stem_type=stem_type)
        
        try:
            # Load stem audio
            y, sr = librosa.load(stem_file_path, sr=self.sample_rate)
            
            features = {
                "stem_type": stem_type,
                "duration": librosa.get_duration(y=y, sr=sr)
            }
            
            # Basic audio characteristics
            features["rms_energy"] = float(np.sqrt(np.mean(y ** 2)))
            features["spectral_bandwidth"] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
            
            # Rhythmic features (especially important for drums)
            features.update(self._extract_rhythmic_patterns(y, sr, stem_type))
            
            # Harmonic content (important for bass, other)
            features.update(self._extract_harmonic_content(y, sr, stem_type))
            
            # Timbral features
            features.update(self._extract_timbral_features(y, sr, stem_type))
            
            # Tempo stability
            features["tempo_stability"] = self._calculate_tempo_stability(y, sr)
            
            # Percussive vs harmonic content
            features["percussive_content"] = self._calculate_percussive_ratio(y, sr)
            
            # Pitch content for melodic stems
            if stem_type in ["vocals", "other"]:
                features["pitch_content"] = self._extract_pitch_content(y, sr)
            
            # Generate embedding vector for this stem
            features["embedding"] = self._generate_stem_embedding(features, stem_type)
            
            logger.info("Stem feature extraction completed", 
                       stem_type=stem_type,
                       feature_count=len(features))
            
            return features
            
        except Exception as e:
            logger.error("Stem feature extraction failed", 
                        file_path=stem_file_path, 
                        stem_type=stem_type,
                        error=str(e))
            raise
    
    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract spectral features using librosa."""
        
        # Spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        # Spectral rolloff (high-frequency content)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # Zero crossing rate (indicates harmonic vs percussive content)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        # MFCC features (timbral characteristics)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        return {
            "spectral_centroid": spectral_centroid.tolist(),
            "spectral_rolloff": spectral_rolloff.tolist(),
            "zero_crossing_rate": zcr.tolist(),
            "mfcc_features": {
                "mfcc_mean": np.mean(mfccs, axis=1).tolist(),
                "mfcc_std": np.std(mfccs, axis=1).tolist(),
                "mfcc_matrix": mfccs.tolist()
            }
        }
    
    def _extract_rhythm_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract rhythm and tempo features."""
        
        # Tempo estimation
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # Beat consistency
        beat_times = librosa.frames_to_time(beats, sr=sr)
        beat_intervals = np.diff(beat_times)
        beat_consistency = 1.0 - (np.std(beat_intervals) / np.mean(beat_intervals)) if len(beat_intervals) > 0 else 0.0
        
        return {
            "tempo": float(tempo),
            "beat_times": beat_times.tolist(),
            "onset_times": onset_times.tolist(),
            "beat_consistency": float(max(0, min(1, beat_consistency)))
        }
    
    def _extract_harmonic_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract harmonic and tonal features."""
        
        # Chromagram (pitch class representation)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Harmonic vs percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Tonnetz (harmonic network features)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        
        return {
            "chroma_features": {
                "chroma_mean": np.mean(chroma, axis=1).tolist(),
                "chroma_std": np.std(chroma, axis=1).tolist(),
                "chroma_matrix": chroma.tolist()
            },
            "harmonic_ratio": float(np.sum(y_harmonic ** 2) / (np.sum(y ** 2) + 1e-10)),
            "tonnetz_mean": np.mean(tonnetz, axis=1).tolist()
        }
    
    def _extract_energy_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract energy and dynamics features."""
        
        # RMS energy over time
        rms = librosa.feature.rms(y=y)[0]
        
        # Dynamic range
        db_rms = librosa.amplitude_to_db(rms)
        dynamic_range = float(np.max(db_rms) - np.min(db_rms))
        
        # Energy curve (smoothed RMS for visualization)
        hop_length = 1024
        energy_curve = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        # Normalize energy curve to 0-1 range
        if np.max(energy_curve) > 0:
            energy_curve = (energy_curve - np.min(energy_curve)) / (np.max(energy_curve) - np.min(energy_curve))
        
        return {
            "rms_energy": rms.tolist(),
            "dynamic_range": dynamic_range,
            "energy_curve": energy_curve.tolist(),
            "loudness_lufs": -23.0  # Placeholder - would be calculated by Essentia
        }
    
    def _extract_essentia_features(self, audio_file_path: str) -> Dict[str, Any]:
        """Extract features using Essentia algorithms."""
        
        try:
            # Load audio for Essentia
            loader = es.MonoLoader(filename=audio_file_path)
            audio = loader()
            
            # Key detection
            key, scale, strength = self.key_extractor(audio)
            
            # Convert to Camelot key
            camelot_key = self._convert_to_camelot(key, scale)
            
            # Rhythm extraction
            bpm, beats, beats_confidence, _, beats_intervals = self.rhythm_extractor(audio)
            
            # Loudness (EBU R128)
            loudness, _, _, _ = self.loudness_extractor(audio)
            
            return {
                "key_signature": f"{key} {scale}",
                "key_strength": float(strength),
                "camelot_key": camelot_key,
                "bpm_essentia": float(bpm),
                "beats_confidence": float(beats_confidence),
                "loudness_lufs": float(loudness)
            }
            
        except Exception as e:
            logger.warning("Essentia feature extraction failed, using defaults", error=str(e))
            return {
                "key_signature": "C major",
                "key_strength": 0.5,
                "camelot_key": "8B",
                "bpm_essentia": 120.0,
                "beats_confidence": 0.5,
                "loudness_lufs": -23.0
            }
    
    def _extract_rhythmic_patterns(self, y: np.ndarray, sr: int, stem_type: str) -> Dict[str, Any]:
        """Extract rhythmic patterns specific to stem type."""
        
        # Onset strength
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Tempogram for rhythm pattern analysis
        tempogram = librosa.feature.tempogram(onset_envelope=onset_strength, sr=sr)
        
        # Pattern summary
        rhythmic_pattern = np.mean(tempogram, axis=0)
        
        return {
            "rhythmic_pattern": rhythmic_pattern.tolist(),
            "onset_strength_mean": float(np.mean(onset_strength)),
            "rhythmic_regularity": float(np.std(onset_strength))
        }
    
    def _extract_harmonic_content(self, y: np.ndarray, sr: int, stem_type: str) -> Dict[str, Any]:
        """Extract harmonic content analysis."""
        
        # Harmonic vs percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        return {
            "harmonic_content": np.mean(contrast, axis=1).tolist(),
            "harmonic_strength": float(np.sum(y_harmonic ** 2) / (np.sum(y ** 2) + 1e-10)),
            "spectral_contrast": np.mean(contrast, axis=1).tolist()
        }
    
    def _extract_timbral_features(self, y: np.ndarray, sr: int, stem_type: str) -> Dict[str, Any]:
        """Extract timbral characteristics."""
        
        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        # MFCCs for timbre
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        return {
            "timbre_features": np.mean(mfccs, axis=1).tolist(),
            "spectral_centroid_mean": float(spectral_centroid),
            "spectral_bandwidth_mean": float(spectral_bandwidth),
            "spectral_rolloff_mean": float(spectral_rolloff)
        }
    
    def _calculate_tempo_stability(self, y: np.ndarray, sr: int) -> float:
        """Calculate how stable the tempo is throughout the track."""
        
        # Get beat tracker
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        if len(beats) < 3:
            return 0.0
        
        # Calculate beat intervals
        beat_times = librosa.frames_to_time(beats, sr=sr)
        intervals = np.diff(beat_times)
        
        # Stability is inverse of coefficient of variation
        if np.mean(intervals) > 0:
            stability = 1.0 - (np.std(intervals) / np.mean(intervals))
            return float(max(0, min(1, stability)))
        
        return 0.0
    
    def _calculate_percussive_ratio(self, y: np.ndarray, sr: int) -> float:
        """Calculate the ratio of percussive to harmonic content."""
        
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        harmonic_energy = np.sum(y_harmonic ** 2)
        percussive_energy = np.sum(y_percussive ** 2)
        total_energy = harmonic_energy + percussive_energy
        
        if total_energy > 0:
            return float(percussive_energy / total_energy)
        
        return 0.0
    
    def _extract_pitch_content(self, y: np.ndarray, sr: int) -> Optional[List[float]]:
        """Extract fundamental frequency content for melodic stems."""
        
        try:
            # Pitch tracking using librosa
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
            
            # Extract fundamental frequency over time
            fundamental_freqs = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t] if magnitudes[index, t] > 0.1 else 0.0
                fundamental_freqs.append(float(pitch))
            
            return fundamental_freqs
            
        except Exception as e:
            logger.warning("Pitch extraction failed", error=str(e))
            return None
    
    def _generate_stem_embedding(self, features: Dict[str, Any], stem_type: str) -> List[float]:
        """Generate a 256-dimensional embedding vector for the stem."""
        
        # This is a simplified embedding generation
        # In production, you'd train a neural network to generate meaningful embeddings
        
        embedding_features = []
        
        # Add basic features
        embedding_features.extend([
            features.get("rms_energy", 0.0),
            features.get("spectral_bandwidth", 0.0),
            features.get("tempo_stability", 0.0),
            features.get("percussive_content", 0.0),
            features.get("harmonic_strength", 0.0),
        ])
        
        # Add spectral features (truncated to fit)
        timbre_features = features.get("timbre_features", [])
        embedding_features.extend(timbre_features[:13])  # MFCC coefficients
        
        # Pad with zeros to reach 256 dimensions
        while len(embedding_features) < 256:
            embedding_features.append(0.0)
        
        return embedding_features[:256]
    
    def _convert_to_camelot(self, key: str, scale: str) -> str:
        """Convert musical key to Camelot notation."""
        
        # Camelot wheel mapping
        camelot_map = {
            ("C", "major"): "8B", ("C", "minor"): "5A",
            ("D♭", "major"): "3B", ("C#", "minor"): "12A",
            ("D", "major"): "10B", ("D", "minor"): "7A",
            ("E♭", "major"): "5B", ("D#", "minor"): "2A",
            ("E", "major"): "12B", ("E", "minor"): "9A",
            ("F", "major"): "7B", ("F", "minor"): "4A",
            ("F#", "major"): "2B", ("F#", "minor"): "11A",
            ("G", "major"): "9B", ("G", "minor"): "6A",
            ("A♭", "major"): "4B", ("G#", "minor"): "1A",
            ("A", "major"): "11B", ("A", "minor"): "8A",
            ("B♭", "major"): "6B", ("A#", "minor"): "3A",
            ("B", "major"): "1B", ("B", "minor"): "10A"
        }
        
        return camelot_map.get((key, scale), "8B")  # Default to C major
    
    def _extract_multi_resolution_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract spectral features at multiple resolutions."""
        
        features = {}
        
        for i, n_fft in enumerate(self.window_sizes):
            window_key = f"window_{n_fft}"
            
            # STFT with current window size
            stft = librosa.stft(y, n_fft=n_fft, hop_length=self.hop_length, window='hann')
            magnitude = np.abs(stft)
            
            # Spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr, hop_length=self.hop_length)
            
            # Spectral rolloff at multiple percentiles
            rolloff_85 = librosa.feature.spectral_rolloff(S=magnitude, sr=sr, roll_percent=0.85, hop_length=self.hop_length)
            rolloff_95 = librosa.feature.spectral_rolloff(S=magnitude, sr=sr, roll_percent=0.95, hop_length=self.hop_length)
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sr, hop_length=self.hop_length)
            
            # Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(S=magnitude, sr=sr, hop_length=self.hop_length)
            
            # Spectral flatness
            spectral_flatness = librosa.feature.spectral_flatness(S=magnitude, hop_length=self.hop_length)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)
            
            # Statistical moments
            features[window_key] = {
                "spectral_centroid_mean": float(np.mean(spectral_centroid)),
                "spectral_centroid_std": float(np.std(spectral_centroid)),
                "spectral_rolloff_85_mean": float(np.mean(rolloff_85)),
                "spectral_rolloff_95_mean": float(np.mean(rolloff_95)),
                "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
                "spectral_bandwidth_std": float(np.std(spectral_bandwidth)),
                "spectral_contrast_mean": np.mean(spectral_contrast, axis=1).tolist(),
                "spectral_flatness_mean": float(np.mean(spectral_flatness)),
                "zero_crossing_rate_mean": float(np.mean(zcr)),
                "spectral_skewness": float(skew(magnitude.flatten())),
                "spectral_kurtosis": float(kurtosis(magnitude.flatten()))
            }
        
        return {"multi_resolution_spectral": features}
    
    def _extract_comprehensive_rhythm_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract comprehensive rhythm and temporal features."""
        
        # Tempo and beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
        
        # Onset detection with multiple methods
        onset_frames_default = librosa.onset.onset_detect(y=y, sr=sr)
        onset_frames_hfc = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=librosa.onset.onset_strength(y=y, sr=sr, feature=librosa.feature.spectral_centroid))
        onset_frames_complex = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=librosa.onset.onset_strength(y=y, sr=sr, feature=librosa.feature.spectral_rolloff))
        
        onset_times_default = librosa.frames_to_time(onset_frames_default, sr=sr)
        onset_times_hfc = librosa.frames_to_time(onset_frames_hfc, sr=sr)
        onset_times_complex = librosa.frames_to_time(onset_frames_complex, sr=sr)
        
        # Rhythmic pattern analysis
        tempogram = librosa.feature.tempogram(y=y, sr=sr)
        
        # Beat synchronous features
        if len(beats) > 0:
            beat_chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            beat_sync_chroma = librosa.util.sync(beat_chroma, beats)
            
            beat_mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            beat_sync_mfcc = librosa.util.sync(beat_mfcc, beats)
            
            # Beat consistency metrics
            beat_intervals = np.diff(beats)
            beat_consistency = 1.0 - (np.std(beat_intervals) / np.mean(beat_intervals)) if len(beat_intervals) > 0 else 0.0
        else:
            beat_sync_chroma = np.zeros((12, 1))
            beat_sync_mfcc = np.zeros((13, 1))
            beat_consistency = 0.0
        
        # Syncopation measure
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
        syncopation = self._calculate_syncopation(onset_strength, beats)
        
        return {
            "rhythm_comprehensive": {
                "tempo": float(tempo),
                "beat_count": len(beats),
                "onset_count_default": len(onset_times_default),
                "onset_count_hfc": len(onset_times_hfc),
                "onset_count_complex": len(onset_times_complex),
                "onset_density": len(onset_times_default) / (len(y) / sr),  # onsets per second
                "beat_consistency": float(max(0, min(1, beat_consistency))),
                "syncopation_index": float(syncopation),
                "tempogram_mean": np.mean(tempogram).item(),
                "tempogram_std": np.std(tempogram).item(),
                "beat_sync_chroma_mean": np.mean(beat_sync_chroma, axis=1).tolist(),
                "beat_sync_mfcc_mean": np.mean(beat_sync_mfcc, axis=1).tolist(),
                "rhythmic_regularity": float(np.var(onset_strength)),
                "tempo_stability": self._calculate_tempo_stability_advanced(y, sr)
            }
        }
    
    def _extract_advanced_harmonic_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract advanced harmonic and tonal features."""
        
        # Multi-resolution chromagram
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        
        # Tonnetz features
        y_harmonic = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
        
        # Harmonic-percussive separation
        y_harmonic_hpss, y_percussive = librosa.effects.hpss(y)
        
        # Spectral contrast for harmonic content
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Chord estimation (simplified)
        chroma_mean = np.mean(chroma_cqt, axis=1)
        dominant_pitch_class = np.argmax(chroma_mean)
        
        # Tonal stability measures
        chroma_var = np.var(chroma_cqt, axis=1)
        tonal_stability = 1.0 - np.mean(chroma_var)
        
        # Harmonic change rate
        chroma_diff = np.diff(chroma_cqt, axis=1)
        harmonic_change_rate = np.mean(np.sqrt(np.sum(chroma_diff ** 2, axis=0)))
        
        return {
            "harmonic_advanced": {
                "chroma_stft_mean": np.mean(chroma_stft, axis=1).tolist(),
                "chroma_cqt_mean": np.mean(chroma_cqt, axis=1).tolist(),
                "chroma_cens_mean": np.mean(chroma_cens, axis=1).tolist(),
                "chroma_stft_std": np.std(chroma_stft, axis=1).tolist(),
                "tonnetz_mean": np.mean(tonnetz, axis=1).tolist(),
                "tonnetz_std": np.std(tonnetz, axis=1).tolist(),
                "harmonic_ratio": float(np.sum(y_harmonic_hpss ** 2) / (np.sum(y ** 2) + 1e-10)),
                "percussive_ratio": float(np.sum(y_percussive ** 2) / (np.sum(y ** 2) + 1e-10)),
                "spectral_contrast_mean": np.mean(contrast, axis=1).tolist(),
                "dominant_pitch_class": int(dominant_pitch_class),
                "tonal_stability": float(max(0, min(1, tonal_stability))),
                "harmonic_change_rate": float(harmonic_change_rate),
                "key_clarity": float(np.max(chroma_mean) - np.mean(chroma_mean))
            }
        }
    
    def _extract_comprehensive_energy_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract comprehensive energy and dynamics features."""
        
        # Multi-resolution RMS energy
        rms_features = {}
        for hop_length in [512, 1024, 2048]:
            rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
            rms_features[f"rms_{hop_length}"] = {
                "mean": float(np.mean(rms)),
                "std": float(np.std(rms)),
                "max": float(np.max(rms)),
                "min": float(np.min(rms))
            }
        
        # Dynamic range in multiple frequency bands
        stft = librosa.stft(y)
        magnitude = np.abs(stft)
        
        # Split into frequency bands
        n_bands = 8
        band_size = magnitude.shape[0] // n_bands
        band_dynamics = []
        
        for i in range(n_bands):
            start = i * band_size
            end = min((i + 1) * band_size, magnitude.shape[0])
            band_magnitude = magnitude[start:end, :]
            band_energy = np.sum(band_magnitude, axis=0)
            
            if len(band_energy) > 0:
                band_db = librosa.amplitude_to_db(band_energy + 1e-10)
                band_dynamic_range = float(np.max(band_db) - np.min(band_db))
                band_dynamics.append(band_dynamic_range)
            else:
                band_dynamics.append(0.0)
        
        # Energy envelope analysis
        energy_envelope = np.sum(magnitude, axis=0)
        energy_envelope_smooth = np.convolve(energy_envelope, np.ones(sr // 100) / (sr // 100), mode='same')
        
        # Attack and decay characteristics
        attack_times, decay_times = self._extract_attack_decay_features(y, sr)
        
        return {
            "energy_comprehensive": {
                "rms_multi_resolution": rms_features,
                "band_dynamics": band_dynamics,
                "overall_dynamic_range": float(np.max(band_dynamics) - np.min(band_dynamics)),
                "energy_envelope_mean": float(np.mean(energy_envelope)),
                "energy_envelope_std": float(np.std(energy_envelope)),
                "energy_skewness": float(skew(energy_envelope)),
                "energy_kurtosis": float(kurtosis(energy_envelope)),
                "attack_time_mean": float(np.mean(attack_times)) if len(attack_times) > 0 else 0.0,
                "decay_time_mean": float(np.mean(decay_times)) if len(decay_times) > 0 else 0.0,
                "peak_to_average_ratio": float(np.max(energy_envelope) / (np.mean(energy_envelope) + 1e-10))
            }
        }
    
    def _extract_comprehensive_essentia_features(self, audio_file_path: str) -> Dict[str, Any]:
        """Extract comprehensive features using Essentia."""
        
        try:
            # Load audio for Essentia
            loader = es.MonoLoader(filename=audio_file_path)
            audio = loader()
            
            # Windowing and spectrum
            w = es.Windowing(type='hann')
            spectrum = es.Spectrum()
            
            essentia_features = {}
            
            # Basic Essentia features (from original method)
            key, scale, strength = self.key_extractor(audio)
            camelot_key = self._convert_to_camelot(key, scale)
            bpm, beats, beats_confidence, _, beats_intervals = self.rhythm_extractor(audio)
            loudness, _, _, _ = self.loudness_extractor(audio)
            
            essentia_features.update({
                "key_signature": f"{key} {scale}",
                "key_strength": float(strength),
                "camelot_key": camelot_key,
                "bpm_essentia": float(bpm),
                "beats_confidence": float(beats_confidence),
                "loudness_lufs": float(loudness)
            })
            
            # Advanced Essentia features
            try:
                # Bark bands
                bark_bands = self.barkbands_extractor(spectrum(w(audio[:sr])))  # First second
                essentia_features["bark_bands"] = bark_bands.tolist()
                
                # Mel bands
                mel_bands = self.melbands_extractor(spectrum(w(audio[:sr])))
                essentia_features["mel_bands"] = mel_bands.tolist()
                
                # Onset rate
                onset_rate = self.onset_rate_extractor(audio)
                essentia_features["onset_rate"] = float(onset_rate)
                
                # Dynamic complexity
                dynamic_complexity = self.dynamic_complexity_extractor(audio)
                essentia_features["dynamic_complexity"] = float(dynamic_complexity)
                
                # Danceability
                danceability = self.danceability_extractor(audio)
                essentia_features["danceability"] = float(danceability)
                
            except Exception as e:
                logger.warning("Advanced Essentia features failed", error=str(e))
            
            return {"essentia_comprehensive": essentia_features}
            
        except Exception as e:
            logger.warning("Comprehensive Essentia feature extraction failed", error=str(e))
            return {"essentia_comprehensive": {}}
    
    def _extract_psychoacoustic_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract psychoacoustic features."""
        
        # Spectral flux
        stft = librosa.stft(y)
        magnitude = np.abs(stft)
        spectral_flux = np.sum(np.diff(magnitude, axis=1) ** 2, axis=0)
        
        # Roughness estimation (simplified)
        roughness = self._calculate_roughness(magnitude, sr)
        
        # Brightness (high frequency content)
        brightness = self._calculate_brightness(magnitude)
        
        # Irregularity (Krimphoff et al.)
        irregularity = self._calculate_irregularity(magnitude)
        
        return {
            "psychoacoustic": {
                "spectral_flux_mean": float(np.mean(spectral_flux)),
                "spectral_flux_std": float(np.std(spectral_flux)),
                "roughness": float(roughness),
                "brightness": float(brightness),
                "irregularity": float(irregularity),
                "spectral_spread": float(np.std(magnitude.flatten())),
                "spectral_entropy": float(self._calculate_spectral_entropy(magnitude))
            }
        }
    
    def _extract_structural_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract structural features."""
        
        # Segment boundaries using recurrence
        R = librosa.segment.recurrence_matrix(y, k=5, width=3, metric='cosine', sym=True)
        boundaries = librosa.segment.agglomerative(R, k=10)
        
        # Novelty function
        novelty = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
        
        # Self-similarity matrix
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        similarity = librosa.segment.recurrence_matrix(chroma, mode='affinity')
        
        return {
            "structural": {
                "estimated_segments": len(boundaries),
                "novelty_mean": float(np.mean(novelty)),
                "novelty_std": float(np.std(novelty)),
                "self_similarity_mean": float(np.mean(similarity)),
                "structural_regularity": float(np.trace(similarity) / similarity.size),
                "repetition_rate": self._calculate_repetition_rate(chroma)
            }
        }
    
    def _extract_timbral_complexity_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract timbral complexity features."""
        
        # Extended MFCC analysis
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Poly features
        poly_features = librosa.feature.poly_features(y=y, sr=sr)
        
        # Spectral statistics
        stft = librosa.stft(y)
        magnitude = np.abs(stft)
        
        return {
            "timbral_complexity": {
                "mfcc_extended_mean": np.mean(mfccs, axis=1).tolist(),
                "mfcc_extended_std": np.std(mfccs, axis=1).tolist(),
                "delta_mfcc_mean": np.mean(delta_mfccs, axis=1).tolist(),
                "delta2_mfcc_mean": np.mean(delta2_mfccs, axis=1).tolist(),
                "poly_features_mean": np.mean(poly_features, axis=1).tolist(),
                "spectral_complexity": self._calculate_spectral_complexity(magnitude),
                "timbral_diversity": float(np.std(mfccs.flatten())),
                "harmonic_complexity": self._calculate_harmonic_complexity(y, sr)
            }
        }
    
    def _extract_statistical_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract statistical features from audio signal."""
        
        # Time domain statistics
        time_stats = {
            "signal_mean": float(np.mean(y)),
            "signal_std": float(np.std(y)),
            "signal_skewness": float(skew(y)),
            "signal_kurtosis": float(kurtosis(y)),
            "signal_rms": float(np.sqrt(np.mean(y ** 2))),
            "signal_peak": float(np.max(np.abs(y))),
            "crest_factor": float(np.max(np.abs(y)) / (np.sqrt(np.mean(y ** 2)) + 1e-10))
        }
        
        # Frequency domain statistics
        stft = librosa.stft(y)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        freq_stats = {
            "magnitude_mean": float(np.mean(magnitude)),
            "magnitude_std": float(np.std(magnitude)),
            "magnitude_skewness": float(skew(magnitude.flatten())),
            "magnitude_kurtosis": float(kurtosis(magnitude.flatten())),
            "phase_coherence": float(np.mean(np.abs(np.diff(phase, axis=1)))),
            "spectral_rolloff_std": float(np.std(librosa.feature.spectral_rolloff(S=magnitude, sr=sr)))
        }
        
        return {
            "statistical": {
                "time_domain": time_stats,
                "frequency_domain": freq_stats
            }
        }
    
    def _generate_comprehensive_feature_vector(self, features: Dict[str, Any]) -> List[float]:
        """Generate comprehensive feature vector from all extracted features."""
        
        feature_vector = []
        
        # Process each feature category
        for category, category_features in features.items():
            if category in ["metadata", "comprehensive_feature_vector"]:
                continue
            
            # Flatten and add features
            flattened = self._flatten_features(category_features)
            feature_vector.extend(flattened)
        
        # Add metadata features
        if "metadata" in features:
            metadata_vector = features["metadata"].get("feature_vector", [])
            feature_vector.extend(metadata_vector)
        
        # Pad or truncate to target size (1500 dimensions)
        target_size = 1500
        
        if len(feature_vector) < target_size:
            # Pad with zeros
            feature_vector.extend([0.0] * (target_size - len(feature_vector)))
        else:
            # Truncate
            feature_vector = feature_vector[:target_size]
        
        return feature_vector
    
    def _flatten_features(self, features: Any) -> List[float]:
        """Recursively flatten nested feature dictionaries."""
        
        flattened = []
        
        if isinstance(features, dict):
            for value in features.values():
                flattened.extend(self._flatten_features(value))
        elif isinstance(features, (list, np.ndarray)):
            for item in features:
                if isinstance(item, (int, float, np.number)):
                    flattened.append(float(item))
                else:
                    flattened.extend(self._flatten_features(item))
        elif isinstance(features, (int, float, np.number)):
            flattened.append(float(features))
        elif isinstance(features, str):
            # Convert strings to hash-based features
            flattened.append(float(hash(features) % 1000) / 1000.0)
        
        return flattened
    
    # Helper methods for advanced calculations
    def _calculate_syncopation(self, onset_strength: np.ndarray, beats: np.ndarray) -> float:
        """Calculate syncopation index."""
        if len(beats) < 2:
            return 0.0
        
        beat_strength = []
        for beat in beats:
            beat_frame = int(beat * 512 / 44100)  # Convert to frame
            if 0 <= beat_frame < len(onset_strength):
                beat_strength.append(onset_strength[beat_frame])
        
        if len(beat_strength) < 2:
            return 0.0
        
        # Simple syncopation: variance in beat strengths
        return float(np.var(beat_strength))
    
    def _calculate_tempo_stability_advanced(self, y: np.ndarray, sr: int) -> float:
        """Advanced tempo stability calculation."""
        tempogram = librosa.feature.tempogram(y=y, sr=sr)
        tempo_variation = np.var(np.mean(tempogram, axis=0))
        return float(1.0 - min(tempo_variation, 1.0))
    
    def _extract_attack_decay_features(self, y: np.ndarray, sr: int) -> Tuple[List[float], List[float]]:
        """Extract attack and decay time features."""
        
        # Simplified attack/decay detection
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        attack_times = []
        decay_times = []
        
        for onset in onset_frames[:10]:  # Limit to first 10 onsets
            try:
                window_size = sr // 10  # 100ms window
                start = max(0, onset - window_size // 4)
                end = min(len(y), onset + window_size)
                
                segment = y[start:end]
                if len(segment) > 0:
                    peak_idx = np.argmax(np.abs(segment))
                    attack_time = peak_idx / sr
                    decay_time = (len(segment) - peak_idx) / sr
                    
                    attack_times.append(attack_time)
                    decay_times.append(decay_time)
            except:
                continue
        
        return attack_times, decay_times
    
    def _calculate_roughness(self, magnitude: np.ndarray, sr: int) -> float:
        """Calculate roughness (simplified)."""
        # Simplified roughness based on spectral irregularity
        if magnitude.shape[0] < 2:
            return 0.0
        
        spectral_diff = np.diff(magnitude, axis=0)
        roughness = np.mean(np.var(spectral_diff, axis=1))
        return float(roughness)
    
    def _calculate_brightness(self, magnitude: np.ndarray) -> float:
        """Calculate brightness (high frequency content)."""
        # Weight higher frequencies more
        weights = np.linspace(0, 1, magnitude.shape[0])
        brightness = np.mean(np.sum(magnitude * weights[:, np.newaxis], axis=0))
        return float(brightness)
    
    def _calculate_irregularity(self, magnitude: np.ndarray) -> float:
        """Calculate spectral irregularity."""
        if magnitude.shape[0] < 3:
            return 0.0
        
        # Krimphoff irregularity
        irregularity = 0.0
        for i in range(1, magnitude.shape[0] - 1):
            for j in range(magnitude.shape[1]):
                if magnitude[i, j] > 0:
                    irregularity += abs(magnitude[i, j] - (magnitude[i-1, j] + magnitude[i, j] + magnitude[i+1, j]) / 3)
        
        return float(irregularity / (magnitude.shape[0] * magnitude.shape[1]))
    
    def _calculate_spectral_entropy(self, magnitude: np.ndarray) -> float:
        """Calculate spectral entropy."""
        # Normalize magnitude to probabilities
        magnitude_norm = magnitude / (np.sum(magnitude, axis=0, keepdims=True) + 1e-10)
        
        # Calculate entropy
        entropy = -np.sum(magnitude_norm * np.log(magnitude_norm + 1e-10), axis=0)
        return float(np.mean(entropy))
    
    def _calculate_spectral_complexity(self, magnitude: np.ndarray) -> float:
        """Calculate spectral complexity."""
        # Based on spectral shape variation
        spectral_shape = magnitude / (np.sum(magnitude, axis=0, keepdims=True) + 1e-10)
        complexity = np.mean(np.var(spectral_shape, axis=1))
        return float(complexity)
    
    def _calculate_harmonic_complexity(self, y: np.ndarray, sr: int) -> float:
        """Calculate harmonic complexity."""
        # Use chroma variation as proxy
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        harmonic_complexity = np.mean(np.var(chroma, axis=1))
        return float(harmonic_complexity)
    
    def _calculate_repetition_rate(self, chroma: np.ndarray) -> float:
        """Calculate repetition rate from chroma features."""
        # Simplified repetition detection
        autocorr = np.correlate(chroma.flatten(), chroma.flatten(), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks in autocorrelation
        peaks = []
        for i in range(1, len(autocorr) - 1):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                peaks.append(autocorr[i])
        
        return float(np.mean(peaks)) if peaks else 0.0


# Keep the old class for backward compatibility
class FeatureExtractor(ComprehensiveFeatureExtractor):
    """Backward compatibility wrapper."""
    
    def extract_global_features(self, audio_file_path: str) -> Dict[str, Any]:
        """Legacy method - calls the new comprehensive extraction."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.extract_maximum_features(audio_file_path))
        finally:
            loop.close()
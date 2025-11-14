"""
AI-powered music labeling system for DJ mixing.
Uses Claude/AI to analyze music and generate training labels automatically.

Identifies:
- Song sections (verse, chorus, bridge, buildup, outro, drop)
- Mix points (optimal transition locations)  
- Similarity relationships for contrastive learning
- Musical characteristics and descriptors
"""

import asyncio
import json
import structlog
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import librosa
import soundfile as sf
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import base64
from io import BytesIO

from .feature_extractor import ComprehensiveFeatureExtractor
from .metadata_extractor import MetadataConfig

logger = structlog.get_logger(__name__)


@dataclass
class SongSection:
    """Represents a labeled section of a song."""
    
    section_type: str  # verse, chorus, bridge, buildup, outro, drop, break, intro
    start_time: float  # seconds
    end_time: float    # seconds
    confidence: float  # 0.0 to 1.0
    characteristics: Dict[str, Any] = None
    mix_suitability: str = "unknown"  # excellent, good, fair, poor
    description: str = ""

    def __post_init__(self):
        if self.characteristics is None:
            self.characteristics = {}


@dataclass  
class MixPoint:
    """Represents an optimal transition point for mixing."""
    
    time: float  # seconds
    mix_type: str  # intro, outro, break, drop, buildup_peak
    suitability: str  # excellent, good, fair
    energy_level: float  # 0.0 to 1.0
    tempo_stability: float  # 0.0 to 1.0
    harmonic_stability: float  # 0.0 to 1.0
    description: str = ""


@dataclass
class SimilarityLabel:
    """Represents similarity relationship between songs."""
    
    track_a_id: str
    track_b_id: str
    similarity_type: str  # style, energy, harmonic, rhythmic, genre, mood
    similarity_score: float  # 0.0 to 1.0
    reason: str  # explanation of similarity
    confidence: float  # 0.0 to 1.0


class MusicAnalysisPromptGenerator:
    """Generates structured prompts for AI music analysis."""
    
    def __init__(self):
        """Initialize with analysis templates."""
        
        self.section_types = [
            "intro", "verse", "chorus", "bridge", "buildup", "drop", 
            "breakdown", "outro", "instrumental", "vocal_section"
        ]
        
        self.mix_point_types = [
            "intro_point", "outro_point", "break_point", "drop_point", 
            "buildup_peak", "energy_shift", "harmonic_change"
        ]
    
    def generate_section_analysis_prompt(self, track_info: Dict[str, Any], 
                                       analysis_data: Dict[str, Any]) -> str:
        """Generate prompt for song section analysis."""
        
        # Extract key information
        title = track_info.get('title', 'Unknown')
        artist = track_info.get('artist', 'Unknown')
        genre = track_info.get('genre', 'Unknown')
        duration = track_info.get('duration', 0)
        
        # Audio analysis summary
        tempo = analysis_data.get('rhythm_comprehensive', {}).get('tempo', 120)
        key = analysis_data.get('essentia_comprehensive', {}).get('key_signature', 'Unknown')
        energy = analysis_data.get('spotify', {}).get('audio_features', {}).get('energy', 0.5)
        
        prompt = f"""
Please analyze this music track and identify its structural sections:

**Track Information:**
- Title: {title}
- Artist: {artist}
- Genre: {genre}
- Duration: {duration:.1f} seconds
- Tempo: {tempo:.1f} BPM
- Key: {key}
- Energy Level: {energy:.2f}

**Analysis Task:**
Identify and label the main sections of this song. For each section, provide:

1. **Section Type**: Choose from {', '.join(self.section_types)}
2. **Time Range**: Start and end times in seconds
3. **Characteristics**: Musical features of this section
4. **Mix Suitability**: How suitable for DJ mixing (excellent/good/fair/poor)
5. **Description**: Brief description of what happens in this section

**Consider:**
- Typical song structures for {genre} music
- Energy progression and dynamics
- Vocal vs instrumental sections
- Repetitive patterns and variations
- Transition points between sections

**Output Format:**
Return a JSON list of sections:
```json
[
  {{
    "section_type": "intro",
    "start_time": 0.0,
    "end_time": 16.0,
    "confidence": 0.9,
    "characteristics": {{"energy": "low", "vocals": false, "build": true}},
    "mix_suitability": "excellent",
    "description": "Instrumental intro with gradual energy build"
  }},
  ...
]
```

Focus on sections most relevant for DJ mixing and transitions.
"""
        
        return prompt.strip()
    
    def generate_mix_point_analysis_prompt(self, track_info: Dict[str, Any],
                                         sections: List[SongSection],
                                         analysis_data: Dict[str, Any]) -> str:
        """Generate prompt for mix point identification."""
        
        title = track_info.get('title', 'Unknown')
        artist = track_info.get('artist', 'Unknown')
        duration = track_info.get('duration', 0)
        
        # Convert sections to simple dict for prompt
        sections_summary = []
        for section in sections:
            sections_summary.append({
                'type': section.section_type,
                'time': f"{section.start_time:.1f}s - {section.end_time:.1f}s",
                'suitability': section.mix_suitability
            })
        
        prompt = f"""
Based on the structural analysis of this track, identify the optimal points for DJ mixing:

**Track:** {title} by {artist} ({duration:.1f}s)

**Identified Sections:**
{json.dumps(sections_summary, indent=2)}

**Mix Point Analysis Task:**
Identify the best points for:
1. **Mix IN** - Where another track can transition into this one
2. **Mix OUT** - Where this track can transition to another
3. **Loop POINTS** - Good sections for extending/looping
4. **CUE POINTS** - Important markers for live performance

**For each mix point, specify:**
- **Time**: Exact time in seconds
- **Mix Type**: {', '.join(self.mix_point_types)}
- **Suitability**: excellent/good/fair
- **Energy Level**: 0.0 (low) to 1.0 (high)
- **Stability Factors**: How stable the tempo/harmony is
- **Description**: Why this is a good mix point

**DJ Mixing Considerations:**
- 8-bar, 16-bar, 32-bar phrasing
- Energy matching for smooth transitions
- Harmonic compatibility points
- Rhythmic stability for beatmatching
- Vocal vs instrumental sections

**Output Format:**
```json
[
  {{
    "time": 32.0,
    "mix_type": "intro_point",
    "suitability": "excellent",
    "energy_level": 0.3,
    "tempo_stability": 0.95,
    "harmonic_stability": 0.9,
    "description": "End of intro, stable beat, low energy - perfect for mixing in"
  }},
  ...
]
```

Focus on practical mixing points a DJ would actually use.
"""
        
        return prompt.strip()
    
    def generate_similarity_analysis_prompt(self, track_a_info: Dict[str, Any],
                                          track_b_info: Dict[str, Any]) -> str:
        """Generate prompt for similarity analysis between two tracks."""
        
        def format_track_summary(info):
            return {
                'title': info.get('title', 'Unknown'),
                'artist': info.get('artist', 'Unknown'),
                'genre': info.get('genre', 'Unknown'),
                'tempo': info.get('tempo', 120),
                'key': info.get('key', 'Unknown'),
                'energy': info.get('energy', 0.5),
                'danceability': info.get('danceability', 0.5)
            }
        
        track_a = format_track_summary(track_a_info)
        track_b = format_track_summary(track_b_info)
        
        prompt = f"""
Analyze the similarity between these two music tracks for DJ mixing and recommendation:

**Track A:** {track_a['title']} by {track_a['artist']}
- Genre: {track_a['genre']}
- Tempo: {track_a['tempo']:.1f} BPM
- Key: {track_a['key']}
- Energy: {track_a['energy']:.2f}
- Danceability: {track_a['danceability']:.2f}

**Track B:** {track_b['title']} by {track_b['artist']}
- Genre: {track_b['genre']}
- Tempo: {track_b['tempo']:.1f} BPM  
- Key: {track_b['key']}
- Energy: {track_b['energy']:.2f}
- Danceability: {track_b['danceability']:.2f}

**Similarity Analysis:**
Evaluate these tracks across multiple dimensions:

1. **Style Similarity** - Genre, subgenre, musical style
2. **Energy Matching** - Energy levels and dynamics
3. **Harmonic Compatibility** - Key relationships, chord progressions
4. **Rhythmic Similarity** - Tempo, beat patterns, groove
5. **Mood/Vibe** - Emotional tone and atmosphere
6. **Mix Compatibility** - How well they would work in a DJ set

**For each similarity type, provide:**
- Similarity score (0.0 = completely different, 1.0 = very similar)
- Reason explaining the similarity/difference
- Confidence in the assessment (0.0 to 1.0)

**Output Format:**
```json
[
  {{
    "similarity_type": "style",
    "similarity_score": 0.8,
    "reason": "Both are progressive house with similar synth work and structure",
    "confidence": 0.9
  }},
  {{
    "similarity_type": "energy",
    "similarity_score": 0.6,
    "reason": "Track A is more energetic but both build to similar peak energy",
    "confidence": 0.8
  }},
  ...
]
```

Consider both technical compatibility (tempo/key matching) and subjective similarity (style/mood).
"""
        
        return prompt.strip()


class AILabelingOrchestrator:
    """
    Orchestrates AI-powered labeling of music database.
    Manages the analysis pipeline and generates training labels.
    """
    
    def __init__(self, metadata_config: Optional[MetadataConfig] = None):
        """Initialize with configuration."""
        
        self.metadata_config = metadata_config or MetadataConfig()
        self.feature_extractor = ComprehensiveFeatureExtractor(
            metadata_config=self.metadata_config
        )
        self.prompt_generator = MusicAnalysisPromptGenerator()
        
        # Cache for analysis results
        self.analysis_cache = {}
        self.labels_cache = {}
        
        logger.info("AILabelingOrchestrator initialized")
    
    async def analyze_track_sections(self, audio_file_path: str, 
                                   track_id: Optional[str] = None) -> Tuple[List[SongSection], str]:
        """
        Analyze track sections using AI.
        
        Args:
            audio_file_path: Path to audio file
            track_id: Optional track identifier
            
        Returns:
            Tuple of (sections_list, analysis_prompt_used)
        """
        
        if track_id is None:
            track_id = Path(audio_file_path).stem
        
        logger.info("Analyzing track sections", track_id=track_id)
        
        try:
            # Check cache first
            cache_key = f"sections_{track_id}"
            if cache_key in self.labels_cache:
                return self.labels_cache[cache_key], "cached"
            
            # Extract comprehensive features
            features = await self.feature_extractor.extract_maximum_features(audio_file_path)
            
            # Prepare track information
            track_info = self._extract_track_info(features, audio_file_path)
            
            # Generate analysis prompt
            prompt = self.prompt_generator.generate_section_analysis_prompt(
                track_info, features
            )
            
            logger.info("Generated section analysis prompt", 
                       track_id=track_id,
                       prompt_length=len(prompt))
            
            # Here you would call your AI service (Claude, GPT, etc.)
            # For now, we'll create a mock response structure
            sections = self._generate_mock_sections(track_info, features)
            
            # Cache results
            self.labels_cache[cache_key] = sections
            
            return sections, prompt
            
        except Exception as e:
            logger.error("Track section analysis failed", 
                        track_id=track_id,
                        error=str(e))
            raise
    
    async def analyze_mix_points(self, audio_file_path: str,
                               sections: List[SongSection],
                               track_id: Optional[str] = None) -> Tuple[List[MixPoint], str]:
        """
        Analyze optimal mix points using AI.
        
        Args:
            audio_file_path: Path to audio file
            sections: Previously identified sections
            track_id: Optional track identifier
            
        Returns:
            Tuple of (mix_points_list, analysis_prompt_used)
        """
        
        if track_id is None:
            track_id = Path(audio_file_path).stem
        
        logger.info("Analyzing mix points", track_id=track_id)
        
        try:
            # Extract features if not cached
            if track_id not in self.analysis_cache:
                features = await self.feature_extractor.extract_maximum_features(audio_file_path)
                self.analysis_cache[track_id] = features
            else:
                features = self.analysis_cache[track_id]
            
            # Prepare track information
            track_info = self._extract_track_info(features, audio_file_path)
            
            # Generate mix point analysis prompt
            prompt = self.prompt_generator.generate_mix_point_analysis_prompt(
                track_info, sections, features
            )
            
            # Generate mock mix points (replace with actual AI call)
            mix_points = self._generate_mock_mix_points(track_info, sections)
            
            return mix_points, prompt
            
        except Exception as e:
            logger.error("Mix point analysis failed", 
                        track_id=track_id,
                        error=str(e))
            raise
    
    async def analyze_track_similarity(self, track_a_path: str, track_b_path: str,
                                     track_a_id: Optional[str] = None,
                                     track_b_id: Optional[str] = None) -> Tuple[List[SimilarityLabel], str]:
        """
        Analyze similarity between two tracks using AI.
        
        Args:
            track_a_path: Path to first track
            track_b_path: Path to second track
            track_a_id: Optional first track ID
            track_b_id: Optional second track ID
            
        Returns:
            Tuple of (similarity_labels, analysis_prompt_used)
        """
        
        if track_a_id is None:
            track_a_id = Path(track_a_path).stem
        if track_b_id is None:
            track_b_id = Path(track_b_path).stem
        
        logger.info("Analyzing track similarity", 
                   track_a=track_a_id, track_b=track_b_id)
        
        try:
            # Extract features for both tracks
            features_a = await self.feature_extractor.extract_maximum_features(track_a_path)
            features_b = await self.feature_extractor.extract_maximum_features(track_b_path)
            
            # Prepare track information
            track_a_info = self._extract_track_info(features_a, track_a_path)
            track_b_info = self._extract_track_info(features_b, track_b_path)
            
            # Generate similarity analysis prompt
            prompt = self.prompt_generator.generate_similarity_analysis_prompt(
                track_a_info, track_b_info
            )
            
            # Generate mock similarity labels (replace with actual AI call)
            similarity_labels = self._generate_mock_similarity_labels(
                track_a_id, track_b_id, track_a_info, track_b_info
            )
            
            return similarity_labels, prompt
            
        except Exception as e:
            logger.error("Similarity analysis failed", 
                        track_a=track_a_id, track_b=track_b_id,
                        error=str(e))
            raise
    
    async def batch_analyze_database(self, audio_files: List[str],
                                   output_dir: str = "ai_labels") -> Dict[str, Any]:
        """
        Analyze entire music database and generate training labels.
        
        Args:
            audio_files: List of audio file paths
            output_dir: Directory to save results
            
        Returns:
            Summary of analysis results
        """
        
        logger.info("Starting batch database analysis", 
                   total_files=len(audio_files))
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = {
            'tracks_analyzed': 0,
            'sections_identified': 0,
            'mix_points_found': 0,
            'similarity_pairs': 0,
            'analysis_timestamp': datetime.now().isoformat(),
            'track_data': {}
        }
        
        # Process each track
        for i, audio_file in enumerate(audio_files):
            try:
                track_id = Path(audio_file).stem
                logger.info("Processing track", 
                           track_id=track_id, 
                           progress=f"{i+1}/{len(audio_files)}")
                
                # Analyze sections
                sections, sections_prompt = await self.analyze_track_sections(
                    audio_file, track_id
                )
                
                # Analyze mix points
                mix_points, mix_points_prompt = await self.analyze_mix_points(
                    audio_file, sections, track_id
                )
                
                # Store track data
                results['track_data'][track_id] = {
                    'file_path': audio_file,
                    'sections': [asdict(section) for section in sections],
                    'mix_points': [asdict(point) for point in mix_points],
                    'sections_prompt': sections_prompt,
                    'mix_points_prompt': mix_points_prompt
                }
                
                results['tracks_analyzed'] += 1
                results['sections_identified'] += len(sections)
                results['mix_points_found'] += len(mix_points)
                
                # Save intermediate results every 10 tracks
                if (i + 1) % 10 == 0:
                    self._save_results(results, output_path / "batch_analysis.json")
                
            except Exception as e:
                logger.error("Failed to process track", 
                           audio_file=audio_file, error=str(e))
                continue
        
        # Analyze similarities between tracks (sample pairs)
        logger.info("Analyzing track similarities")
        
        # Sample pairs for similarity analysis (don't do all combinations)
        import random
        track_ids = list(results['track_data'].keys())
        
        if len(track_ids) > 1:
            # Sample up to 100 pairs
            max_pairs = min(100, len(track_ids) * (len(track_ids) - 1) // 2)
            sampled_pairs = random.sample(
                [(i, j) for i in range(len(track_ids)) 
                 for j in range(i+1, len(track_ids))], 
                max_pairs
            )
            
            similarity_data = {}
            
            for i, j in sampled_pairs[:20]:  # Limit to 20 for demo
                track_a_id = track_ids[i]
                track_b_id = track_ids[j]
                
                track_a_path = results['track_data'][track_a_id]['file_path']
                track_b_path = results['track_data'][track_b_id]['file_path']
                
                try:
                    similarities, similarity_prompt = await self.analyze_track_similarity(
                        track_a_path, track_b_path, track_a_id, track_b_id
                    )
                    
                    pair_key = f"{track_a_id}_{track_b_id}"
                    similarity_data[pair_key] = {
                        'similarities': [asdict(sim) for sim in similarities],
                        'prompt': similarity_prompt
                    }
                    
                    results['similarity_pairs'] += 1
                    
                except Exception as e:
                    logger.error("Similarity analysis failed", 
                               track_a=track_a_id, track_b=track_b_id,
                               error=str(e))
            
            results['similarity_data'] = similarity_data
        
        # Save final results
        self._save_results(results, output_path / "batch_analysis_final.json")
        
        logger.info("Batch analysis completed", 
                   tracks_analyzed=results['tracks_analyzed'],
                   sections_found=results['sections_identified'],
                   mix_points_found=results['mix_points_found'],
                   similarity_pairs=results['similarity_pairs'])
        
        return results
    
    def _extract_track_info(self, features: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """Extract essential track information from features."""
        
        file_tags = features.get('metadata', {}).get('file_tags', {})
        spotify_data = features.get('metadata', {}).get('spotify', {})
        
        track_info = {
            'title': file_tags.get('title', Path(file_path).stem),
            'artist': file_tags.get('artist', 'Unknown Artist'),
            'genre': file_tags.get('genre', 'Unknown'),
            'duration': features.get('duration', 0),
            'tempo': features.get('rhythm_comprehensive', {}).get('tempo', 120),
            'key': features.get('essentia_comprehensive', {}).get('key_signature', 'Unknown'),
            'energy': spotify_data.get('audio_features', {}).get('energy', 0.5),
            'danceability': spotify_data.get('audio_features', {}).get('danceability', 0.5),
            'valence': spotify_data.get('audio_features', {}).get('valence', 0.5)
        }
        
        return track_info
    
    def _generate_mock_sections(self, track_info: Dict[str, Any], 
                              features: Dict[str, Any]) -> List[SongSection]:
        """Generate mock sections for demonstration (replace with AI analysis)."""
        
        duration = track_info.get('duration', 180)  # Default 3 minutes
        
        # Basic song structure estimation based on duration and genre
        sections = []
        
        # Intro
        sections.append(SongSection(
            section_type="intro",
            start_time=0.0,
            end_time=min(16.0, duration * 0.1),
            confidence=0.8,
            characteristics={"energy": "low", "vocals": False, "build": True},
            mix_suitability="excellent",
            description="Instrumental intro with gradual energy build"
        ))
        
        # Verse 1
        verse1_start = sections[-1].end_time
        verse1_end = min(verse1_start + 32.0, duration * 0.3)
        sections.append(SongSection(
            section_type="verse",
            start_time=verse1_start,
            end_time=verse1_end,
            confidence=0.9,
            characteristics={"energy": "medium", "vocals": True, "stable": True},
            mix_suitability="good",
            description="First verse with vocals"
        ))
        
        # Chorus
        chorus_start = verse1_end
        chorus_end = min(chorus_start + 32.0, duration * 0.5)
        sections.append(SongSection(
            section_type="chorus",
            start_time=chorus_start,
            end_time=chorus_end,
            confidence=0.95,
            characteristics={"energy": "high", "vocals": True, "catchy": True},
            mix_suitability="fair",
            description="Main chorus with high energy"
        ))
        
        # Outro
        if duration > 120:  # If song is longer than 2 minutes
            outro_start = duration - 16.0
            sections.append(SongSection(
                section_type="outro",
                start_time=outro_start,
                end_time=duration,
                confidence=0.8,
                characteristics={"energy": "decreasing", "fade": True},
                mix_suitability="excellent",
                description="Outro section with energy fade"
            ))
        
        return sections
    
    def _generate_mock_mix_points(self, track_info: Dict[str, Any],
                                sections: List[SongSection]) -> List[MixPoint]:
        """Generate mock mix points for demonstration."""
        
        mix_points = []
        
        for section in sections:
            if section.section_type == "intro" and section.mix_suitability == "excellent":
                # Good mix-in point at end of intro
                mix_points.append(MixPoint(
                    time=section.end_time,
                    mix_type="intro_point",
                    suitability="excellent",
                    energy_level=0.3,
                    tempo_stability=0.95,
                    harmonic_stability=0.9,
                    description="End of intro - perfect for mixing in"
                ))
            
            elif section.section_type == "outro" and section.mix_suitability == "excellent":
                # Good mix-out point at start of outro
                mix_points.append(MixPoint(
                    time=section.start_time,
                    mix_type="outro_point", 
                    suitability="excellent",
                    energy_level=0.7,
                    tempo_stability=0.9,
                    harmonic_stability=0.85,
                    description="Start of outro - good for mixing out"
                ))
        
        return mix_points
    
    def _generate_mock_similarity_labels(self, track_a_id: str, track_b_id: str,
                                       track_a_info: Dict[str, Any],
                                       track_b_info: Dict[str, Any]) -> List[SimilarityLabel]:
        """Generate mock similarity labels for demonstration."""
        
        labels = []
        
        # Tempo similarity
        tempo_diff = abs(track_a_info.get('tempo', 120) - track_b_info.get('tempo', 120))
        tempo_similarity = max(0.0, 1.0 - tempo_diff / 40.0)  # 40 BPM = 0 similarity
        
        labels.append(SimilarityLabel(
            track_a_id=track_a_id,
            track_b_id=track_b_id,
            similarity_type="rhythmic",
            similarity_score=tempo_similarity,
            reason=f"Tempo difference: {tempo_diff:.1f} BPM",
            confidence=0.9
        ))
        
        # Energy similarity
        energy_diff = abs(track_a_info.get('energy', 0.5) - track_b_info.get('energy', 0.5))
        energy_similarity = 1.0 - energy_diff
        
        labels.append(SimilarityLabel(
            track_a_id=track_a_id,
            track_b_id=track_b_id,
            similarity_type="energy",
            similarity_score=energy_similarity,
            reason=f"Energy difference: {energy_diff:.2f}",
            confidence=0.8
        ))
        
        # Genre similarity (simple string matching)
        genre_a = track_a_info.get('genre', '').lower()
        genre_b = track_b_info.get('genre', '').lower()
        
        if genre_a == genre_b:
            genre_similarity = 0.9
        elif any(word in genre_b for word in genre_a.split()) or any(word in genre_a for word in genre_b.split()):
            genre_similarity = 0.6
        else:
            genre_similarity = 0.2
        
        labels.append(SimilarityLabel(
            track_a_id=track_a_id,
            track_b_id=track_b_id,
            similarity_type="style",
            similarity_score=genre_similarity,
            reason=f"Genre comparison: {genre_a} vs {genre_b}",
            confidence=0.7
        ))
        
        return labels
    
    def _save_results(self, results: Dict[str, Any], output_file: Path):
        """Save analysis results to JSON file."""
        
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info("Results saved", output_file=str(output_file))
            
        except Exception as e:
            logger.error("Failed to save results", 
                        output_file=str(output_file), 
                        error=str(e))


# Convenience functions for quick labeling
async def label_track_sections(audio_file_path: str) -> List[SongSection]:
    """Quick function to label track sections."""
    
    labeler = AILabelingOrchestrator()
    sections, _ = await labeler.analyze_track_sections(audio_file_path)
    return sections


async def label_mix_points(audio_file_path: str, sections: List[SongSection]) -> List[MixPoint]:
    """Quick function to identify mix points."""
    
    labeler = AILabelingOrchestrator()
    mix_points, _ = await labeler.analyze_mix_points(audio_file_path, sections)
    return mix_points


async def generate_similarity_labels(track_pairs: List[Tuple[str, str]]) -> List[SimilarityLabel]:
    """Generate similarity labels for track pairs."""
    
    labeler = AILabelingOrchestrator()
    all_labels = []
    
    for track_a, track_b in track_pairs:
        labels, _ = await labeler.analyze_track_similarity(track_a, track_b)
        all_labels.extend(labels)
    
    return all_labels
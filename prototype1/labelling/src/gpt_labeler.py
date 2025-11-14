"""
GPT-4 Integration for Music Labeling.
Sends structured prompts to OpenAI API and processes responses.
"""

import json
import asyncio
import structlog
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import openai
from openai import AsyncOpenAI
from pathlib import Path

from .music_analyzer import TrackAnalysis

logger = structlog.get_logger(__name__)


@dataclass
class SongSection:
    """Represents a labeled section of a song."""
    
    section_type: str  # verse, chorus, bridge, buildup, outro, drop, break, intro
    start_time: float  # seconds
    end_time: float    # seconds
    confidence: float  # 0.0 to 1.0
    energy_level: str  # low, medium, high
    has_vocals: bool
    mix_suitability: str  # excellent, good, fair, poor
    description: str


@dataclass  
class MixPoint:
    """Represents an optimal transition point for mixing."""
    
    time: float  # seconds
    mix_type: str  # intro, outro, break, drop, buildup_peak
    suitability: str  # excellent, good, fair
    energy_description: str
    why_good_for_mixing: str


class GPTMusicLabeler:
    """
    Uses GPT-4 to analyze music and generate labels for DJ mixing.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        Initialize GPT labeler.
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use (gpt-4, gpt-3.5-turbo)
        """
        
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        
        logger.info("GPTMusicLabeler initialized", model=model)
    
    def _create_section_analysis_prompt(self, analysis: TrackAnalysis) -> str:
        """Create prompt for analyzing song sections."""
        
        # Summarize key analysis points
        energy_summary = self._summarize_energy_curve(analysis.energy_curve, analysis.duration)
        beats_per_section = len(analysis.beat_times) // analysis.estimated_sections if analysis.estimated_sections > 0 else 0
        
        prompt = f'''Analyze this music track and identify its structural sections for DJ mixing:

**TRACK INFO:**
- Title: {analysis.title}
- Duration: {analysis.duration:.1f} seconds
- Tempo: {analysis.tempo:.1f} BPM  
- Key: {analysis.key}
- Estimated Sections: {analysis.estimated_sections}

**AUDIO ANALYSIS:**
- Energy Pattern: {energy_summary}
- Total Beats: {len(analysis.beat_times)}
- Average Beats per Section: {beats_per_section}
- Onsets Detected: {len(analysis.onset_times)}

**YOUR TASK:**
Identify the main structural sections of this song. For each section, determine:

1. **Section Type**: intro, verse, chorus, bridge, buildup, drop, breakdown, outro
2. **Time Range**: Start and end times in seconds (0 to {analysis.duration:.1f})
3. **Energy Level**: low, medium, high
4. **Vocals**: true/false - does this section have vocals?
5. **Mix Suitability**: How good is this section for DJ mixing?
   - excellent: Perfect for transitions (stable beat, clear energy)
   - good: Usable for mixing with some care
   - fair: Challenging but possible
   - poor: Avoid mixing during this section
6. **Description**: Brief explanation of what happens musically

**MIXING CONTEXT:**
DJs need sections with:
- Stable beat patterns (no tempo changes)
- Clear energy levels (not chaotic)
- Predictable structure (8, 16, 32 bar phrases)
- Minimal vocal interference for smooth transitions

**OUTPUT FORMAT:**
Return ONLY a JSON array of sections:
```json
[
  {{
    "section_type": "intro",
    "start_time": 0.0,
    "end_time": 16.0,
    "confidence": 0.9,
    "energy_level": "low", 
    "has_vocals": false,
    "mix_suitability": "excellent",
    "description": "Instrumental intro with steady beat, perfect for mixing in"
  }},
  {{
    "section_type": "verse",
    "start_time": 16.0,
    "end_time": 48.0,
    "confidence": 0.85,
    "energy_level": "medium",
    "has_vocals": true, 
    "mix_suitability": "good",
    "description": "First verse with vocals, stable rhythm"
  }}
]
```

Focus on practical sections a DJ would use. Aim for 3-6 main sections.'''

        return prompt
    
    def _create_mix_points_prompt(self, analysis: TrackAnalysis, sections: List[SongSection]) -> str:
        """Create prompt for identifying optimal mix points."""
        
        # Convert sections to simple summary
        sections_summary = []
        for section in sections:
            sections_summary.append({
                'type': section.section_type,
                'time': f"{section.start_time:.1f}-{section.end_time:.1f}s",
                'energy': section.energy_level,
                'vocals': section.has_vocals,
                'mix_suitability': section.mix_suitability
            })
        
        prompt = f'''Based on the structural analysis, identify the BEST points for DJ mixing transitions:

**TRACK:** {analysis.title} ({analysis.duration:.1f}s, {analysis.tempo:.1f} BPM)

**IDENTIFIED SECTIONS:**
{json.dumps(sections_summary, indent=2)}

**YOUR TASK:**
Find the optimal points where a DJ can:
1. **MIX IN** to this track (from another song)
2. **MIX OUT** of this track (to another song)  
3. **LOOP** or extend sections

**MIXING REQUIREMENTS:**
- Points should be on musical boundaries (4, 8, 16, 32 bars)
- Stable tempo and rhythm
- Clear harmonic structure
- Appropriate energy level for transitions

**FOR EACH MIX POINT PROVIDE:**
1. **Time**: Exact time in seconds
2. **Mix Type**: 
   - "intro" - Good point to mix into this track
   - "outro" - Good point to mix out of this track  
   - "break" - Breakdown/pause good for transitions
   - "drop" - Energy drop good for dramatic mixing
   - "buildup_peak" - Peak energy good for energy matching
3. **Suitability**: excellent, good, fair
4. **Energy Description**: What's the energy like at this point?
5. **Why Good for Mixing**: Explain why this works for DJs

**OUTPUT FORMAT:**
Return ONLY a JSON array:
```json
[
  {{
    "time": 16.0,
    "mix_type": "intro", 
    "suitability": "excellent",
    "energy_description": "Low energy, steady beat established",
    "why_good_for_mixing": "End of intro - beat is locked, no vocals, perfect entry point"
  }},
  {{
    "time": 144.0,
    "mix_type": "outro",
    "suitability": "good", 
    "energy_description": "Medium energy starting to decrease",
    "why_good_for_mixing": "Start of outro section - good time to bring in next track"
  }}
]
```

Focus on the 2-4 MOST practical mixing points a real DJ would use.'''

        return prompt
    
    async def analyze_sections(self, analysis: TrackAnalysis) -> List[SongSection]:
        """
        Analyze track sections using GPT.
        
        Args:
            analysis: TrackAnalysis object
            
        Returns:
            List of identified sections
        """
        
        logger.info("Analyzing sections with GPT", title=analysis.title)
        
        try:
            # Create prompt
            prompt = self._create_section_analysis_prompt(analysis)
            
            # Call GPT-4
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert DJ and music analyst. Analyze music tracks and identify sections useful for DJ mixing."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_tokens=2000
            )
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                logger.error("No JSON found in GPT response")
                return []
            
            json_text = response_text[json_start:json_end]
            sections_data = json.loads(json_text)
            
            # Convert to SongSection objects
            sections = []
            for section_data in sections_data:
                section = SongSection(
                    section_type=section_data.get("section_type", "unknown"),
                    start_time=float(section_data.get("start_time", 0)),
                    end_time=float(section_data.get("end_time", 0)),
                    confidence=float(section_data.get("confidence", 0.5)),
                    energy_level=section_data.get("energy_level", "medium"),
                    has_vocals=bool(section_data.get("has_vocals", False)),
                    mix_suitability=section_data.get("mix_suitability", "fair"),
                    description=section_data.get("description", "")
                )
                sections.append(section)
            
            logger.info("Sections analyzed", 
                       title=analysis.title,
                       sections_found=len(sections))
            
            return sections
            
        except json.JSONDecodeError as e:
            logger.error("Failed to parse GPT JSON response", 
                        title=analysis.title, 
                        error=str(e),
                        response=response_text[:200] + "..." if len(response_text) > 200 else response_text)
            return []
            
        except Exception as e:
            logger.error("Section analysis failed", 
                        title=analysis.title, 
                        error=str(e))
            return []
    
    async def analyze_mix_points(self, analysis: TrackAnalysis, 
                               sections: List[SongSection]) -> List[MixPoint]:
        """
        Analyze mix points using GPT.
        
        Args:
            analysis: TrackAnalysis object
            sections: Previously identified sections
            
        Returns:
            List of optimal mix points
        """
        
        logger.info("Analyzing mix points with GPT", title=analysis.title)
        
        try:
            # Create prompt
            prompt = self._create_mix_points_prompt(analysis, sections)
            
            # Call GPT-4
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert DJ with years of mixing experience. Identify the best transition points for seamless DJ mixing."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.2,  # Very low temperature for consistent practical advice
                max_tokens=1500
            )
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                logger.error("No JSON found in mix points response")
                return []
            
            json_text = response_text[json_start:json_end]
            points_data = json.loads(json_text)
            
            # Convert to MixPoint objects
            mix_points = []
            for point_data in points_data:
                mix_point = MixPoint(
                    time=float(point_data.get("time", 0)),
                    mix_type=point_data.get("mix_type", "unknown"),
                    suitability=point_data.get("suitability", "fair"),
                    energy_description=point_data.get("energy_description", ""),
                    why_good_for_mixing=point_data.get("why_good_for_mixing", "")
                )
                mix_points.append(mix_point)
            
            logger.info("Mix points analyzed", 
                       title=analysis.title,
                       mix_points_found=len(mix_points))
            
            return mix_points
            
        except json.JSONDecodeError as e:
            logger.error("Failed to parse mix points JSON response", 
                        title=analysis.title, 
                        error=str(e))
            return []
            
        except Exception as e:
            logger.error("Mix points analysis failed", 
                        title=analysis.title, 
                        error=str(e))
            return []
    
    def _summarize_energy_curve(self, energy_curve: List[float], duration: float) -> str:
        """Create a text summary of the energy curve."""
        
        if len(energy_curve) < 3:
            return "insufficient data"
        
        # Find energy patterns
        start_energy = np.mean(energy_curve[:3]) if len(energy_curve) >= 3 else energy_curve[0]
        middle_energy = np.mean(energy_curve[len(energy_curve)//3:2*len(energy_curve)//3])
        end_energy = np.mean(energy_curve[-3:]) if len(energy_curve) >= 3 else energy_curve[-1]
        
        max_energy = max(energy_curve)
        min_energy = min(energy_curve)
        
        # Describe pattern
        if middle_energy > start_energy * 1.3 and middle_energy > end_energy * 1.2:
            pattern = "builds to peak in middle then decreases"
        elif end_energy > start_energy * 1.3:
            pattern = "builds energy throughout"
        elif start_energy > end_energy * 1.3:
            pattern = "starts high then decreases"
        else:
            pattern = "relatively stable energy"
        
        return f"{pattern} (range: {min_energy:.3f}-{max_energy:.3f})"
    
    async def process_track_complete(self, analysis: TrackAnalysis) -> Dict[str, Any]:
        """
        Complete analysis of a track - sections and mix points.
        
        Args:
            analysis: TrackAnalysis object
            
        Returns:
            Dictionary with complete labeling results
        """
        
        logger.info("Starting complete track analysis", title=analysis.title)
        
        # Analyze sections first
        sections = await self.analyze_sections(analysis)
        
        # Then analyze mix points based on sections
        mix_points = await self.analyze_mix_points(analysis, sections)
        
        # Combine results
        result = {
            "track_info": {
                "title": analysis.title,
                "artist": analysis.artist,
                "duration": analysis.duration,
                "tempo": analysis.tempo,
                "key": analysis.key,
                "file_path": analysis.file_path
            },
            "sections": [asdict(section) for section in sections],
            "mix_points": [asdict(point) for point in mix_points],
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        logger.info("Complete track analysis finished",
                   title=analysis.title,
                   sections_found=len(sections),
                   mix_points_found=len(mix_points))
        
        return result


# Import numpy for energy calculations
import numpy as np
from datetime import datetime
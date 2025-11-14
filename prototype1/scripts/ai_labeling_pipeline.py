#!/usr/bin/env python3
"""
AI Music Labeling Pipeline Script

This script processes a music library and generates AI-powered labels for:
- Song sections (verse, chorus, buildup, outro, etc.)  
- Mix points (optimal transition locations)
- Similarity relationships for training

Usage:
    python ai_labeling_pipeline.py /path/to/music/library /path/to/output

The script will:
1. Scan the music library for audio files
2. Extract comprehensive features for each track
3. Generate structured prompts for AI analysis
4. Create training labels for ML models
5. Export results in JSON format for training

For actual AI analysis, connect to Claude/GPT API by implementing the 
`call_ai_service` function with your preferred AI service.
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import structlog

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared_libs.audio_processing.ai_music_labeler import (
    AILabelingOrchestrator,
    SongSection, 
    MixPoint,
    SimilarityLabel
)
from shared_libs.audio_processing.metadata_extractor import MetadataConfig

logger = structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)


class AILabelingPipeline:
    """Main pipeline for AI-powered music labeling."""
    
    def __init__(self, music_library_path: str, output_path: str,
                 api_key: str = None, ai_service: str = "claude"):
        """
        Initialize labeling pipeline.
        
        Args:
            music_library_path: Path to directory containing music files
            output_path: Path to save results
            api_key: API key for AI service  
            ai_service: AI service to use ("claude", "openai", "mock")
        """
        
        self.music_library_path = Path(music_library_path)
        self.output_path = Path(output_path)
        self.api_key = api_key
        self.ai_service = ai_service
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Configure metadata extraction (add your API keys here)
        metadata_config = MetadataConfig(
            # spotify_client_id="your_spotify_client_id",
            # spotify_client_secret="your_spotify_client_secret", 
            # lastfm_api_key="your_lastfm_api_key"
        )
        
        self.orchestrator = AILabelingOrchestrator(metadata_config)
        
        logger.info("Pipeline initialized",
                   library_path=str(self.music_library_path),
                   output_path=str(self.output_path),
                   ai_service=ai_service)
    
    def scan_music_library(self) -> List[str]:
        """Scan music library and return list of audio files."""
        
        audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg'}
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(self.music_library_path.glob(f"**/*{ext}"))
            audio_files.extend(self.music_library_path.glob(f"**/*{ext.upper()}"))
        
        audio_files = [str(f) for f in audio_files]
        
        logger.info("Music library scanned", 
                   total_files=len(audio_files),
                   extensions=list(audio_extensions))
        
        return audio_files
    
    async def call_ai_service(self, prompt: str, analysis_type: str) -> Dict[str, Any]:
        """
        Call AI service for analysis.
        
        IMPLEMENT THIS METHOD with your preferred AI service:
        - Claude API (Anthropic)
        - OpenAI GPT API  
        - Local AI model
        - Other AI services
        
        Args:
            prompt: Analysis prompt
            analysis_type: Type of analysis (sections, mix_points, similarity)
            
        Returns:
            Parsed AI response as dictionary
        """
        
        if self.ai_service == "mock":
            return await self._mock_ai_response(prompt, analysis_type)
        
        elif self.ai_service == "claude":
            # TODO: Implement Claude API call
            # return await self._call_claude_api(prompt)
            return await self._mock_ai_response(prompt, analysis_type)
        
        elif self.ai_service == "openai":
            # TODO: Implement OpenAI API call  
            # return await self._call_openai_api(prompt)
            return await self._mock_ai_response(prompt, analysis_type)
        
        else:
            raise ValueError(f"Unsupported AI service: {self.ai_service}")
    
    async def _mock_ai_response(self, prompt: str, analysis_type: str) -> Dict[str, Any]:
        """Generate mock AI response for testing."""
        
        await asyncio.sleep(0.1)  # Simulate API delay
        
        if analysis_type == "sections":
            return {
                "sections": [
                    {
                        "section_type": "intro",
                        "start_time": 0.0,
                        "end_time": 16.0,
                        "confidence": 0.9,
                        "characteristics": {"energy": "low", "vocals": False},
                        "mix_suitability": "excellent",
                        "description": "Instrumental intro with build"
                    },
                    {
                        "section_type": "verse",
                        "start_time": 16.0,
                        "end_time": 48.0,
                        "confidence": 0.85,
                        "characteristics": {"energy": "medium", "vocals": True},
                        "mix_suitability": "good", 
                        "description": "First verse with vocals"
                    },
                    {
                        "section_type": "chorus",
                        "start_time": 48.0,
                        "end_time": 80.0,
                        "confidence": 0.95,
                        "characteristics": {"energy": "high", "vocals": True},
                        "mix_suitability": "fair",
                        "description": "Main chorus section"
                    },
                    {
                        "section_type": "outro",
                        "start_time": 160.0,
                        "end_time": 180.0,
                        "confidence": 0.8,
                        "characteristics": {"energy": "decreasing", "fade": True},
                        "mix_suitability": "excellent",
                        "description": "Outro with fade"
                    }
                ]
            }
        
        elif analysis_type == "mix_points":
            return {
                "mix_points": [
                    {
                        "time": 16.0,
                        "mix_type": "intro_point",
                        "suitability": "excellent",
                        "energy_level": 0.3,
                        "tempo_stability": 0.95,
                        "harmonic_stability": 0.9,
                        "description": "Perfect intro mix point"
                    },
                    {
                        "time": 160.0,
                        "mix_type": "outro_point",
                        "suitability": "excellent", 
                        "energy_level": 0.7,
                        "tempo_stability": 0.85,
                        "harmonic_stability": 0.8,
                        "description": "Good outro transition point"
                    }
                ]
            }
        
        elif analysis_type == "similarity":
            return {
                "similarities": [
                    {
                        "similarity_type": "rhythmic",
                        "similarity_score": 0.8,
                        "reason": "Similar tempo and beat patterns",
                        "confidence": 0.9
                    },
                    {
                        "similarity_type": "energy",
                        "similarity_score": 0.6,
                        "reason": "Both have high energy but different dynamics",
                        "confidence": 0.8
                    },
                    {
                        "similarity_type": "style",
                        "similarity_score": 0.7,
                        "reason": "Same genre with similar production style",
                        "confidence": 0.85
                    }
                ]
            }
        
        return {}
    
    async def process_track_sections(self, audio_file: str, track_id: str) -> List[SongSection]:
        """Process a single track for section analysis."""
        
        try:
            # Get analysis prompt
            _, prompt = await self.orchestrator.analyze_track_sections(audio_file, track_id)
            
            # Call AI service
            ai_response = await self.call_ai_service(prompt, "sections")
            
            # Parse response into SongSection objects
            sections = []
            
            for section_data in ai_response.get("sections", []):
                section = SongSection(
                    section_type=section_data.get("section_type", "unknown"),
                    start_time=float(section_data.get("start_time", 0)),
                    end_time=float(section_data.get("end_time", 0)),
                    confidence=float(section_data.get("confidence", 0.5)),
                    characteristics=section_data.get("characteristics", {}),
                    mix_suitability=section_data.get("mix_suitability", "unknown"),
                    description=section_data.get("description", "")
                )
                sections.append(section)
            
            logger.info("Track sections processed", 
                       track_id=track_id, 
                       sections_found=len(sections))
            
            return sections
            
        except Exception as e:
            logger.error("Section processing failed", 
                        track_id=track_id, 
                        error=str(e))
            return []
    
    async def process_track_mix_points(self, audio_file: str, track_id: str, 
                                     sections: List[SongSection]) -> List[MixPoint]:
        """Process a single track for mix point analysis."""
        
        try:
            # Get analysis prompt
            _, prompt = await self.orchestrator.analyze_mix_points(
                audio_file, sections, track_id
            )
            
            # Call AI service
            ai_response = await self.call_ai_service(prompt, "mix_points")
            
            # Parse response into MixPoint objects
            mix_points = []
            
            for point_data in ai_response.get("mix_points", []):
                mix_point = MixPoint(
                    time=float(point_data.get("time", 0)),
                    mix_type=point_data.get("mix_type", "unknown"),
                    suitability=point_data.get("suitability", "fair"),
                    energy_level=float(point_data.get("energy_level", 0.5)),
                    tempo_stability=float(point_data.get("tempo_stability", 0.5)),
                    harmonic_stability=float(point_data.get("harmonic_stability", 0.5)),
                    description=point_data.get("description", "")
                )
                mix_points.append(mix_point)
            
            logger.info("Mix points processed", 
                       track_id=track_id, 
                       mix_points_found=len(mix_points))
            
            return mix_points
            
        except Exception as e:
            logger.error("Mix point processing failed", 
                        track_id=track_id, 
                        error=str(e))
            return []
    
    async def process_similarity_batch(self, track_pairs: List[tuple]) -> List[SimilarityLabel]:
        """Process a batch of track pairs for similarity analysis."""
        
        all_similarities = []
        
        for track_a_path, track_b_path in track_pairs:
            track_a_id = Path(track_a_path).stem
            track_b_id = Path(track_b_path).stem
            
            try:
                # Get analysis prompt
                _, prompt = await self.orchestrator.analyze_track_similarity(
                    track_a_path, track_b_path, track_a_id, track_b_id
                )
                
                # Call AI service
                ai_response = await self.call_ai_service(prompt, "similarity")
                
                # Parse response into SimilarityLabel objects
                for sim_data in ai_response.get("similarities", []):
                    similarity = SimilarityLabel(
                        track_a_id=track_a_id,
                        track_b_id=track_b_id,
                        similarity_type=sim_data.get("similarity_type", "unknown"),
                        similarity_score=float(sim_data.get("similarity_score", 0.5)),
                        reason=sim_data.get("reason", ""),
                        confidence=float(sim_data.get("confidence", 0.5))
                    )
                    all_similarities.append(similarity)
                
                logger.info("Similarity processed", 
                           track_a=track_a_id, 
                           track_b=track_b_id,
                           similarities_found=len(ai_response.get("similarities", [])))
                
            except Exception as e:
                logger.error("Similarity processing failed",
                           track_a=track_a_id, 
                           track_b=track_b_id, 
                           error=str(e))
                continue
        
        return all_similarities
    
    async def run_full_pipeline(self, max_files: int = None, 
                              max_similarity_pairs: int = 50) -> Dict[str, Any]:
        """Run the complete labeling pipeline."""
        
        logger.info("Starting full labeling pipeline")
        
        # Scan music library
        audio_files = self.scan_music_library()
        
        if max_files:
            audio_files = audio_files[:max_files]
            logger.info("Limited to max files", max_files=max_files)
        
        # Results storage
        results = {
            "pipeline_info": {
                "music_library_path": str(self.music_library_path),
                "output_path": str(self.output_path),
                "ai_service": self.ai_service,
                "total_files_processed": 0,
                "timestamp": str(asyncio.get_event_loop().time())
            },
            "tracks": {},
            "similarities": []
        }
        
        # Process each track
        logger.info("Processing individual tracks", total_tracks=len(audio_files))
        
        for i, audio_file in enumerate(audio_files):
            track_id = Path(audio_file).stem
            
            try:
                logger.info("Processing track", 
                           track_id=track_id, 
                           progress=f"{i+1}/{len(audio_files)}")
                
                # Process sections
                sections = await self.process_track_sections(audio_file, track_id)
                
                # Process mix points
                mix_points = await self.process_track_mix_points(
                    audio_file, track_id, sections
                )
                
                # Store results
                results["tracks"][track_id] = {
                    "file_path": audio_file,
                    "sections": [
                        {
                            "section_type": s.section_type,
                            "start_time": s.start_time,
                            "end_time": s.end_time,
                            "confidence": s.confidence,
                            "characteristics": s.characteristics,
                            "mix_suitability": s.mix_suitability,
                            "description": s.description
                        } for s in sections
                    ],
                    "mix_points": [
                        {
                            "time": mp.time,
                            "mix_type": mp.mix_type,
                            "suitability": mp.suitability,
                            "energy_level": mp.energy_level,
                            "tempo_stability": mp.tempo_stability,
                            "harmonic_stability": mp.harmonic_stability,
                            "description": mp.description
                        } for mp in mix_points
                    ]
                }
                
                results["pipeline_info"]["total_files_processed"] += 1
                
                # Save intermediate results every 5 tracks
                if (i + 1) % 5 == 0:
                    await self.save_results(results, "intermediate")
                
            except Exception as e:
                logger.error("Track processing failed", 
                           track_id=track_id, 
                           error=str(e))
                continue
        
        # Process similarities
        if len(audio_files) > 1:
            logger.info("Processing track similarities")
            
            # Create pairs for similarity analysis
            import random
            all_pairs = [(audio_files[i], audio_files[j]) 
                        for i in range(len(audio_files)) 
                        for j in range(i+1, len(audio_files))]
            
            # Sample pairs to avoid too many API calls
            if len(all_pairs) > max_similarity_pairs:
                sampled_pairs = random.sample(all_pairs, max_similarity_pairs)
            else:
                sampled_pairs = all_pairs
            
            # Process in smaller batches to avoid overwhelming the API
            batch_size = 5
            for i in range(0, len(sampled_pairs), batch_size):
                batch = sampled_pairs[i:i+batch_size]
                
                logger.info("Processing similarity batch", 
                           batch=f"{i//batch_size + 1}/{(len(sampled_pairs) + batch_size - 1)//batch_size}")
                
                batch_similarities = await self.process_similarity_batch(batch)
                
                # Convert to serializable format
                for sim in batch_similarities:
                    results["similarities"].append({
                        "track_a_id": sim.track_a_id,
                        "track_b_id": sim.track_b_id,
                        "similarity_type": sim.similarity_type,
                        "similarity_score": sim.similarity_score,
                        "reason": sim.reason,
                        "confidence": sim.confidence
                    })
        
        # Save final results
        await self.save_results(results, "final")
        
        logger.info("Pipeline completed", 
                   tracks_processed=results["pipeline_info"]["total_files_processed"],
                   similarities_generated=len(results["similarities"]))
        
        return results
    
    async def save_results(self, results: Dict[str, Any], suffix: str = ""):
        """Save results to JSON files."""
        
        try:
            # Main results file
            main_file = self.output_path / f"ai_labels_{suffix}.json"
            with open(main_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Separate files for different data types
            if results.get("tracks"):
                sections_file = self.output_path / f"track_sections_{suffix}.json"
                sections_data = {
                    track_id: track_data["sections"]
                    for track_id, track_data in results["tracks"].items()
                }
                with open(sections_file, 'w') as f:
                    json.dump(sections_data, f, indent=2)
                
                mix_points_file = self.output_path / f"mix_points_{suffix}.json"
                mix_points_data = {
                    track_id: track_data["mix_points"]
                    for track_id, track_data in results["tracks"].items()
                }
                with open(mix_points_file, 'w') as f:
                    json.dump(mix_points_data, f, indent=2)
            
            if results.get("similarities"):
                similarities_file = self.output_path / f"similarities_{suffix}.json"
                with open(similarities_file, 'w') as f:
                    json.dump(results["similarities"], f, indent=2)
            
            logger.info("Results saved", 
                       main_file=str(main_file),
                       suffix=suffix)
            
        except Exception as e:
            logger.error("Failed to save results", 
                        error=str(e))


async def main():
    """Main entry point for the labeling pipeline."""
    
    parser = argparse.ArgumentParser(
        description="AI-powered music labeling pipeline for DJ mixing"
    )
    
    parser.add_argument(
        "music_library", 
        help="Path to music library directory"
    )
    
    parser.add_argument(
        "output_directory",
        help="Path to output directory for results"
    )
    
    parser.add_argument(
        "--max-files", 
        type=int, 
        default=None,
        help="Maximum number of files to process (for testing)"
    )
    
    parser.add_argument(
        "--max-similarity-pairs",
        type=int,
        default=50,
        help="Maximum number of similarity pairs to analyze"
    )
    
    parser.add_argument(
        "--ai-service",
        choices=["claude", "openai", "mock"],
        default="mock",
        help="AI service to use for analysis"
    )
    
    parser.add_argument(
        "--api-key",
        help="API key for AI service"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    library_path = Path(args.music_library)
    if not library_path.exists():
        logger.error("Music library path does not exist", path=str(library_path))
        sys.exit(1)
    
    # Initialize pipeline
    pipeline = AILabelingPipeline(
        music_library_path=str(library_path),
        output_path=args.output_directory,
        api_key=args.api_key,
        ai_service=args.ai_service
    )
    
    # Run pipeline
    try:
        results = await pipeline.run_full_pipeline(
            max_files=args.max_files,
            max_similarity_pairs=args.max_similarity_pairs
        )
        
        print("\\n=== AI Labeling Pipeline Complete ===")
        print(f"Tracks processed: {results['pipeline_info']['total_files_processed']}")
        print(f"Similarities generated: {len(results['similarities'])}")
        print(f"Results saved to: {args.output_directory}")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error("Pipeline failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
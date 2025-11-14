#!/usr/bin/env python3
"""
Simple Music Labeling Script

This is the ONLY script you need to run!

Usage:
1. Put your OpenAI API key in config.json
2. Put your music files in a folder
3. Run: python run_labeling.py /path/to/your/music/folder

The script will:
- Analyze each song's audio features
- Send structured prompts to GPT-4 for intelligent labeling  
- Generate song sections (verse, chorus, buildup, outro)
- Find optimal DJ mix points
- Save all results as JSON files for training

Output files:
- individual_tracks/ - Analysis for each song
- combined_results.json - All results combined
- training_data.json - Formatted for ML training
"""

import asyncio
import json
import sys
from pathlib import Path
import argparse
from typing import List, Dict, Any
import structlog

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from music_analyzer import BasicMusicAnalyzer, TrackAnalysis
from gpt_labeler import GPTMusicLabeler

# Configure logging
logging = structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class MusicLabelingPipeline:
    """
    Simple pipeline that processes music files and generates AI labels.
    """
    
    def __init__(self, api_key: str, output_dir: str = "output"):
        """
        Initialize the labeling pipeline.
        
        Args:
            api_key: OpenAI API key
            output_dir: Directory to save results
        """
        
        self.analyzer = BasicMusicAnalyzer()
        self.labeler = GPTMusicLabeler(api_key=api_key, model="gpt-4")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "individual_tracks").mkdir(exist_ok=True)
        (self.output_dir / "analysis").mkdir(exist_ok=True)
        
        logger.info("Pipeline initialized", output_dir=str(self.output_dir))
    
    def find_music_files(self, directory: str) -> List[str]:
        """Find all music files in directory."""
        
        music_dir = Path(directory)
        if not music_dir.exists():
            logger.error("Music directory does not exist", directory=directory)
            return []
        
        # Common music file extensions
        extensions = ['.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma']
        
        music_files = []
        for ext in extensions:
            music_files.extend(music_dir.glob(f"**/*{ext}"))
            music_files.extend(music_dir.glob(f"**/*{ext.upper()}"))
        
        music_files = [str(f) for f in music_files]
        
        logger.info("Music files found", 
                   directory=directory, 
                   total_files=len(music_files),
                   extensions=extensions)
        
        return music_files
    
    async def process_single_track(self, audio_file: str) -> Dict[str, Any]:
        """
        Process a single track completely.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Complete labeling results
        """
        
        track_name = Path(audio_file).stem
        logger.info("Processing track", track_name=track_name)
        
        try:
            # Step 1: Analyze audio features
            logger.info("→ Analyzing audio features", track_name=track_name)
            analysis = self.analyzer.analyze_track(audio_file)
            
            # Save analysis
            analysis_file = self.output_dir / "analysis" / f"{track_name}_analysis.json"
            self.analyzer.save_analysis(analysis, str(analysis_file))
            
            # Step 2: Get GPT labels
            logger.info("→ Getting GPT-4 labels", track_name=track_name)
            result = await self.labeler.process_track_complete(analysis)
            
            # Step 3: Save individual result
            result_file = self.output_dir / "individual_tracks" / f"{track_name}_labels.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info("✓ Track completed", 
                       track_name=track_name,
                       sections=len(result['sections']),
                       mix_points=len(result['mix_points']))
            
            return result
            
        except Exception as e:
            logger.error("✗ Track failed", 
                        track_name=track_name, 
                        error=str(e))
            return {
                "track_info": {"title": track_name, "error": str(e)},
                "sections": [],
                "mix_points": []
            }
    
    async def process_all_tracks(self, music_files: List[str]) -> Dict[str, Any]:
        """
        Process all tracks in the list.
        
        Args:
            music_files: List of audio file paths
            
        Returns:
            Combined results for all tracks
        """
        
        logger.info("Starting batch processing", total_tracks=len(music_files))
        
        all_results = {
            "metadata": {
                "total_tracks": len(music_files),
                "processed_tracks": 0,
                "failed_tracks": 0,
                "timestamp": str(asyncio.get_event_loop().time())
            },
            "tracks": {}
        }
        
        # Process each track
        for i, audio_file in enumerate(music_files):
            track_name = Path(audio_file).stem
            
            print(f"\\n[{i+1}/{len(music_files)}] Processing: {track_name}")
            
            result = await self.process_single_track(audio_file)
            
            if result.get("track_info", {}).get("error"):
                all_results["metadata"]["failed_tracks"] += 1
            else:
                all_results["metadata"]["processed_tracks"] += 1
            
            all_results["tracks"][track_name] = result
            
            # Save intermediate results every 3 tracks
            if (i + 1) % 3 == 0:
                await self.save_combined_results(all_results, "intermediate")
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(1)
        
        # Save final results
        await self.save_combined_results(all_results, "final")
        
        return all_results
    
    async def save_combined_results(self, all_results: Dict[str, Any], suffix: str = ""):
        """Save combined results to files."""
        
        try:
            # Main results file
            results_file = self.output_dir / f"combined_results_{suffix}.json"
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            # Training data format
            training_data = self.format_for_training(all_results)
            training_file = self.output_dir / f"training_data_{suffix}.json"
            with open(training_file, 'w') as f:
                json.dump(training_data, f, indent=2)
            
            logger.info("Results saved", 
                       results_file=str(results_file),
                       training_file=str(training_file))
            
        except Exception as e:
            logger.error("Failed to save results", error=str(e))
    
    def format_for_training(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Format results for ML training."""
        
        training_data = {
            "sections": [],
            "mix_points": [],
            "track_metadata": []
        }
        
        for track_name, result in all_results["tracks"].items():
            if result.get("track_info", {}).get("error"):
                continue
            
            track_info = result["track_info"]
            
            # Add sections with track reference
            for section in result["sections"]:
                section_data = section.copy()
                section_data["track_id"] = track_name
                section_data["track_tempo"] = track_info.get("tempo", 0)
                section_data["track_key"] = track_info.get("key", "unknown")
                section_data["track_duration"] = track_info.get("duration", 0)
                training_data["sections"].append(section_data)
            
            # Add mix points with track reference
            for mix_point in result["mix_points"]:
                mix_point_data = mix_point.copy()
                mix_point_data["track_id"] = track_name
                mix_point_data["track_tempo"] = track_info.get("tempo", 0)
                mix_point_data["track_key"] = track_info.get("key", "unknown")
                training_data["mix_points"].append(mix_point_data)
            
            # Add track metadata
            training_data["track_metadata"].append({
                "track_id": track_name,
                "title": track_info.get("title", ""),
                "duration": track_info.get("duration", 0),
                "tempo": track_info.get("tempo", 0),
                "key": track_info.get("key", ""),
                "total_sections": len(result["sections"]),
                "total_mix_points": len(result["mix_points"]),
                "file_path": track_info.get("file_path", "")
            })
        
        return training_data
    
    def print_summary(self, all_results: Dict[str, Any]):
        """Print a summary of results."""
        
        metadata = all_results["metadata"]
        
        print("\\n" + "="*50)
        print("LABELING COMPLETE!")
        print("="*50)
        print(f"Total tracks processed: {metadata['processed_tracks']}")
        print(f"Failed tracks: {metadata['failed_tracks']}")
        
        # Count sections and mix points
        total_sections = 0
        total_mix_points = 0
        
        for result in all_results["tracks"].values():
            total_sections += len(result.get("sections", []))
            total_mix_points += len(result.get("mix_points", []))
        
        print(f"Total sections identified: {total_sections}")
        print(f"Total mix points found: {total_mix_points}")
        print(f"\\nResults saved to: {self.output_dir}")
        print("\\nFiles created:")
        print("- individual_tracks/ - Labels for each song")
        print("- combined_results_final.json - All results")
        print("- training_data_final.json - Formatted for ML training")
        print("="*50)


def load_config() -> Dict[str, str]:
    """Load configuration from config.json."""
    
    config_file = Path(__file__).parent / "config.json"
    
    if not config_file.exists():
        # Create template config
        template_config = {
            "openai_api_key": "YOUR_API_KEY_HERE",
            "model": "gpt-4",
            "notes": "Replace YOUR_API_KEY_HERE with your actual OpenAI API key"
        }
        
        with open(config_file, 'w') as f:
            json.dump(template_config, f, indent=2)
        
        print(f"\\n❌ Config file created at: {config_file}")
        print("Please edit config.json and add your OpenAI API key, then run again.")
        sys.exit(1)
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    if config.get("openai_api_key") == "YOUR_API_KEY_HERE":
        print("\\n❌ Please edit config.json and add your OpenAI API key")
        sys.exit(1)
    
    return config


async def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="AI Music Labeling - Generate DJ mixing labels using GPT-4"
    )
    
    parser.add_argument(
        "music_directory",
        help="Directory containing music files to analyze"
    )
    
    parser.add_argument(
        "--output",
        default="output",
        help="Output directory for results (default: output)"
    )
    
    parser.add_argument(
        "--max-tracks",
        type=int,
        help="Maximum number of tracks to process (for testing)"
    )
    
    args = parser.parse_args()
    
    print("🎵 AI Music Labeling Pipeline")
    print("============================")
    
    # Load config
    try:
        config = load_config()
        api_key = config["openai_api_key"]
        print("✓ Configuration loaded")
    except Exception as e:
        print(f"❌ Config error: {e}")
        sys.exit(1)
    
    # Validate music directory
    if not Path(args.music_directory).exists():
        print(f"❌ Music directory does not exist: {args.music_directory}")
        sys.exit(1)
    
    # Initialize pipeline
    try:
        pipeline = MusicLabelingPipeline(api_key=api_key, output_dir=args.output)
        print("✓ Pipeline initialized")
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        sys.exit(1)
    
    # Find music files
    music_files = pipeline.find_music_files(args.music_directory)
    
    if not music_files:
        print("❌ No music files found!")
        sys.exit(1)
    
    if args.max_tracks:
        music_files = music_files[:args.max_tracks]
        print(f"📝 Limited to {args.max_tracks} tracks")
    
    print(f"🎵 Found {len(music_files)} music files")
    print("\\nStarting analysis...")
    
    # Process all tracks
    try:
        results = await pipeline.process_all_tracks(music_files)
        pipeline.print_summary(results)
        
    except KeyboardInterrupt:
        print("\\n⏹️ Stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error("Pipeline failed", error=str(e))
        print(f"\\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
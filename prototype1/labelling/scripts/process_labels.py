#!/usr/bin/env python3
"""
Label Processing Utilities

After running the labeling pipeline, use these scripts to:
1. Convert labels to different formats
2. Generate training datasets for ML models
3. Analyze labeling quality and statistics
4. Export for use in other DJ software

Usage:
    python process_labels.py training_data_final.json --export-format ml_ready
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

import structlog

logger = structlog.get_logger(__name__)


class LabelProcessor:
    """Process and convert AI-generated labels to different formats."""
    
    def __init__(self, labels_file: str):
        """
        Initialize with labels file.
        
        Args:
            labels_file: Path to training_data.json file
        """
        
        self.labels_file = Path(labels_file)
        
        if not self.labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        
        # Load data
        with open(self.labels_file, 'r') as f:
            self.data = json.load(f)
        
        self.sections = self.data.get("sections", [])
        self.mix_points = self.data.get("mix_points", [])
        self.track_metadata = self.data.get("track_metadata", [])
        
        logger.info("Labels loaded",
                   sections=len(self.sections),
                   mix_points=len(self.mix_points),
                   tracks=len(self.track_metadata))
    
    def generate_ml_training_data(self) -> Dict[str, Any]:
        """Generate data ready for ML model training."""
        
        # 1. Section classification dataset
        section_classification = []
        
        for section in self.sections:
            # Create features for section classification
            features = {
                'track_tempo': section.get('track_tempo', 120),
                'track_duration': section.get('track_duration', 180),
                'section_start_ratio': section.get('start_time', 0) / max(section.get('track_duration', 180), 1),
                'section_duration': section.get('end_time', 0) - section.get('start_time', 0),
                'section_duration_ratio': (section.get('end_time', 0) - section.get('start_time', 0)) / max(section.get('track_duration', 180), 1),
                'confidence': section.get('confidence', 0.5),
                'has_vocals': 1 if section.get('has_vocals', False) else 0,
                'energy_level_low': 1 if section.get('energy_level') == 'low' else 0,
                'energy_level_medium': 1 if section.get('energy_level') == 'medium' else 0,
                'energy_level_high': 1 if section.get('energy_level') == 'high' else 0,
            }
            
            # Target
            target = {
                'section_type': section.get('section_type', 'unknown'),
                'mix_suitability': section.get('mix_suitability', 'fair')
            }
            
            section_classification.append({
                'features': features,
                'target': target,
                'track_id': section.get('track_id', ''),
                'meta': {
                    'start_time': section.get('start_time', 0),
                    'end_time': section.get('end_time', 0),
                    'description': section.get('description', '')
                }
            })
        
        # 2. Mix point detection dataset
        mix_point_detection = []
        
        for point in self.mix_points:
            # Create features for mix point detection
            track_duration = next(
                (t['duration'] for t in self.track_metadata 
                 if t['track_id'] == point.get('track_id')), 
                180
            )
            
            features = {
                'track_tempo': point.get('track_tempo', 120),
                'time_ratio': point.get('time', 0) / max(track_duration, 1),
                'energy_description_length': len(point.get('energy_description', '')),
                'explanation_length': len(point.get('why_good_for_mixing', '')),
            }
            
            # Target
            target = {
                'mix_type': point.get('mix_type', 'unknown'),
                'suitability': point.get('suitability', 'fair')
            }
            
            mix_point_detection.append({
                'features': features,
                'target': target,
                'track_id': point.get('track_id', ''),
                'meta': {
                    'time': point.get('time', 0),
                    'energy_description': point.get('energy_description', ''),
                    'why_good_for_mixing': point.get('why_good_for_mixing', '')
                }
            })
        
        # 3. Track similarity pairs (based on same section types and similar tempos)
        similarity_pairs = []
        
        # Group tracks by section types
        tracks_by_sections = defaultdict(list)
        
        for track in self.track_metadata:
            track_id = track['track_id']
            track_sections = [s for s in self.sections if s.get('track_id') == track_id]
            section_types = sorted(set(s.get('section_type', 'unknown') for s in track_sections))
            
            # Create signature of section types
            signature = ','.join(section_types)
            tracks_by_sections[signature].append(track)
        
        # Create positive similarity pairs
        for signature, tracks in tracks_by_sections.items():
            if len(tracks) > 1:
                for i in range(len(tracks)):
                    for j in range(i + 1, len(tracks)):
                        track_a = tracks[i]
                        track_b = tracks[j]
                        
                        # Check tempo similarity
                        tempo_diff = abs(track_a.get('tempo', 120) - track_b.get('tempo', 120))
                        
                        similarity_score = 1.0 - min(tempo_diff / 40.0, 1.0)  # Similar if within 40 BPM
                        
                        if similarity_score > 0.5:  # Only include reasonably similar tracks
                            similarity_pairs.append({
                                'track_a_id': track_a['track_id'],
                                'track_b_id': track_b['track_id'],
                                'similarity_score': similarity_score,
                                'similarity_type': 'structural_and_rhythmic',
                                'features_a': {
                                    'tempo': track_a.get('tempo', 120),
                                    'duration': track_a.get('duration', 180),
                                    'total_sections': track_a.get('total_sections', 0)
                                },
                                'features_b': {
                                    'tempo': track_b.get('tempo', 120), 
                                    'duration': track_b.get('duration', 180),
                                    'total_sections': track_b.get('total_sections', 0)
                                }
                            })
        
        return {
            'section_classification': section_classification,
            'mix_point_detection': mix_point_detection,
            'similarity_pairs': similarity_pairs,
            'dataset_stats': {
                'total_sections': len(section_classification),
                'total_mix_points': len(mix_point_detection),
                'total_similarity_pairs': len(similarity_pairs),
                'unique_tracks': len(self.track_metadata)
            }
        }
    
    def export_to_csv(self, output_dir: str = "csv_export"):
        """Export labels to CSV files for analysis."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Sections CSV
        sections_df = pd.DataFrame(self.sections)
        sections_df.to_csv(output_path / "sections.csv", index=False)
        
        # Mix points CSV
        mix_points_df = pd.DataFrame(self.mix_points)
        mix_points_df.to_csv(output_path / "mix_points.csv", index=False)
        
        # Track metadata CSV
        tracks_df = pd.DataFrame(self.track_metadata)
        tracks_df.to_csv(output_path / "tracks.csv", index=False)
        
        logger.info("CSV export completed", output_dir=output_dir)
    
    def generate_statistics(self) -> Dict[str, Any]:
        """Generate statistics about the labeled data."""
        
        stats = {
            'overview': {
                'total_tracks': len(self.track_metadata),
                'total_sections': len(self.sections),
                'total_mix_points': len(self.mix_points),
                'avg_sections_per_track': len(self.sections) / max(len(self.track_metadata), 1),
                'avg_mix_points_per_track': len(self.mix_points) / max(len(self.track_metadata), 1)
            }
        }
        
        # Section type distribution
        section_types = [s.get('section_type', 'unknown') for s in self.sections]
        stats['section_types'] = dict(Counter(section_types))
        
        # Energy level distribution
        energy_levels = [s.get('energy_level', 'unknown') for s in self.sections]
        stats['energy_levels'] = dict(Counter(energy_levels))
        
        # Mix suitability distribution
        mix_suitabilities = [s.get('mix_suitability', 'unknown') for s in self.sections]
        stats['mix_suitabilities'] = dict(Counter(mix_suitabilities))
        
        # Mix point type distribution
        mix_types = [mp.get('mix_type', 'unknown') for mp in self.mix_points]
        stats['mix_types'] = dict(Counter(mix_types))
        
        # Mix point suitability distribution
        mix_point_suitabilities = [mp.get('suitability', 'unknown') for mp in self.mix_points]
        stats['mix_point_suitabilities'] = dict(Counter(mix_point_suitabilities))
        
        # Tempo distribution
        tempos = [t.get('tempo', 0) for t in self.track_metadata if t.get('tempo', 0) > 0]
        if tempos:
            stats['tempo_stats'] = {
                'min': min(tempos),
                'max': max(tempos),
                'mean': np.mean(tempos),
                'median': np.median(tempos),
                'std': np.std(tempos)
            }
        
        # Duration distribution
        durations = [t.get('duration', 0) for t in self.track_metadata if t.get('duration', 0) > 0]
        if durations:
            stats['duration_stats'] = {
                'min_seconds': min(durations),
                'max_seconds': max(durations),
                'mean_seconds': np.mean(durations),
                'median_seconds': np.median(durations),
                'mean_minutes': np.mean(durations) / 60
            }
        
        return stats
    
    def export_for_rekordbox(self, output_file: str = "rekordbox_cues.xml"):
        """Export mix points as Rekordbox-compatible cue points (simplified XML)."""
        
        # Group mix points by track
        tracks_mix_points = defaultdict(list)
        
        for mp in self.mix_points:
            track_id = mp.get('track_id', '')
            tracks_mix_points[track_id].append(mp)
        
        # Generate simple XML structure
        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<PLAYLISTS>',
            '  <COLLECTION>'
        ]
        
        for track in self.track_metadata:
            track_id = track['track_id']
            mix_points = tracks_mix_points.get(track_id, [])
            
            if mix_points:
                xml_lines.extend([
                    f'    <TRACK TrackID="{track_id}" Name="{track.get("title", "")}" Artist="{track.get("artist", "")}">',
                ])
                
                for i, mp in enumerate(mix_points):
                    time_ms = int(mp.get('time', 0) * 1000)  # Convert to milliseconds
                    xml_lines.append(
                        f'      <POSITION_MARK Name="Mix_{mp.get("mix_type", "")}" Type="0" Start="{time_ms}" />'
                    )
                
                xml_lines.append('    </TRACK>')
        
        xml_lines.extend([
            '  </COLLECTION>',
            '</PLAYLISTS>'
        ])
        
        with open(output_file, 'w') as f:
            f.write('\\n'.join(xml_lines))
        
        logger.info("Rekordbox XML export completed", output_file=output_file)
    
    def print_statistics_report(self, stats: Dict[str, Any]):
        """Print a formatted statistics report."""
        
        print("\\n" + "="*60)
        print("LABELING STATISTICS REPORT")
        print("="*60)
        
        # Overview
        overview = stats['overview']
        print(f"\\n📊 OVERVIEW:")
        print(f"   Total tracks analyzed: {overview['total_tracks']}")
        print(f"   Total sections found: {overview['total_sections']}")
        print(f"   Total mix points: {overview['total_mix_points']}")
        print(f"   Avg sections per track: {overview['avg_sections_per_track']:.1f}")
        print(f"   Avg mix points per track: {overview['avg_mix_points_per_track']:.1f}")
        
        # Section types
        print(f"\\n🎵 SECTION TYPES:")
        for section_type, count in stats['section_types'].items():
            percentage = (count / overview['total_sections']) * 100
            print(f"   {section_type}: {count} ({percentage:.1f}%)")
        
        # Energy levels
        print(f"\\n⚡ ENERGY LEVELS:")
        for energy, count in stats['energy_levels'].items():
            percentage = (count / overview['total_sections']) * 100
            print(f"   {energy}: {count} ({percentage:.1f}%)")
        
        # Mix types
        print(f"\\n🎧 MIX POINT TYPES:")
        for mix_type, count in stats['mix_types'].items():
            percentage = (count / overview['total_mix_points']) * 100
            print(f"   {mix_type}: {count} ({percentage:.1f}%)")
        
        # Tempo stats
        if 'tempo_stats' in stats:
            tempo = stats['tempo_stats']
            print(f"\\n🥁 TEMPO ANALYSIS:")
            print(f"   Range: {tempo['min']:.0f} - {tempo['max']:.0f} BPM")
            print(f"   Average: {tempo['mean']:.1f} BPM")
            print(f"   Median: {tempo['median']:.1f} BPM")
        
        print("\\n" + "="*60)


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Process and convert AI-generated music labels"
    )
    
    parser.add_argument(
        "labels_file",
        help="Path to training_data.json file from labeling pipeline"
    )
    
    parser.add_argument(
        "--export-format",
        choices=["ml_ready", "csv", "rekordbox", "statistics", "all"],
        default="statistics",
        help="Export format (default: statistics)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="processed_labels",
        help="Output directory for processed files"
    )
    
    args = parser.parse_args()
    
    print("🏷️  Label Processing Utilities")
    print("=============================")
    
    # Initialize processor
    try:
        processor = LabelProcessor(args.labels_file)
        print(f"✓ Labels loaded from: {args.labels_file}")
    except Exception as e:
        print(f"❌ Failed to load labels: {e}")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Process based on format
    if args.export_format == "statistics" or args.export_format == "all":
        print("\\n📊 Generating statistics...")
        stats = processor.generate_statistics()
        
        # Save stats
        with open(output_path / "statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        processor.print_statistics_report(stats)
    
    if args.export_format == "ml_ready" or args.export_format == "all":
        print("\\n🤖 Generating ML-ready dataset...")
        ml_data = processor.generate_ml_training_data()
        
        with open(output_path / "ml_training_data.json", 'w') as f:
            json.dump(ml_data, f, indent=2)
        
        print(f"✓ ML dataset saved with {ml_data['dataset_stats']['total_sections']} sections")
        print(f"   and {ml_data['dataset_stats']['total_similarity_pairs']} similarity pairs")
    
    if args.export_format == "csv" or args.export_format == "all":
        print("\\n📄 Exporting to CSV...")
        csv_dir = output_path / "csv_export"
        processor.export_to_csv(str(csv_dir))
        print(f"✓ CSV files saved to: {csv_dir}")
    
    if args.export_format == "rekordbox" or args.export_format == "all":
        print("\\n💿 Generating Rekordbox XML...")
        xml_file = output_path / "mix_points.xml"
        processor.export_for_rekordbox(str(xml_file))
        print(f"✓ Rekordbox XML saved to: {xml_file}")
    
    print(f"\\n✅ Processing complete! Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
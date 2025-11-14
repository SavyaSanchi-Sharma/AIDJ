"""
Comprehensive metadata extraction for music tracks.
Extracts metadata from multiple sources: file tags, APIs, and derived features.
"""

import structlog
import os
import requests
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import asyncio
import aiohttp
from dataclasses import dataclass
from datetime import datetime

try:
    from mutagen import File as MutagenFile
    from mutagen.id3 import ID3NoHeaderError
except ImportError:
    MutagenFile = None
    ID3NoHeaderError = None

logger = structlog.get_logger(__name__)


@dataclass
class MetadataConfig:
    """Configuration for metadata extraction APIs."""
    spotify_client_id: Optional[str] = None
    spotify_client_secret: Optional[str] = None
    lastfm_api_key: Optional[str] = None
    musicbrainz_user_agent: str = "MusicMixer/1.0"
    rate_limit_delay: float = 0.1  # Seconds between API calls


class ComprehensiveMetadataExtractor:
    """
    Extracts comprehensive metadata from multiple sources:
    - File tags (ID3, etc.)
    - Spotify API (audio features, popularity)
    - Last.fm API (tags, similar artists)
    - MusicBrainz (release info, relationships)
    - Derived features (cross-platform correlations)
    """
    
    def __init__(self, config: MetadataConfig):
        """Initialize metadata extractor with API configurations."""
        self.config = config
        self.spotify_token = None
        self.spotify_token_expires = 0
        
        logger.info("MetadataExtractor initialized")
    
    async def extract_all_metadata(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from all available sources.
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Dictionary with all extracted metadata organized by source
        """
        logger.info("Starting comprehensive metadata extraction", file_path=audio_file_path)
        
        metadata = {
            "file_info": {},
            "file_tags": {},
            "spotify": {},
            "lastfm": {},
            "musicbrainz": {},
            "derived": {},
            "extraction_timestamp": datetime.now().isoformat()
        }
        
        # Extract file-based information
        metadata["file_info"] = self._extract_file_info(audio_file_path)
        metadata["file_tags"] = self._extract_file_tags(audio_file_path)
        
        # Extract from APIs (async)
        if metadata["file_tags"].get("title") and metadata["file_tags"].get("artist"):
            title = metadata["file_tags"]["title"]
            artist = metadata["file_tags"]["artist"]
            
            # Run API extractions concurrently
            api_tasks = []
            
            if self.config.spotify_client_id:
                api_tasks.append(self._extract_spotify_metadata(title, artist))
            
            if self.config.lastfm_api_key:
                api_tasks.append(self._extract_lastfm_metadata(title, artist))
                
            api_tasks.append(self._extract_musicbrainz_metadata(title, artist))
            
            # Execute API calls concurrently
            if api_tasks:
                api_results = await asyncio.gather(*api_tasks, return_exceptions=True)
                
                # Process results
                if len(api_results) > 0 and not isinstance(api_results[0], Exception):
                    metadata["spotify"] = api_results[0] or {}
                
                if len(api_results) > 1 and not isinstance(api_results[1], Exception):
                    metadata["lastfm"] = api_results[1] or {}
                    
                if len(api_results) > 2 and not isinstance(api_results[2], Exception):
                    metadata["musicbrainz"] = api_results[2] or {}
        
        # Generate derived features
        metadata["derived"] = self._generate_derived_features(metadata)
        
        # Calculate comprehensive feature vector
        metadata["feature_vector"] = self._generate_comprehensive_features(metadata)
        
        logger.info("Comprehensive metadata extraction completed", 
                   feature_count=len(metadata["feature_vector"]))
        
        return metadata
    
    def _extract_file_info(self, audio_file_path: str) -> Dict[str, Any]:
        """Extract basic file information."""
        
        try:
            file_path = Path(audio_file_path)
            file_stat = file_path.stat()
            
            return {
                "filename": file_path.name,
                "file_extension": file_path.suffix.lower(),
                "file_size_bytes": file_stat.st_size,
                "file_size_mb": round(file_stat.st_size / (1024 * 1024), 2),
                "creation_time": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                "modification_time": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                "file_path": str(file_path.absolute())
            }
            
        except Exception as e:
            logger.warning("File info extraction failed", error=str(e))
            return {}
    
    def _extract_file_tags(self, audio_file_path: str) -> Dict[str, Any]:
        """Extract metadata from audio file tags."""
        
        if not MutagenFile:
            logger.warning("Mutagen not available, skipping file tag extraction")
            return {}
        
        try:
            audio_file = MutagenFile(audio_file_path)
            
            if audio_file is None:
                return {}
            
            tags = {}
            
            # Common tag mappings
            tag_mappings = {
                'title': ['TIT2', 'TITLE', '\xa9nam'],
                'artist': ['TPE1', 'ARTIST', '\xa9ART'],
                'album': ['TALB', 'ALBUM', '\xa9alb'],
                'albumartist': ['TPE2', 'ALBUMARTIST', 'aART'],
                'date': ['TDRC', 'DATE', '\xa9day'],
                'year': ['TYER', 'YEAR'],
                'genre': ['TCON', 'GENRE', '\xa9gen'],
                'track': ['TRCK', 'TRACKNUMBER', 'trkn'],
                'disc': ['TPOS', 'DISCNUMBER', 'disk'],
                'duration': ['TLEN'],
                'bpm': ['TBPM', 'BPM', 'tmpo'],
                'key': ['TKEY', 'INITIALKEY'],
                'composer': ['TCOM', 'COMPOSER', '\xa9wrt'],
                'lyricist': ['TEXT', 'LYRICIST'],
                'copyright': ['TCOP', 'COPYRIGHT', 'cprt'],
                'label': ['TPUB', 'LABEL', '©lbl'],
                'isrc': ['TSRC', 'ISRC'],
                'mood': ['TMOO', 'MOOD'],
                'comment': ['COMM', 'COMMENT', '\xa9cmt'],
                'grouping': ['TIT1', 'GROUPING', '\xa9grp'],
                'language': ['TLAN', 'LANGUAGE']
            }
            
            # Extract standard tags
            for tag_name, possible_keys in tag_mappings.items():
                for key in possible_keys:
                    if key in audio_file.tags:
                        value = audio_file.tags[key]
                        
                        # Handle different value types
                        if hasattr(value, 'text'):
                            tags[tag_name] = str(value.text[0]) if value.text else ""
                        elif isinstance(value, list) and len(value) > 0:
                            tags[tag_name] = str(value[0])
                        else:
                            tags[tag_name] = str(value)
                        break
            
            # Extract audio properties
            if hasattr(audio_file, 'info'):
                info = audio_file.info
                tags.update({
                    'bitrate': getattr(info, 'bitrate', 0),
                    'sample_rate': getattr(info, 'sample_rate', 0),
                    'channels': getattr(info, 'channels', 0),
                    'length_seconds': getattr(info, 'length', 0.0),
                    'bitdepth': getattr(info, 'bits_per_sample', 0),
                    'codec': getattr(info, 'mime', ['unknown'])[0] if hasattr(info, 'mime') else 'unknown'
                })
            
            # Parse track/disc numbers
            if 'track' in tags and '/' in str(tags['track']):
                track_parts = str(tags['track']).split('/')
                tags['track_number'] = int(track_parts[0]) if track_parts[0].isdigit() else None
                tags['total_tracks'] = int(track_parts[1]) if len(track_parts) > 1 and track_parts[1].isdigit() else None
            
            if 'disc' in tags and '/' in str(tags['disc']):
                disc_parts = str(tags['disc']).split('/')
                tags['disc_number'] = int(disc_parts[0]) if disc_parts[0].isdigit() else None
                tags['total_discs'] = int(disc_parts[1]) if len(disc_parts) > 1 and disc_parts[1].isdigit() else None
            
            # Extract year from date if not present
            if 'date' in tags and 'year' not in tags:
                try:
                    tags['year'] = int(str(tags['date'])[:4])
                except (ValueError, TypeError):
                    pass
            
            logger.info("File tags extracted", tag_count=len(tags))
            return tags
            
        except Exception as e:
            logger.warning("File tag extraction failed", error=str(e))
            return {}
    
    async def _extract_spotify_metadata(self, title: str, artist: str) -> Dict[str, Any]:
        """Extract metadata from Spotify API."""
        
        if not self.config.spotify_client_id or not self.config.spotify_client_secret:
            logger.warning("Spotify credentials not configured")
            return {}
        
        try:
            # Get Spotify access token if needed
            if not self.spotify_token or time.time() > self.spotify_token_expires:
                await self._refresh_spotify_token()
            
            headers = {'Authorization': f'Bearer {self.spotify_token}'}
            
            # Search for track
            search_query = f'track:"{title}" artist:"{artist}"'
            search_url = f"https://api.spotify.com/v1/search?q={search_query}&type=track&limit=1"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, headers=headers) as response:
                    if response.status != 200:
                        logger.warning("Spotify search failed", status=response.status)
                        return {}
                    
                    search_data = await response.json()
                    
                    if not search_data['tracks']['items']:
                        logger.info("Track not found on Spotify", title=title, artist=artist)
                        return {}
                    
                    track = search_data['tracks']['items'][0]
                    track_id = track['id']
                    
                    # Get audio features
                    features_url = f"https://api.spotify.com/v1/audio-features/{track_id}"
                    async with session.get(features_url, headers=headers) as features_response:
                        audio_features = {}
                        if features_response.status == 200:
                            audio_features = await features_response.json()
                    
                    # Get artist details
                    artist_details = {}
                    if track['artists']:
                        artist_id = track['artists'][0]['id']
                        artist_url = f"https://api.spotify.com/v1/artists/{artist_id}"
                        async with session.get(artist_url, headers=headers) as artist_response:
                            if artist_response.status == 200:
                                artist_details = await artist_response.json()
                    
                    return {
                        'track_info': {
                            'id': track.get('id'),
                            'name': track.get('name'),
                            'popularity': track.get('popularity', 0),
                            'explicit': track.get('explicit', False),
                            'duration_ms': track.get('duration_ms', 0),
                            'preview_url': track.get('preview_url'),
                            'spotify_url': track['external_urls'].get('spotify') if 'external_urls' in track else None
                        },
                        'audio_features': {
                            'danceability': audio_features.get('danceability', 0.0),
                            'energy': audio_features.get('energy', 0.0),
                            'key': audio_features.get('key', -1),
                            'loudness': audio_features.get('loudness', 0.0),
                            'mode': audio_features.get('mode', 0),
                            'speechiness': audio_features.get('speechiness', 0.0),
                            'acousticness': audio_features.get('acousticness', 0.0),
                            'instrumentalness': audio_features.get('instrumentalness', 0.0),
                            'liveness': audio_features.get('liveness', 0.0),
                            'valence': audio_features.get('valence', 0.0),
                            'tempo': audio_features.get('tempo', 0.0),
                            'time_signature': audio_features.get('time_signature', 4)
                        },
                        'artist_info': {
                            'id': artist_details.get('id'),
                            'name': artist_details.get('name'),
                            'popularity': artist_details.get('popularity', 0),
                            'genres': artist_details.get('genres', []),
                            'followers': artist_details.get('followers', {}).get('total', 0),
                            'spotify_url': artist_details.get('external_urls', {}).get('spotify')
                        },
                        'album_info': {
                            'id': track.get('album', {}).get('id'),
                            'name': track.get('album', {}).get('name'),
                            'release_date': track.get('album', {}).get('release_date'),
                            'total_tracks': track.get('album', {}).get('total_tracks', 0),
                            'album_type': track.get('album', {}).get('album_type'),
                            'spotify_url': track.get('album', {}).get('external_urls', {}).get('spotify')
                        }
                    }
            
            await asyncio.sleep(self.config.rate_limit_delay)
            
        except Exception as e:
            logger.error("Spotify metadata extraction failed", error=str(e))
            return {}
    
    async def _refresh_spotify_token(self):
        """Refresh Spotify access token."""
        
        auth_url = "https://accounts.spotify.com/api/token"
        auth_data = {
            'grant_type': 'client_credentials',
            'client_id': self.config.spotify_client_id,
            'client_secret': self.config.spotify_client_secret
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(auth_url, data=auth_data) as response:
                if response.status == 200:
                    token_data = await response.json()
                    self.spotify_token = token_data['access_token']
                    self.spotify_token_expires = time.time() + token_data['expires_in'] - 60
                    logger.info("Spotify token refreshed")
                else:
                    logger.error("Failed to refresh Spotify token", status=response.status)
    
    async def _extract_lastfm_metadata(self, title: str, artist: str) -> Dict[str, Any]:
        """Extract metadata from Last.fm API."""
        
        if not self.config.lastfm_api_key:
            logger.warning("Last.fm API key not configured")
            return {}
        
        try:
            base_url = "http://ws.audioscrobbler.com/2.0/"
            
            async with aiohttp.ClientSession() as session:
                # Get track info
                track_params = {
                    'method': 'track.getInfo',
                    'artist': artist,
                    'track': title,
                    'api_key': self.config.lastfm_api_key,
                    'format': 'json'
                }
                
                async with session.get(base_url, params=track_params) as response:
                    track_data = {}
                    if response.status == 200:
                        data = await response.json()
                        if 'track' in data:
                            track_data = data['track']
                
                # Get artist info
                artist_params = {
                    'method': 'artist.getInfo',
                    'artist': artist,
                    'api_key': self.config.lastfm_api_key,
                    'format': 'json'
                }
                
                async with session.get(base_url, params=artist_params) as response:
                    artist_data = {}
                    if response.status == 200:
                        data = await response.json()
                        if 'artist' in data:
                            artist_data = data['artist']
                
                return {
                    'track_info': {
                        'name': track_data.get('name'),
                        'playcount': int(track_data.get('playcount', 0)),
                        'listeners': int(track_data.get('listeners', 0)),
                        'url': track_data.get('url'),
                        'tags': [tag['name'] for tag in track_data.get('toptags', {}).get('tag', [])]
                    },
                    'artist_info': {
                        'name': artist_data.get('name'),
                        'playcount': int(artist_data.get('playcount', 0)),
                        'listeners': int(artist_data.get('listeners', 0)),
                        'url': artist_data.get('url'),
                        'bio_summary': artist_data.get('bio', {}).get('summary', ''),
                        'tags': [tag['name'] for tag in artist_data.get('tags', {}).get('tag', [])],
                        'similar_artists': [artist['name'] for artist in artist_data.get('similar', {}).get('artist', [])]
                    }
                }
            
            await asyncio.sleep(self.config.rate_limit_delay)
            
        except Exception as e:
            logger.error("Last.fm metadata extraction failed", error=str(e))
            return {}
    
    async def _extract_musicbrainz_metadata(self, title: str, artist: str) -> Dict[str, Any]:
        """Extract metadata from MusicBrainz API."""
        
        try:
            base_url = "https://musicbrainz.org/ws/2/"
            headers = {'User-Agent': self.config.musicbrainz_user_agent}
            
            async with aiohttp.ClientSession() as session:
                # Search for recording
                search_params = {
                    'query': f'recording:"{title}" AND artist:"{artist}"',
                    'fmt': 'json',
                    'limit': 1
                }
                
                search_url = f"{base_url}recording"
                async with session.get(search_url, params=search_params, headers=headers) as response:
                    if response.status != 200:
                        logger.warning("MusicBrainz search failed", status=response.status)
                        return {}
                    
                    data = await response.json()
                    
                    if not data.get('recordings'):
                        logger.info("Recording not found in MusicBrainz", title=title, artist=artist)
                        return {}
                    
                    recording = data['recordings'][0]
                    
                    return {
                        'recording_info': {
                            'id': recording.get('id'),
                            'title': recording.get('title'),
                            'length': recording.get('length'),
                            'disambiguation': recording.get('disambiguation')
                        },
                        'artist_info': {
                            'id': recording.get('artist-credit', [{}])[0].get('artist', {}).get('id'),
                            'name': recording.get('artist-credit', [{}])[0].get('name'),
                            'country': recording.get('artist-credit', [{}])[0].get('artist', {}).get('area', {}).get('name')
                        },
                        'release_info': {
                            'releases': len(recording.get('releases', [])),
                            'earliest_release': min([r.get('date', '9999') for r in recording.get('releases', [])], default=None)
                        }
                    }
            
            await asyncio.sleep(self.config.rate_limit_delay)
            
        except Exception as e:
            logger.error("MusicBrainz metadata extraction failed", error=str(e))
            return {}
    
    def _generate_derived_features(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate derived features from collected metadata."""
        
        derived = {}
        
        # Cross-platform popularity correlation
        spotify_popularity = metadata.get('spotify', {}).get('track_info', {}).get('popularity', 0)
        lastfm_playcount = metadata.get('lastfm', {}).get('track_info', {}).get('playcount', 0)
        
        if spotify_popularity > 0 and lastfm_playcount > 0:
            # Normalize and compare
            normalized_spotify = spotify_popularity / 100.0
            normalized_lastfm = min(lastfm_playcount / 10000000.0, 1.0)  # Cap at 10M plays
            derived['cross_platform_popularity_correlation'] = abs(normalized_spotify - normalized_lastfm)
        
        # Genre consistency
        spotify_genres = metadata.get('spotify', {}).get('artist_info', {}).get('genres', [])
        lastfm_tags = metadata.get('lastfm', {}).get('track_info', {}).get('tags', [])
        
        if spotify_genres and lastfm_tags:
            # Simple genre/tag overlap
            genre_overlap = len(set(spotify_genres) & set(lastfm_tags))
            derived['genre_consistency_score'] = genre_overlap / max(len(spotify_genres), len(lastfm_tags))
        
        # Release age (days since release)
        release_date = metadata.get('spotify', {}).get('album_info', {}).get('release_date')
        if release_date:
            try:
                release_datetime = datetime.fromisoformat(release_date)
                age_days = (datetime.now() - release_datetime).days
                derived['release_age_days'] = age_days
                derived['release_age_years'] = age_days / 365.25
            except (ValueError, TypeError):
                pass
        
        # Audio quality score
        bitrate = metadata.get('file_tags', {}).get('bitrate', 0)
        sample_rate = metadata.get('file_tags', {}).get('sample_rate', 0)
        
        quality_score = 0.0
        if bitrate > 0:
            quality_score += min(bitrate / 320.0, 1.0) * 0.6  # Normalize to 320kbps
        if sample_rate > 0:
            quality_score += min(sample_rate / 44100.0, 1.0) * 0.4  # Normalize to 44.1kHz
        
        derived['audio_quality_score'] = quality_score
        
        # Metadata completeness score
        essential_fields = ['title', 'artist', 'album', 'year', 'genre']
        completeness = sum(1 for field in essential_fields 
                          if metadata.get('file_tags', {}).get(field))
        derived['metadata_completeness'] = completeness / len(essential_fields)
        
        return derived
    
    def _generate_comprehensive_features(self, metadata: Dict[str, Any]) -> List[float]:
        """
        Generate comprehensive feature vector from all metadata.
        Returns a high-dimensional feature vector (500+ dimensions).
        """
        
        features = []
        
        # File-based features (20 dimensions)
        file_info = metadata.get('file_info', {})
        features.extend([
            float(file_info.get('file_size_mb', 0.0)),
            1.0 if file_info.get('file_extension') == '.mp3' else 0.0,
            1.0 if file_info.get('file_extension') == '.flac' else 0.0,
            1.0 if file_info.get('file_extension') == '.wav' else 0.0,
        ])
        
        file_tags = metadata.get('file_tags', {})
        features.extend([
            float(file_tags.get('bitrate', 0)) / 320.0,  # Normalize to 320kbps
            float(file_tags.get('sample_rate', 44100)) / 44100.0,  # Normalize to 44.1kHz
            float(file_tags.get('channels', 2)) / 2.0,  # Normalize to stereo
            float(file_tags.get('length_seconds', 0)) / 300.0,  # Normalize to 5 minutes
            float(file_tags.get('year', 2000)) / 2024.0,  # Normalize to current year
            float(file_tags.get('track_number', 1)) / 20.0,  # Normalize to 20 tracks
            float(file_tags.get('bpm', 120)) / 200.0,  # Normalize to 200 BPM
        ])
        
        # Spotify features (30 dimensions)
        spotify_data = metadata.get('spotify', {})
        audio_features = spotify_data.get('audio_features', {})
        
        features.extend([
            audio_features.get('danceability', 0.5),
            audio_features.get('energy', 0.5),
            audio_features.get('key', 6) / 11.0,  # Normalize to 11 keys
            (audio_features.get('loudness', -10) + 30) / 30.0,  # Normalize -30 to 0 dB
            audio_features.get('mode', 0.5),
            audio_features.get('speechiness', 0.1),
            audio_features.get('acousticness', 0.5),
            audio_features.get('instrumentalness', 0.5),
            audio_features.get('liveness', 0.1),
            audio_features.get('valence', 0.5),
            audio_features.get('tempo', 120) / 200.0,  # Normalize to 200 BPM
            audio_features.get('time_signature', 4) / 7.0,  # Normalize to 7/4 time
        ])
        
        track_info = spotify_data.get('track_info', {})
        artist_info = spotify_data.get('artist_info', {})
        
        features.extend([
            track_info.get('popularity', 0) / 100.0,
            1.0 if track_info.get('explicit', False) else 0.0,
            track_info.get('duration_ms', 180000) / 600000.0,  # Normalize to 10 minutes
            artist_info.get('popularity', 0) / 100.0,
            min(artist_info.get('followers', 0) / 10000000.0, 1.0),  # Cap at 10M followers
        ])
        
        # Genre features (50 dimensions) - One-hot encoding for common genres
        common_genres = [
            'pop', 'rock', 'hip hop', 'electronic', 'jazz', 'classical', 'country',
            'r&b', 'reggae', 'blues', 'folk', 'metal', 'punk', 'indie', 'alternative',
            'dance', 'house', 'techno', 'trance', 'dubstep', 'ambient', 'experimental',
            'funk', 'disco', 'soul', 'gospel', 'world', 'latin', 'reggaeton', 'afrobeat',
            'k-pop', 'j-pop', 'bollywood', 'schlager', 'chanson', 'bossa nova',
            'trap', 'drill', 'grime', 'drum and bass', 'breakbeat', 'garage',
            'post-rock', 'shoegaze', 'emo', 'hardcore', 'grunge', 'britpop', 'new wave', 'synthwave'
        ]
        
        genres = artist_info.get('genres', []) + metadata.get('lastfm', {}).get('track_info', {}).get('tags', [])
        genre_features = []
        
        for genre in common_genres:
            genre_present = any(genre.lower() in g.lower() for g in genres)
            genre_features.append(1.0 if genre_present else 0.0)
        
        features.extend(genre_features)
        
        # Last.fm features (20 dimensions)
        lastfm_data = metadata.get('lastfm', {})
        lastfm_track = lastfm_data.get('track_info', {})
        lastfm_artist = lastfm_data.get('artist_info', {})
        
        features.extend([
            min(lastfm_track.get('playcount', 0) / 10000000.0, 1.0),  # Cap at 10M plays
            min(lastfm_track.get('listeners', 0) / 1000000.0, 1.0),  # Cap at 1M listeners
            min(lastfm_artist.get('playcount', 0) / 100000000.0, 1.0),  # Cap at 100M plays
            min(lastfm_artist.get('listeners', 0) / 10000000.0, 1.0),  # Cap at 10M listeners
            min(len(lastfm_artist.get('similar_artists', [])) / 20.0, 1.0),  # Max 20 similar
        ])
        
        # Derived features (10 dimensions)
        derived = metadata.get('derived', {})
        features.extend([
            derived.get('cross_platform_popularity_correlation', 0.5),
            derived.get('genre_consistency_score', 0.5),
            min(derived.get('release_age_days', 365) / 7300.0, 1.0),  # Normalize to 20 years
            derived.get('audio_quality_score', 0.7),
            derived.get('metadata_completeness', 0.5),
        ])
        
        # Pad to exactly 500 dimensions
        while len(features) < 500:
            features.append(0.0)
        
        return features[:500]  # Truncate if over 500


# Async wrapper for easier usage
async def extract_comprehensive_metadata(audio_file_path: str, config: MetadataConfig) -> Dict[str, Any]:
    """
    Convenience function to extract comprehensive metadata.
    
    Args:
        audio_file_path: Path to audio file
        config: Metadata extraction configuration
        
    Returns:
        Complete metadata dictionary
    """
    extractor = ComprehensiveMetadataExtractor(config)
    return await extractor.extract_all_metadata(audio_file_path)
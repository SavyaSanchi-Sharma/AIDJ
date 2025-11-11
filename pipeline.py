import os
import json
import re
import sqlite3
import time
import google.generativeai as genai
from spotdl import Spotdl

class MusicPipelineStage1:
    """
    First stage of the music generation pipeline:
    - Takes user prompt
    - Uses LLM (Gemini) to identify suitable tracks
    - Checks database for existing tracks
    - Downloads missing tracks using spotDL
    - Stores track info in database
    """
    
    def __init__(self, gemini_api_key, spotify_client_id, spotify_client_secret,
                 db_path="music_tracks.db", download_dir="downloaded_music",
                 model_name="gemini-2.5-flash"):
        """
        Initialize the pipeline
        
        Args:
            gemini_api_key: Your Gemini API key
            spotify_client_id: Spotify API client ID
            spotify_client_secret: Spotify API client secret
            db_path: Path to SQLite database (stores metadata)
            download_dir: Directory where raw audio files will be stored
            model_name: Gemini model to use
        """
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_name)
        self.db_path = db_path
        self.download_dir = download_dir

        # Create download directory if it doesn't exist
        os.makedirs(download_dir, exist_ok=True)

        # Initialize spotDL with Spotify credentials
        self.spotdl = Spotdl(
            client_id=spotify_client_id,
            client_secret=spotify_client_secret
        )

        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for tracks"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_name TEXT NOT NULL,
                artist TEXT NOT NULL,
                spotify_url TEXT UNIQUE,
                local_path TEXT,
                energy_level REAL,
                genre TEXT,
                added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        print(f"✓ Database initialized: {self.db_path}")
    
    def parse_llm_json_response(self, response_text):
        """
        Robustly parse JSON from LLM responses that may include:
        - Markdown code fences (```json ... ```)
        - Extra text before/after JSON
        - Comments or trailing commas
        - Escaped characters
        """
        
        # Step 1: Strip whitespace
        text = response_text.strip()
        
        # Step 2: Remove markdown code fences
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*```$', '', text)
        
        # Step 3: Try to extract JSON from text if it has extra content
        json_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        
        # Step 4: Remove trailing commas (common LLM mistake)
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        
        # Step 5: Try parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}")
            print(f"Attempted to parse: {text[:500]}...")
            
            # Step 6: Try fixing single quotes to double quotes
            try:
                fixed_text = text.replace("'", '"')
                return json.loads(fixed_text)
            except:
                pass
            
            # Step 7: Last resort - try to find and parse just the JSON array/object
            try:
                start = min([i for i in [text.find('['), text.find('{')] if i != -1] or [0])
                end = max([i for i in [text.rfind(']'), text.rfind('}')] if i != -1] or [len(text)])
                if end != -1:
                    end += 1
                
                cleaned = text[start:end]
                return json.loads(cleaned)
            except Exception as final_error:
                print(f"\n{'='*60}")
                print("FAILED TO PARSE LLM RESPONSE")
                print(f"{'='*60}")
                print(f"Raw response:\n{response_text}\n")
                print(f"{'='*60}")
                raise ValueError(f"Unable to parse JSON from LLM response. Error: {e}")
    
    def get_track_suggestions_from_llm(self, user_prompt):
        """
        Use Gemini LLM to analyze user prompt and suggest specific tracks
        Returns: List of track suggestions with metadata
        """
        prompt = f"""You are a music expert assistant. When given a user's description 
of the type of music they want, suggest 5-10 specific songs that match their criteria.

IMPORTANT: Respond with ONLY a valid JSON array. No explanations, no markdown formatting, 
no text before or after. Just the JSON array.

Format your response exactly like this:
[
    {{
        "track_name": "Song Title",
        "artist": "Artist Name",
        "spotify_url": "",
        "energy_level": 0.85,
        "genre": "pop"
    }}
]

User request: {user_prompt}

Remember: ONLY return the JSON array, nothing else."""
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            print("\n--- LLM Raw Response ---")
            print(response_text[:300] + "..." if len(response_text) > 300 else response_text)
            print("--- End Raw Response ---\n")
            
            # Use robust parser
            tracks = self.parse_llm_json_response(response_text)
            
            # Validate the response structure
            if not isinstance(tracks, list):
                raise ValueError("LLM response is not a list/array")
            
            if len(tracks) == 0:
                raise ValueError("LLM returned empty track list")
            
            # Validate each track has required fields
            for i, track in enumerate(tracks):
                if not isinstance(track, dict):
                    raise ValueError(f"Track {i} is not a dictionary")
                if 'track_name' not in track or 'artist' not in track:
                    raise ValueError(f"Track {i} missing required fields (track_name, artist)")
            
            return tracks
            
        except Exception as e:
            print(f"\n{'!'*60}")
            print(f"ERROR in get_track_suggestions_from_llm: {e}")
            print(f"{'!'*60}\n")
            raise
    
    def check_track_in_db(self, track_name, artist):
        """Check if track already exists in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM tracks WHERE track_name = ? AND artist = ?",
            (track_name, artist)
        )
        result = cursor.fetchone()
        conn.close()
        return result is not None, result
    
    def download_track(self, track_name, artist, spotify_url=None):
        """
        Download track using spotDL
        Returns: local file path
        """
        try:
            print(f"  Searching for: {track_name} by {artist}")
            
            # Search for the track
            if spotify_url and spotify_url.strip():
                search_results = self.spotdl.search([spotify_url])
            else:
                query = f"{track_name} {artist}"
                search_results = self.spotdl.search([query])
            
            if not search_results or len(search_results) == 0:
                print(f"  No results found")
                return None
            
            song = search_results[0]
            print(f"  Found: {song.name} by {song.artist}")
            
            # Check if already downloaded
            for filename in os.listdir(self.download_dir):
                if filename.endswith(('.mp3', '.m4a', '.opus', '.ogg')):
                    # Loose matching - check if track name OR artist is in filename
                    if (track_name.lower()[:10] in filename.lower() or 
                        artist.lower()[:10] in filename.lower()):
                        file_path = os.path.join(self.download_dir, filename)
                        print(f"  File already exists: {filename}")
                        return file_path
            
            # Download the song
            print(f"  Downloading...")
            download_results, errors = self.spotdl.downloader.download_multiple_songs(search_results)
            
            # Wait a moment for file to be written
            time.sleep(1)
            
            # Find the newly downloaded file
            for filename in os.listdir(self.download_dir):
                if filename.endswith(('.mp3', '.m4a', '.opus', '.ogg')):
                    file_path = os.path.join(self.download_dir, filename)
                    file_time = os.path.getmtime(file_path)
                    
                    # If file was modified in last 10 seconds, it's probably our download
                    if time.time() - file_time < 10:
                        print(f"  Successfully downloaded: {filename}")
                        return file_path
            
            # Fallback: return any matching file
            for filename in os.listdir(self.download_dir):
                if filename.endswith(('.mp3', '.m4a', '.opus', '.ogg')):
                    if (track_name.lower()[:5] in filename.lower() or 
                        artist.lower()[:5] in filename.lower()):
                        file_path = os.path.join(self.download_dir, filename)
                        print(f"  Found matching file: {filename}")
                        return file_path
            
            print(f"  Download completed but file not found")
            return None
            
        except Exception as e:
            print(f"  Error downloading {track_name} by {artist}: {e}")
            return None
    
    def add_track_to_db(self, track_name, artist, local_path, spotify_url=None, 
                        energy_level=None, genre=None):
        """Add downloaded track to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO tracks (track_name, artist, spotify_url, local_path, 
                                  energy_level, genre)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (track_name, artist, spotify_url, local_path, energy_level, genre))
            conn.commit()
            track_id = cursor.lastrowid
            print(f"  Added to database (ID: {track_id})")
            return track_id
        except sqlite3.IntegrityError as e:
            print(f"  Track already in database: {e}")
            return None
        finally:
            conn.close()
    
    def populate_db_from_folder(self):
        """Scan download folder and add all existing tracks to database"""
        
        print(f"\n{'='*60}")
        print("POPULATING DATABASE FROM EXISTING FILES")
        print(f"{'='*60}\n")
        
        if not os.path.exists(self.download_dir):
            print(f"Error: {self.download_dir} folder not found")
            return
        
        mp3_files = [f for f in os.listdir(self.download_dir) 
                     if f.endswith(('.mp3', '.m4a', '.opus', '.ogg'))]
        
        print(f"Found {len(mp3_files)} audio files in {self.download_dir}/")
        print("-" * 60)
        
        added = 0
        skipped = 0
        
        for filename in mp3_files:
            # Parse filename (usually "Artist - Track.mp3")
            name_without_ext = os.path.splitext(filename)[0]
            
            if " - " in name_without_ext:
                parts = name_without_ext.split(" - ", 1)
                artist = parts[0].strip()
                track_name = parts[1].strip()
            else:
                artist = "Unknown"
                track_name = name_without_ext
            
            local_path = os.path.join(self.download_dir, filename)
            
            # Check if already in database
            in_db, _ = self.check_track_in_db(track_name, artist)
            
            if in_db:
                print(f"⊘ Already in DB: {artist} - {track_name}")
                skipped += 1
                continue
            
            # Add to database
            track_id = self.add_track_to_db(track_name, artist, local_path)
            if track_id:
                print(f"✓ Added: {artist} - {track_name}")
                added += 1
        
        print("-" * 60)
        print(f"Added: {added} tracks")
        print(f"Skipped: {skipped} tracks (already in DB)")
        print(f"Total in folder: {len(mp3_files)} tracks\n")
    
    def process_user_prompt(self, user_prompt):
        """
        Main pipeline function:
        1. Get track suggestions from LLM
        2. Check which tracks are in DB
        3. Download missing tracks
        4. Add to DB
        5. Return list of all tracks (ready for next stage)
        """
        print(f"\n{'='*60}")
        print(f"PROCESSING PROMPT: {user_prompt}")
        print(f"{'='*60}\n")
        
        # Step 1: Get suggestions from LLM
        print("Consulting LLM for track suggestions...")
        try:
            suggested_tracks = self.get_track_suggestions_from_llm(user_prompt)
            print(f"✓ LLM suggested {len(suggested_tracks)} tracks\n")
        except Exception as e:
            print(f"✗ Failed to get track suggestions: {e}")
            return []
        
        # Step 2 & 3: Check DB and download if needed
        processed_tracks = []
        for i, track in enumerate(suggested_tracks, 1):
            track_name = track.get('track_name', 'Unknown')
            artist = track.get('artist', 'Unknown')
            
            print(f"[{i}/{len(suggested_tracks)}] Processing: {track_name} by {artist}")
            
            # Check if in DB
            in_db, db_record = self.check_track_in_db(track_name, artist)
            
            if in_db:
                print(f"  ✓ Found in database")
                processed_tracks.append({
                    'track_name': track_name,
                    'artist': artist,
                    'local_path': db_record[4],  # local_path column
                    'energy_level': track.get('energy_level'),
                    'genre': track.get('genre'),
                    'source': 'database'
                })
            else:
                print(f"  ✗ Not in database, downloading...")
                # Download track
                local_path = self.download_track(
                    track_name, 
                    artist, 
                    track.get('spotify_url')
                )
                
                if local_path:
                    # Add to database
                    self.add_track_to_db(
                        track_name,
                        artist,
                        local_path,
                        track.get('spotify_url'),
                        track.get('energy_level'),
                        track.get('genre')
                    )
                    processed_tracks.append({
                        'track_name': track_name,
                        'artist': artist,
                        'local_path': local_path,
                        'energy_level': track.get('energy_level'),
                        'genre': track.get('genre'),
                        'source': 'downloaded'
                    })
                else:
                    print(f"  ✗ Download failed")
            
            print()  # Empty line for readability
        
        print(f"{'='*60}")
        print(f"PIPELINE STAGE 1 COMPLETE!")
        print(f"Processed {len(processed_tracks)} tracks successfully")
        print(f"Audio files stored in: {self.download_dir}/")
        print(f"Database stored at: {self.db_path}")
        print(f"{'='*60}\n")
        
        return processed_tracks
    
    def view_database(self):
        """Display all tracks in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, track_name, artist, local_path, energy_level, genre FROM tracks")
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            print("Database is empty!")
            return
        
        print(f"\n{'='*60}")
        print(f"DATABASE CONTENTS ({len(rows)} tracks)")
        print(f"{'='*60}\n")
        
        for row in rows:
            print(f"ID: {row[0]}")
            print(f"  Track: {row[1]}")
            print(f"  Artist: {row[2]}")
            print(f"  Path: {row[3]}")
            print(f"  Energy: {row[4]}")
            print(f"  Genre: {row[5]}")
            print()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Configuration
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyC_g4y4AnxW13_JoyPFuLXcS23oaWdgx7g")
    SPOTIFY_CLIENT_ID = "9e04c40b14e14a8aad50fda3d0f01913"
    SPOTIFY_CLIENT_SECRET = "2daadce09a2345bd82013ada0c1dc216"
    
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not set!")
        print("Set it with: export GEMINI_API_KEY='your_key'")
        exit(1)
    
    # Initialize pipeline
    print("\n🎵 Music Pipeline Stage 1 - Track Discovery & Download")
    print("=" * 60)
    
    pipeline = MusicPipelineStage1(
        gemini_api_key=GEMINI_API_KEY,
        spotify_client_id=SPOTIFY_CLIENT_ID,
        spotify_client_secret=SPOTIFY_CLIENT_SECRET,
        db_path="music_tracks.db",
        download_dir="downloaded_music"
    )
    
    # Option 1: Populate database from existing files
    print("\nOption: Populate database from existing downloads")
    response = input("Do you want to scan downloaded_music/ and add existing files to database? (y/n): ")
    if response.lower() == 'y':
        pipeline.populate_db_from_folder()
    
    # Option 2: Process user prompt
    print("\nOption: Process new music request")
    user_prompt = input("Enter your music request (or press Enter to skip): ")
    
    if user_prompt.strip():
        tracks = pipeline.process_user_prompt(user_prompt)
        
        # Display results
        print("\n📋 Tracks ready for next stage:")
        for i, track in enumerate(tracks, 1):
            print(f"{i}. {track['track_name']} by {track['artist']}")
            print(f"   Path: {track['local_path']}")
            print(f"   Energy: {track['energy_level']}")
            print(f"   Source: {track['source']}")
            print()
    
    # Display database contents
    print("\n" + "="*60)
    pipeline.view_database()
    
    print("\n✅ Ready to forward to Stage 2: Feature Extraction")
    print("=" * 60)
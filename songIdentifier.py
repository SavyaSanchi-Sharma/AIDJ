import os
import re
import json
import time
import sqlite3
import logging
from typing import Optional, List

from spotdl import Spotdl
from dotenv import load_dotenv
import google.generativeai as genai

# Optional: Spotipy exception class (spotdl uses spotipy under the hood)
try:
    import spotipy
    from spotipy.exceptions import SpotifyException
except Exception:
    SpotifyException = Exception  # fallback if spotipy not importable

# ---------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------
load_dotenv("project.env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# ---------------------------------------------------------------------
# MusicPipelineStage1
# ---------------------------------------------------------------------
class MusicPipelineStage1:
    """
    Stage 1: Generate track list via Gemini, download using SpotDL,
    and maintain metadata in an SQLite database.
    """

    SPOTIFY_TRACK_RE = re.compile(r"^https?://open\.spotify\.com/track/([A-Za-z0-9]+)")

    def __init__(self,
                 gemini_api_key: str,
                 spotify_client_id: str,
                 spotify_client_secret: str,
                 db_path: str = "music_tracks.db",
                 download_dir: str = "downloaded_music",
                 model_name: str = "gemini-2.5-flash"):
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not set in environment.")
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_name)

        self.db_path = db_path
        self.download_dir = os.path.abspath(download_dir)
        os.makedirs(self.download_dir, exist_ok=True)

        # SpotDL init: single thread to reduce rate-limit pressure.
        self.spotdl = Spotdl(
            client_id=spotify_client_id,
            client_secret=spotify_client_secret,
            downloader_settings={
                "output": self.download_dir,
                "format": "mp3",
                "threads": 1,              # keep single threaded
                # "cookies": "cookies.txt" # optional: use authenticated cookies for more quota
            },
        )

        self._init_database()
        logging.info("MusicPipelineStage1 ready. Downloads -> %s", self.download_dir)

    # -----------------------------------------------------------------
    # Utility methods
    # -----------------------------------------------------------------
    def _normalize(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        return text.strip().lower().replace("’", "'").replace("`", "'")

    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_name TEXT NOT NULL,
                artist TEXT NOT NULL,
                spotify_url TEXT,
                local_path TEXT,
                energy_level REAL,
                genre TEXT,
                added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(track_name, artist)
            )
            """
        )
        conn.commit()
        conn.close()
        logging.info("Database initialized: %s", self.db_path)

    # -----------------------------------------------------------------
    # LLM → JSON Parsing (more robust)
    # -----------------------------------------------------------------
    def parse_llm_json_response(self, response_text: str):
        if not response_text:
            raise ValueError("Empty LLM response.")

        text = response_text.strip()
        # Remove triple-backtick wrappers
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

        # Try to find the first JSON array/object substring
        json_match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        # Remove trailing commas before closing braces/brackets
        text = re.sub(r",(\s*[}\]])", r"\1", text)

        # Try several parsing strategies
        for attempt_text in (text, text.replace("'", '"')):
            try:
                parsed = json.loads(attempt_text)
                return parsed
            except Exception:
                continue

        # As a last resort try to extract top-level start/end
        start_idx = min([i for i in [text.find("["), text.find("{")] if i != -1] or [0])
        end_idx = max([i for i in [text.rfind("]"), text.rfind("}")] if i != -1] or [len(text) - 1])
        if start_idx < 0 or end_idx < 0 or end_idx < start_idx:
            raise ValueError("Could not parse LLM JSON response.")
        snippet = text[start_idx:end_idx + 1]
        # final attempt
        return json.loads(snippet.replace("'", '"'))

    # -----------------------------------------------------------------
    # Gemini LLM Track Suggestions
    # -----------------------------------------------------------------
    def get_track_suggestions_from_llm(self, user_prompt: str) -> List[dict]:
        prompt = f"""
        You are a music expert assistant. When given a user's description
        of the type of music they want, suggest 2 songs that match.

        Respond with ONLY a valid JSON array, nothing else.
        Example:
        [
          {{
            "track_name": "Song Title",
            "artist": "Artist Name",
            "spotify_url": "",
            "energy_level": 0.8,
            "genre": "pop"
          }}
        ]

        User request: {user_prompt}
        """

        response = self.model.generate_content(prompt)
        response_text = response.text
        logging.info("Raw LLM response (truncated): %s", response_text[:300])
        tracks = self.parse_llm_json_response(response_text)

        if not isinstance(tracks, list) or len(tracks) == 0:
            raise ValueError("Invalid or empty LLM track list.")
        # Basic normalization: ensure minimal keys exist
        normalized = []
        for t in tracks:
            if not isinstance(t, dict):
                continue
            normalized.append({
                "track_name": t.get("track_name", "").strip(),
                "artist": t.get("artist", "").strip(),
                "spotify_url": t.get("spotify_url") or "",
                "energy_level": t.get("energy_level"),
                "genre": t.get("genre")
            })
        return normalized

    # -----------------------------------------------------------------
    # Database Ops
    # -----------------------------------------------------------------
    def check_track_in_db(self, track_name: str, artist: str):
        track_name = self._normalize(track_name)
        artist = self._normalize(artist)
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM tracks WHERE track_name=? AND artist=?",
            (track_name, artist)
        )
        result = cur.fetchone()
        conn.close()
        return result is not None, result

    def add_track_to_db(self, track_name: str, artist: str, local_path: Optional[str],
                        spotify_url: Optional[str] = None, energy_level: Optional[float] = None,
                        genre: Optional[str] = None):
        track_name_n = self._normalize(track_name)
        artist_n = self._normalize(artist)

        if local_path is None:
            filename = f"{artist_n} - {track_name_n}.mp3"
            local_path = os.path.join(self.download_dir, filename)

        local_path = os.path.abspath(local_path)

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO tracks
                (track_name, artist, spotify_url, local_path, energy_level, genre)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (track_name_n, artist_n, spotify_url, local_path, energy_level, genre),
            )
            conn.commit()
            logging.info("Added to DB: %s - %s", artist, track_name)
        except sqlite3.IntegrityError:
            cur.execute(
                """
                UPDATE tracks SET local_path=?, spotify_url=?, energy_level=?, genre=?
                WHERE track_name=? AND artist=?
                """,
                (local_path, spotify_url, energy_level, genre, track_name_n, artist_n),
            )
            conn.commit()
            logging.info("Updated existing DB entry: %s - %s", artist, track_name)
        finally:
            conn.close()

    # -----------------------------------------------------------------
    # File Detection & Download helpers
    # -----------------------------------------------------------------
    def find_downloaded_file(self, track_name: str, artist: str, wait_time: float = 3) -> Optional[str]:
        """
        Find downloaded file matching track and artist.
        Prioritizes exact substring matches; does not fallback blindly.
        """
        if wait_time and wait_time > 0:
            time.sleep(wait_time)

        audio_extensions = ('.mp3', '.wav', '.m4a', '.flac')
        try:
            files = [f for f in os.listdir(self.download_dir) if f.lower().endswith(audio_extensions)]
        except FileNotFoundError:
            return None

        if not files:
            return None

        track_lower = self._normalize(track_name)
        artist_lower = self._normalize(artist)

        # 1. Exact artist-track match first
        for f in files:
            fname = f.lower()
            if artist_lower and track_lower and artist_lower in fname and track_lower in fname:
                return os.path.join(self.download_dir, f)

        # 2. Partial track match
        for f in files:
            if track_lower and track_lower in f.lower():
                return os.path.join(self.download_dir, f)

        # 3. Partial artist match
        for f in files:
            if artist_lower and artist_lower in f.lower():
                return os.path.join(self.download_dir, f)

        # 4. No match found
        return None

    def _is_valid_spotify_track_url(self, url: Optional[str]) -> bool:
        if not url or not isinstance(url, str):
            return False
        return bool(self.SPOTIFY_TRACK_RE.search(url.strip()))

    # -----------------------------------------------------------------
    # download_track (robust with retries, backoff, validation)
    # -----------------------------------------------------------------
    def download_track(self, track_name: str, artist: str, spotify_url: Optional[str] = None):
        print("Searching:", track_name, "by", artist)

        # ---------------------------------------------------------
        # 100% ignore the LLM-provided spotify_url (hallucinates IDs)
        # Force SpotDL to search only via textual query
        # ---------------------------------------------------------
        query = f"{track_name} {artist}".strip()
        print("Final query used for SpotDL search:", query)

        # Try searching
        search_results = None
        try:
            search_results = self.spotdl.search([query])
        except Exception as e:
            print("Search error:", e)
            return None

        if not search_results:
            print("No results:", query)
            return None

        # Check if already exists
        existing = self.find_downloaded_file(track_name, artist, wait_time=0)
        if existing:
            print("Already downloaded:", existing)
            return existing

        # Attempt download
        try:
            self.spotdl.downloader.download_multiple_songs(search_results)
        except Exception as e:
            print("Download error:", e)
            return None

        # Check file
        path = self.find_downloaded_file(track_name, artist)
        if path:
            print("Downloaded:", os.path.basename(path))
            return path

        print("Download finished, but file not found.")
        return None

    # -----------------------------------------------------------------
    # Main pipeline
    # -----------------------------------------------------------------
    def process_user_prompt(self, user_prompt: str):
        logging.info("=" * 60)
        logging.info("Processing prompt: %s", user_prompt)
        logging.info("=" * 60)

        try:
            tracks = self.get_track_suggestions_from_llm(user_prompt)
        except Exception as e:
            logging.error("LLM error: %s", e)
            return []

        processed = []
        for t in tracks:
            name = t.get("track_name", "unknown")
            artist = t.get("artist", "unknown")
            logging.info("Track: %s by %s", name, artist)

            in_db, record = self.check_track_in_db(name, artist)
            if in_db and record and record[4]:
                logging.info("Found in DB: %s", record[4])
                processed.append({
                    "track_name": name,
                    "artist": artist,
                    "local_path": record[4],
                    "energy_level": t.get("energy_level"),
                    "genre": t.get("genre"),
                    "source": "database"
                })
                continue

            # small delay to avoid hitting APIs too fast (configurable)
            time.sleep(1.0)

            path = self.download_track(name, artist, t.get("spotify_url"))
            # Always add/update DB — even if path is None (keeps metadata)
            self.add_track_to_db(name, artist, path, t.get("spotify_url"),
                                 t.get("energy_level"), t.get("genre"))
            processed.append({
                "track_name": name,
                "artist": artist,
                "local_path": path,
                "energy_level": t.get("energy_level"),
                "genre": t.get("genre"),
                "source": "downloaded" if path else "metadata_only"
            })

        logging.info("\nStage 1 complete. Processed %d tracks.", len(processed))
        logging.info("Files in: %s", self.download_dir)
        logging.info("Database: %s", self.db_path)
        return processed

    # -----------------------------------------------------------------
    def view_database(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT id, track_name, artist, local_path FROM tracks")
        rows = cur.fetchall()
        conn.close()
        if not rows:
            logging.info("Database empty.")
            return
        logging.info("=" * 60)
        for r in rows:
            logging.info(f"{r[0]}. {r[2]} - {r[1]} ({r[3]})")
        logging.info("=" * 60)


# ---------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------
def get_songs(user_input: str):
    logging.info("\nStage 1 - Track Discovery & Download")
    logging.info("=" * 60)

    pipeline = MusicPipelineStage1(
        gemini_api_key=GEMINI_API_KEY,
        spotify_client_id=SPOTIFY_CLIENT_ID,
        spotify_client_secret=SPOTIFY_CLIENT_SECRET,
    )

    tracks = []
    if user_input.strip():
        tracks = pipeline.process_user_prompt(user_input)

        logging.info("\nTracks ready for next stage:")
        for t in tracks:
            logging.info("-", t["artist"], "-", t["track_name"], "|", t["local_path"])

    logging.info("\nDatabase Summary:")
    pipeline.view_database()

    names = [t["track_name"] for t in tracks]
    return tracks, names

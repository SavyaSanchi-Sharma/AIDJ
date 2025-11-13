
import os
import re
import json
import time
import sqlite3
from spotdl import Spotdl
from dotenv import load_dotenv
import google.generativeai as genai

# ---------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------
load_dotenv("project.env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")


# ---------------------------------------------------------------------
# MusicPipelineStage1
# ---------------------------------------------------------------------
class MusicPipelineStage1:
    """
    Stage 1: Generate track list via Gemini, download using SpotDL,
    and maintain metadata in an SQLite database.
    """

    def __init__(self,
                 gemini_api_key,
                 spotify_client_id,
                 spotify_client_secret,
                 db_path="music_tracks.db",
                 download_dir="downloaded_music",
                 model_name="gemini-2.5-flash"):

        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_name)
        self.db_path = db_path
        self.download_dir = os.path.abspath(download_dir)

        os.makedirs(self.download_dir, exist_ok=True)

        self.spotdl = Spotdl(
            client_id=spotify_client_id,
            client_secret=spotify_client_secret,
            downloader_settings={
                "output": self.download_dir,
                "format": "mp3",
                "threads": 4
            },
        )

        self._init_database()

    # -----------------------------------------------------------------
    # Utility methods
    # -----------------------------------------------------------------
    def _normalize(self, text: str) -> str:
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
        print("Database initialized:", self.db_path)

    # -----------------------------------------------------------------
    # LLM → JSON Parsing
    # -----------------------------------------------------------------
    def parse_llm_json_response(self, response_text: str):
        text = response_text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        json_match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        text = re.sub(r",(\s*[}\]])", r"\1", text)

        try:
            return json.loads(text)
        except Exception:
            try:
                return json.loads(text.replace("'", '"'))
            except Exception:
                start = min([i for i in [text.find("["), text.find("{")] if i != -1] or [0])
                end = max([i for i in [text.rfind("]"), text.rfind("}")] if i != -1] or [len(text)])
                return json.loads(text[start:end + 1])

    # -----------------------------------------------------------------
    # Gemini LLM Track Suggestions
    # -----------------------------------------------------------------
    def get_track_suggestions_from_llm(self, user_prompt: str):
        prompt = f"""
        You are a music expert assistant. When given a user's description
        of the type of music they want, suggest 1-3 songs that match.

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
        print("Raw LLM response:", response_text[:300])
        tracks = self.parse_llm_json_response(response_text)

        if not isinstance(tracks, list) or len(tracks) == 0:
            raise ValueError("Invalid or empty LLM track list.")
        return tracks

    # -----------------------------------------------------------------
    # Database Ops
    # -----------------------------------------------------------------
    def check_track_in_db(self, track_name, artist):
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

    def add_track_to_db(self, track_name, artist, local_path,
                        spotify_url=None, energy_level=None, genre=None):
        track_name = self._normalize(track_name)
        artist = self._normalize(artist)

        if local_path is None:
            filename = f"{artist} - {track_name}.mp3"
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
                (track_name, artist, spotify_url, local_path, energy_level, genre),
            )
            conn.commit()
            print("Added to DB:", artist, "-", track_name)
        except sqlite3.IntegrityError:
            cur.execute(
                """
                UPDATE tracks SET local_path=?, spotify_url=?, energy_level=?, genre=?
                WHERE track_name=? AND artist=?
                """,
                (local_path, spotify_url, energy_level, genre, track_name, artist),
            )
            conn.commit()
            print("Updated existing DB entry:", artist, "-", track_name)
        finally:
            conn.close()

    # -----------------------------------------------------------------
    # File Detection & Download
    # -----------------------------------------------------------------
    def find_downloaded_file(self, track_name, artist, wait_time=3):
            """
            Find downloaded file matching track and artist.
            Prioritizes exact substring matches; does not fallback blindly.
            """
            time.sleep(wait_time)
            audio_extensions = ('.mp3', '.wav')
            files = [f for f in os.listdir(self.download_dir) if f.endswith(audio_extensions)]
            if not files:
                return None

            track_lower = self._normalize(track_name)
            artist_lower = self._normalize(artist)

            # 1. Exact artist-track match first
            for f in files:
                fname = f.lower()
                if artist_lower in fname and track_lower in fname:
                    return os.path.join(self.download_dir, f)

            # 2. Partial track match
            for f in files:
                if track_lower in f.lower():
                    return os.path.join(self.download_dir, f)

            # 3. Partial artist match
            for f in files:
                if artist_lower in f.lower():
                    return os.path.join(self.download_dir, f)

            # 4. No match found
            return None

    def download_track(self, track_name, artist, spotify_url=None):
        print("Searching:", track_name, "by", artist)
        query = spotify_url if spotify_url else f"{track_name} {artist}"
        search_results = self.spotdl.search([query])
        if not search_results:
            print("No results for", track_name)
            return None

        existing = self.find_downloaded_file(track_name, artist, wait_time=0)
        if existing:
            print("File already exists:", existing)
            return existing

        try:
            self.spotdl.downloader.download_multiple_songs(search_results)
        except Exception as e:
            print("Download error:", e)
            return None

        path = self.find_downloaded_file(track_name, artist)
        if path:
            print("Downloaded:", os.path.basename(path))
        else:
            print("Download finished but file not found.")
        return path

    # -----------------------------------------------------------------
    # Main pipeline
    # -----------------------------------------------------------------
    def process_user_prompt(self, user_prompt):
        print("=" * 60)
        print("Processing prompt:", user_prompt)
        print("=" * 60)

        try:
            tracks = self.get_track_suggestions_from_llm(user_prompt)
        except Exception as e:
            print("LLM error:", e)
            return []

        processed = []
        for t in tracks:
            name = t.get("track_name", "unknown")
            artist = t.get("artist", "unknown")
            print("\nTrack:", name, "by", artist)

            in_db, record = self.check_track_in_db(name, artist)
            if in_db and record and record[4]:
                print("Found in DB:", record[4])
                processed.append({
                    "track_name": name,
                    "artist": artist,
                    "local_path": record[4],
                    "energy_level": t.get("energy_level"),
                    "genre": t.get("genre"),
                    "source": "database"
                })
                continue

            path = self.download_track(name, artist, t.get("spotify_url"))
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

        print("\nStage 1 complete. Processed", len(processed), "tracks.")
        print("Files in:", self.download_dir)
        print("Database:", self.db_path)
        return processed

    # -----------------------------------------------------------------
    def view_database(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT id, track_name, artist, local_path FROM tracks")
        rows = cur.fetchall()
        conn.close()
        if not rows:
            print("Database empty.")
            return
        print("=" * 60)
        for r in rows:
            print(f"{r[0]}. {r[2]} - {r[1]} ({r[3]})")
        print("=" * 60)


# ---------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------
def get_songs(user_input: str):
    print("\nStage 1 - Track Discovery & Download")
    print("=" * 60)

    pipeline = MusicPipelineStage1(
        gemini_api_key=GEMINI_API_KEY,
        spotify_client_id=SPOTIFY_CLIENT_ID,
        spotify_client_secret=SPOTIFY_CLIENT_SECRET,
    )

    tracks = []
    if user_input.strip():
        tracks = pipeline.process_user_prompt(user_input)

        print("\nTracks ready for next stage:")
        for t in tracks:
            print("-", t["artist"], "-", t["track_name"], "|", t["local_path"])

    print("\nDatabase Summary:")
    pipeline.view_database()

    names = [t["track_name"] for t in tracks]
    return tracks, names

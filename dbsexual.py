# dbsexual.py
import sqlite3
import os

DB_PATH = "music_tracks.db"


def _normalize(text: str) -> str:
    """Normalize strings to match how they're stored in the DB."""
    return text.strip().lower().replace("’", "'").replace("`", "'")


def _get_connection():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    return sqlite3.connect(DB_PATH)


def trackExists(name: str, artist: str = None) -> bool:
    """
    Check if a track exists in the database.
    If artist is not given, searches only by track name.
    """
    name = _normalize(name)
    conn = _get_connection()
    cur = conn.cursor()
    if artist:
        artist = _normalize(artist)
        cur.execute("SELECT 1 FROM tracks WHERE track_name=? AND artist=?", (name, artist))
    else:
        cur.execute("SELECT 1 FROM tracks WHERE track_name=?", (name,))
    exists = cur.fetchone() is not None
    conn.close()
    return exists


def getFilePath(name: str, artist: str = None):
    """
    Return local_path for a track; returns None if not found or path missing.
    """
    name = _normalize(name)
    conn = _get_connection()
    cur = conn.cursor()
    if artist:
        artist = _normalize(artist)
        cur.execute("SELECT local_path FROM tracks WHERE track_name=? AND artist=?", (name, artist))
    else:
        cur.execute("SELECT local_path FROM tracks WHERE track_name=?", (name,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row and row[0] and os.path.exists(row[0]) else None

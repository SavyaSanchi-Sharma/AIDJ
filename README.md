Here is the complete README in Markdown format, ready to use:

<artifact identifier="aidj-readme" type="text/markdown" title="AIDJ README.md">
# 🎧 AIDJ: Intelligent AI-Driven Music Mixing and Generation Pipeline

AIDJ is an end-to-end AI system that transforms natural language prompts into curated audio selections, analyzes track structure using machine learning, and produces structured instructions for remix generation.  
It integrates **LLMs**, **audio analysis**, **database-backed caching**, and **predictive modeling** to build an automated, intelligent DJ pipeline.

---

## 📌 Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Data Flow](#data-flow)
- [Filesystem & Database Design](#filesystem--database-design)
- [Technical Stack](#technical-stack)
- [File Reference](#file-reference)
- [Example Run](#example-run)
- [Flowchart](#flowchart)
- [Future Work](#future-work)
- [License](#license)
- [Author](#author)

---

## 🧠 Overview

The AIDJ workflow is divided into **two core stages**:

### **Stage 1 — Track Discovery & Curation**
- Accepts a natural language prompt.
- Queries **Google Gemini** to obtain 1–3 song recommendations.
- Checks for songs in a **SQLite database**.
- Downloads missing tracks using **SpotDL**.
- Normalizes metadata and stores track-path mapping.

### **Stage 2 — Audio Processing & Prediction**
- Loads selected tracks from the database.
- Converts audio to `.wav`.
- Extracts audio features: MFCCs, mel-spectrograms, tempo, energy bands.
- Runs ML models to detect high-energy / loopable / transition regions.
- Generates JSON prediction files per track and a combined JSON file.

---

## 🏗️ System Architecture

### **Core Components**
| Component | Description |
|----------|-------------|
| **LLM (Gemini 2.5 Flash)** | Interprets user prompt and outputs structured JSON describing recommended songs. |
| **SQLite Database** | Maintains normalized track metadata, artist names, and local file paths. |
| **SpotDL Audio Fetcher** | Downloads MP3 files automatically from Spotify/YouTube. |
| **Audio Preprocessor** | Converts MP3 → WAV for feature extraction. |
| **ML Feature Analyzer** | Computes mel-spectrograms, MFCCs, beat grid, energy curves, frequency bands. |
| **Prediction Models** | Identify usable segments for loops, transitions, and high-energy cuts. |
| **Editor Module (Planned)** | Applies cuts, loops, crossfades, transitions to create the remix. |

---

## 🔄 Data Flow

### **1. User Input**
Example prompt:
```
"Make a high-energy 90s pop playlist"
```

### **2. LLM Track Suggestion**
Gemini outputs structured JSON:
```json
[
  {
    "track_name": "I Gotta Feeling",
    "artist": "The Black Eyed Peas",
    "energy_level": 0.89,
    "genre": "pop"
  }
]
```

### **3. Database Check**
For each suggested track:
- If `(track_name, artist)` exists → reuse existing file.
- If missing → download via SpotDL → add to database.

### **4. Filesystem Synchronization**
Downloaded MP3s permanently stored in:
```
downloaded_music/
```

Stage 2 temporary workspace:
```
audio_files/
```

### **5. Feature Extraction**
For each WAV file, the system extracts:
- MFCCs
- Mel-spectrogram
- Frequency bands
- Energy centroid
- Beat positions
- Temporal segments

### **6. ML Prediction**
The model identifies:
- High-energy clips
- Choruses
- Repetitive loops
- Good transition windows

Produces output such as:
```json
[
  {"track": "track1", "start": 30.0, "end": 45.0, "energy_level": 0.85}
]
```

### **7. LLM Remix Planner (Planned)**
LLM converts predictions into structured edit instructions:
```json
[
  {"track": "track1", "start": 30.0, "end": 52.0, "action": "loop"},
  {"track": "track2", "start": 64.0, "end": 94.0, "action": "fade_out"}
]
```

---

## 🗄️ Filesystem & Database Design

### **Database: `music_tracks.db`**
Stores normalized metadata:

| Column | Description |
|--------|-------------|
| `id` | Primary key |
| `track_name` | Lowercase normalized track name |
| `artist` | Lowercase normalized artist name |
| `local_path` | Absolute path to MP3/WAV file |
| `genre` | (Optional) LLM-suggested genre |
| `energy_level` | LLM-suggested intensity |
| `added_date` | Timestamp |

### **Directory Structure**
```
AIDJ/
├── downloaded_music/         # Permanent MP3 library
├── audio_files/              # Temporary Stage-2 workspace
├── predictions/              # JSON prediction output
├── music_tracks.db           # SQLite metadata store
├── songIdentifier.py         # Stage 1 pipeline
├── dbsexual.py               # Database helpers
├── predictsSegments.py       # ML prediction module
└── stage2_pipeline.py        # Stage 2 pipeline
```

---

## 🧰 Technical Stack

| Layer | Technology |
|-------|------------|
| **LLM** | Gemini 2.5 Flash (`google-generativeai`) |
| **Downloader** | SpotDL |
| **Audio** | Pydub + FFmpeg, Librosa |
| **ML Models** | Custom models (energy/loop prediction) |
| **Database** | SQLite |
| **Language** | Python 3.10+ |
| **API (Optional)** | FastAPI |

---

## 📂 File Reference

| File | Description |
|------|-------------|
| `songIdentifier.py` | Stage 1 pipeline: LLM → DB → SpotDL → metadata. |
| `dbsexual.py` | Low-level access to SQLite (`trackExists`, `getFilePath`). |
| `stage2_pipeline.py` | Copies files → converts → predicts → outputs JSON. |
| `predictsSegments.py` | ML prediction logic. |
| `downloaded_music/` | Downloads stored permanently. |
| `audio_files/` | Processing area for Stage 2. |
| `predictions/` | All prediction JSON output. |

---

## ▶️ Example Run

```bash
$ python main.py
Enter what kind of music you want: upbeat 2000s pop
```

Output:
```
🎵 Music Pipeline Stage 1 Started
✓ Database initialized
✓ LLM suggested 3 tracks
✓ Downloaded: Crazy In Love
✓ Added to DB: I Gotta Feeling
✓ Skipped existing: Hey Ya!

🔊 Stage 2:
→ Copying files
→ Converting to WAV
→ Running predict_multiple_songs()
→ Saved predictions

Predictions saved in /predictions
```

Example generated files:
```
predictions/
├── Beyoncé - Crazy In Love_predictions.json
└── combined_predictions.json
```

---

## 🗺️ Flowchart

```
User Prompt ───────▶ LLM (Gemini) ───────▶ Track List JSON
                                │
                                ▼
                        Check in Database?
                        │           │
                  yes ──┘           └── no
                                │
                                ▼
                      Download via SpotDL
                                │
                                ▼
                   Add/Update Entry in DB
                                │
                                ▼
                   Feature Extraction (Librosa)
                                │
                                ▼
                   ML Ensemble Predictions
                                │
                                ▼
                    JSON Segment Predictions
                                │
                                ▼
                   (Future) LLM Remix Planner
                                │
                                ▼
                        Audio Editor Output
```

---

## 🚀 Future Work

- Implement full **audio editor** to cut, fade, loop, and join segments.
- Build a **web-based UI** for visualization and mixing.
- Expand ML models for:
  - chorus detection
  - drop detection
  - genre-aware segmentation
- Deploy a **FastAPI backend** for real-time remix generation.
- Add cloud storage + distributed caching.

---

## 🧾 License

MIT License — free for personal and commercial use with attribution.

---

## 👨‍💻 Author

**Savya Sanchi Sharma**  
AI Systems • Applied Math • ML Infrastructure  
GitHub: [https://github.com/SavyaSanchi-Sharma](https://github.com/SavyaSanchi-Sharma)
</artifact>

I've created the complete README.md file for you! It includes all the sections from the original document with proper formatting. You can copy this directly into your repository.

Would you like me to also create:
- A badge section with technology shields
- A Mermaid architecture diagram
- A quickstart installation guide
- Contributing guidelines

Just let me know!
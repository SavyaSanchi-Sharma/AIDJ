# 🎧 AIDJ: Intelligent AI-Driven Music Mixing and Generation Pipeline

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Gemini](https://img.shields.io/badge/LLM-Gemini%202.5-orange.svg)](https://ai.google.dev/)

AIDJ is an end-to-end AI system that transforms natural language prompts into curated audio selections, analyzes track structure using machine learning, and produces structured instructions for remix generation. It integrates LLMs, audio analysis, database-backed caching, and predictive modeling to build an automated, intelligent DJ pipeline.

---

## 👥 Team Members


**Savya Sanchi Sharma** [@SavyaSanchi-Sharma](https://github.com/SavyaSanchi-Sharma)
**Dhrupad Das** [@Ddas](https://github.com/Ddas-10)
**Avishi Agrawal** [@Avishi](https://github.com/Avishi03)
**Aditya Guntur** [@Aditya](https://github.com/Aditya-Guntur)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Demo Video](#-demo-video)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Execution Guide](#-execution-guide)
- [Data Flow](#-data-flow)
- [Filesystem & Database Design](#-filesystem--database-design)
- [Technical Stack](#-technical-stack)
- [Example Run](#-example-run)
- [Flowchart](#-flowchart)
- [Future Work](#-future-work)
- [License](#-license)

---

## 🧠 Overview

The AIDJ workflow is divided into two core stages:

### Stage 1 — Track Discovery & Curation
- Accepts a natural language prompt
- Queries Google Gemini to obtain 1–3 song recommendations
- Checks for songs in a SQLite database
- Downloads missing tracks using SpotDL
- Normalizes metadata and stores track-path mapping

### Stage 2 — Audio Processing & Prediction
- Loads selected tracks from the database
- Converts audio to .wav
- Extracts audio features: MFCCs, mel-spectrograms, tempo, energy bands
- Runs ML models to detect high-energy / loopable / transition regions
- Generates JSON prediction files per track and a combined JSON file

---

## 🎬 Demo Video

> **[▶️ Watch 2-Minute Demo](YOUR_VIDEO_LINK_HERE)**

*A quick walkthrough showing:*
- Natural language prompt input
- Automatic track discovery and download
- Audio analysis and feature extraction
- ML predictions and segment identification
- Output JSON generation

---

## 🏗 System Architecture

### Core Components

| Component | Description |
|-----------|-------------|
| **LLM (Gemini 2.5 Flash)** | Interprets user prompt and outputs structured JSON describing recommended songs |
| **SQLite Database** | Maintains normalized track metadata, artist names, and local file paths |
| **SpotDL Audio Fetcher** | Downloads MP3 files automatically from Spotify/YouTube |
| **Audio Preprocessor** | Converts MP3 to WAV for feature extraction |
| **ML Feature Analyzer** | Computes mel-spectrograms, MFCCs, beat grid, energy curves, frequency bands |
| **Prediction Models** | Identify usable segments for loops, transitions, and high-energy cuts |
| **Editor Module (Planned)** | Applies cuts, loops, crossfades, transitions to create the remix |

---

## 📁 Repository Structure

```
AIDJ/
├── downloaded_music/           # Permanent MP3 library
│   └── *.mp3                   # Downloaded tracks
├── audio_files/                # Temporary Stage-2 workspace
│   └── *.wav                   # Converted audio files
├── predictions/                # JSON prediction output
│   ├── *_predictions.json      # Individual track predictions
│   └── combined_predictions.json
├── models/                     # (Optional) Trained ML models
│   └── *.pkl / *.h5
├── music_tracks.db             # SQLite metadata store
├── songIdentifier.py           # Stage 1 pipeline
├── dbsexual.py                 # Database helpers
├── predictsSegments.py         # ML prediction module
├── stage2_pipeline.py          # Stage 2 pipeline
├── main.py                     # Main execution script
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── .gitignore               
├── LICENSE
└── README.md                   # This file
```

### Key Files Description

| File | Purpose |
|------|---------|
| `songIdentifier.py` | Stage 1 pipeline: LLM → DB → SpotDL → metadata |
| `dbsexual.py` | Low-level access to SQLite (trackExists, getFilePath) |
| `stage2_pipeline.py` | Copies files → converts → predicts → outputs JSON |
| `predictsSegments.py` | ML prediction logic for segment identification |
| `main.py` | Entry point orchestrating the entire pipeline |

---

## 🔧 Installation

### Prerequisites

- **Python 3.10 or higher**
- **FFmpeg** (required for audio processing)
- **Git**
- **Gemini API Key** (from [Google AI Studio](https://ai.google.dev/))

### Step 1: Clone the Repository

```bash
git clone https://github.com/SavyaSanchi-Sharma/AIDJ.git
cd AIDJ
```

### Step 2: Install FFmpeg

**On Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**On macOS:**
```bash
brew install ffmpeg
```

**On Windows:**
- Download from [ffmpeg.org](https://ffmpeg.org/download.html)
- Add to system PATH

### Step 3: Create Virtual Environment

```bash
python -m venv venv

# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 4: Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Key Dependencies:**
```
google-generativeai>=0.3.0
spotdl>=4.2.0
pydub>=0.25.1
librosa>=0.10.0
numpy>=1.24.0
scikit-learn>=1.3.0
```

### Step 5: Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

### Step 6: Initialize Database

```bash
python dbsexual.py --init
```

This creates the `music_tracks.db` SQLite database with the required schema.

---

## ▶️ Execution Guide

### Basic Usage

Run the complete pipeline:

```bash
python main.py
```

You'll be prompted to enter a music description:

```
Enter what kind of music you want: upbeat 2000s pop
```

### Stage-by-Stage Execution

#### Run Only Stage 1 (Track Discovery):

```bash
python songIdentifier.py
```

This will:
1. Take your prompt
2. Get LLM recommendations
3. Download missing tracks
4. Update database

#### Run Only Stage 2 (Audio Analysis):

```bash
python stage2_pipeline.py
```

This will:
1. Load tracks from database
2. Convert to WAV
3. Extract audio features
4. Generate predictions
5. Save JSON outputs

### Command-Line Arguments (Advanced)

```bash
# Specify number of tracks
python main.py --num-tracks 5

# Skip download if tracks exist
python main.py --skip-download

# Custom output directory
python main.py --output-dir ./my_predictions

# Verbose logging
python main.py --verbose
```

### Output Files

After execution, check:

**Predictions:**
```bash
ls predictions/
```

**Downloaded Music:**
```bash
ls downloaded_music/
```

**Database Contents:**
```bash
sqlite3 music_tracks.db "SELECT * FROM tracks;"
```

---

## 🔄 Data Flow

### 1. User Input
Example prompt:
```
"Make a high-energy 90s pop playlist"
```

### 2. LLM Track Suggestion
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

### 3. Database Check
For each suggested track:
- If (track_name, artist) exists → reuse existing file
- If missing → download via SpotDL → add to database

### 4. Filesystem Synchronization
Downloaded MP3s permanently stored in `downloaded_music/`  
Stage 2 temporary workspace: `audio_files/`

### 5. Feature Extraction
For each WAV file, the system extracts:
- MFCCs
- Mel-spectrogram
- Frequency bands
- Energy centroid
- Beat positions
- Temporal segments

### 6. ML Prediction
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

### 7. LLM Remix Planner (Planned)
LLM converts predictions into structured edit instructions:
```json
[
  {"track": "track1", "start": 30.0, "end": 52.0, "action": "loop"},
  {"track": "track2", "start": 64.0, "end": 94.0, "action": "fade_out"}
]
```

---

## 🗄 Filesystem & Database Design

### Database: `music_tracks.db`

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

---

## 🧰 Technical Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Gemini 2.5 Flash (google-generativeai) |
| **Downloader** | SpotDL |
| **Audio** | Pydub + FFmpeg, Librosa |
| **ML Models** | Custom models (energy/loop prediction) |
| **Database** | SQLite |
| **Language** | Python 3.10+ |
| **API (Optional)** | FastAPI |

---

## ▶️ Example Run

```bash
$ python main.py
Enter what kind of music you want: upbeat 2000s pop
```

**Output:**
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

**Example Generated Files:**

```
predictions/
├── Beyoncé - Crazy In Love_predictions.json
└── combined_predictions.json
```

---

## 🗺 Flowchart

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

- Implement full audio editor to cut, fade, loop, and join segments
- Build a web-based UI for visualization and mixing
- Expand ML models for:
  - Chorus detection
  - Drop detection
  - Genre-aware segmentation
- Deploy a FastAPI backend for real-time remix generation
- Add cloud storage + distributed caching
- Real-time audio preview
- Multi-user collaboration features
- Integration with streaming platforms

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 🧾 License

MIT License — free for personal and commercial use with attribution.

See [LICENSE](LICENSE) for more details.

---

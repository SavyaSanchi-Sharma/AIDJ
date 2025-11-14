# AI Music Labeling System

This system automatically analyzes your music library and generates intelligent labels for DJ mixing using GPT-4.

## 🎵 What It Does

**Analyzes Song Structure:**
- Identifies sections: verse, chorus, buildup, outro, drop, break
- Determines energy levels and vocal presence
- Rates mixing suitability for each section

**Finds Mix Points:**  
- Locates optimal DJ transition points
- Identifies intro/outro points for seamless mixing
- Explains why each point works for DJs

**Creates Training Data:**
- Generates labeled datasets for ML model training
- Exports similarity relationships between tracks
- Formats data for recommendation systems

## 🚀 Quick Start

### Step 1: Setup
```bash
cd labelling/scripts
```

Edit `config.json` and add your OpenAI API key:
```json
{
  "openai_api_key": "sk-your-actual-key-here",
  "model": "gpt-4"
}
```

### Step 2: Install Dependencies
```bash
pip install openai librosa soundfile structlog pandas numpy
```

### Step 3: Run Labeling
```bash
python run_labeling.py /path/to/your/music/folder
```

That's it! The system will:
1. Analyze each song's audio features
2. Send structured prompts to GPT-4
3. Generate intelligent labels
4. Save results as JSON files

### Step 4: Process Results
```bash
python process_labels.py output/training_data_final.json --export-format all
```

## 📁 Folder Structure

```
labelling/
├── src/                     # Core modules
│   ├── music_analyzer.py    # Audio feature extraction
│   └── gpt_labeler.py       # GPT-4 integration
├── scripts/                 # User scripts
│   ├── run_labeling.py      # Main script - RUN THIS
│   ├── process_labels.py    # Post-processing utilities
│   └── config.json          # Your API key goes here
└── output/                  # Results folder (auto-created)
    ├── individual_tracks/   # Labels for each song
    ├── combined_results.json # All results
    └── training_data.json   # Formatted for ML
```

## 📊 Output Files

**Individual Track Labels** (`individual_tracks/*.json`):
```json
{
  "track_info": {
    "title": "Song Name", 
    "tempo": 128.0,
    "key": "A"
  },
  "sections": [
    {
      "section_type": "intro",
      "start_time": 0.0,
      "end_time": 16.0,
      "energy_level": "low",
      "has_vocals": false,
      "mix_suitability": "excellent",
      "description": "Perfect intro for mixing in"
    }
  ],
  "mix_points": [
    {
      "time": 16.0,
      "mix_type": "intro",
      "suitability": "excellent",
      "why_good_for_mixing": "Stable beat, no vocals, clear entry point"
    }
  ]
}
```

**Training Data** (`training_data.json`):
- Ready-to-use ML datasets
- Section classification data
- Mix point detection data  
- Track similarity pairs

## 🛠 Processing Options

**Generate Statistics:**
```bash
python process_labels.py training_data.json --export-format statistics
```

**Export for ML Training:**
```bash
python process_labels.py training_data.json --export-format ml_ready
```

**Export to CSV:**
```bash
python process_labels.py training_data.json --export-format csv
```

**Export to DJ Software:**
```bash
python process_labels.py training_data.json --export-format rekordbox
```

## 🎧 For DJs

The system identifies:

**Section Types:**
- `intro` - Perfect for mixing in
- `outro` - Good for mixing out
- `verse` - Usually has vocals, be careful
- `chorus` - High energy, challenging to mix
- `buildup` - Energy increasing, great for drops
- `drop` - Peak energy moment
- `break` - Low energy, good for transitions

**Mix Suitability:**
- `excellent` - Perfect for mixing, stable beat
- `good` - Usable with some skill
- `fair` - Challenging but possible  
- `poor` - Avoid mixing here

## 💡 Tips

**For Best Results:**
- Use high-quality audio files (MP3 320kbps+, FLAC, WAV)
- Ensure files have proper metadata (title, artist)
- Start with 5-10 tracks to test the system
- Review the generated labels - GPT-4 is smart but not perfect!

**API Usage:**
- Each track uses ~2-4 API calls (~$0.10-0.20 per track)
- Use `--max-tracks 10` for testing
- The system includes rate limiting to avoid API errors

## 🔧 Advanced Usage

**Custom Models:**
Edit `config.json` to use different models:
```json
{
  "openai_api_key": "your-key",
  "model": "gpt-3.5-turbo"  // Cheaper but less accurate
}
```

**Batch Processing:**
The system automatically saves progress every 3 tracks, so you can stop/resume large jobs.

## ❓ Troubleshooting

**"Config file created" error:**
- Edit `scripts/config.json` and add your real API key

**"No music files found":**
- Check the path to your music folder
- Supported formats: MP3, WAV, FLAC, M4A, AAC, OGG

**API errors:**
- Check your OpenAI API key and billing
- Reduce batch size with `--max-tracks 5`

**Analysis fails:**
- Check audio file quality and format
- Some very short or very long tracks may fail

## 📈 What's Next

Use the generated labels to:
1. Train your own music similarity models
2. Build recommendation systems
3. Create automatic DJ mixing software
4. Analyze your music library patterns

The `training_data.json` file is perfectly formatted for machine learning frameworks like scikit-learn, PyTorch, and TensorFlow.

---

**Need help?** Check the logs in the console output - the system provides detailed progress information and error messages.
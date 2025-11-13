# stage2_pipeline.py
import os
import shutil
from pydub import AudioSegment
from songIdentifier import get_songs
from predictsSegments import predict_multiple_songs
from dbsexual import trackExists, getFilePath
from editgenerator import editor
from edit import edit

AUDIO_PATH = "audio_files"
TRAINING_SET="Data/genres_original/new"
PREDICTION_DIR = "predictions"

print("Stage 2: Music Processing & Prediction Pipeline")

# ------------------------------------------------------------------
# Step 1 – User prompt
# ------------------------------------------------------------------
prompt = input("Enter what kind of music you want: ").strip()
print("User prompt:", prompt)

# ------------------------------------------------------------------
# Step 2 – Run Stage 1 pipeline
# ------------------------------------------------------------------
print("Running Stage 1 to discover and download songs ...")
tracks, track_names = get_songs(prompt)
print("Tracks returned:", track_names)

# ------------------------------------------------------------------
# Step 3 – Prepare audio directory
# ------------------------------------------------------------------
if os.path.exists(AUDIO_PATH):
    print("Clearing previous audio directory ...")
    for f in os.listdir(AUDIO_PATH):
        path = os.path.join(AUDIO_PATH, f)
        if os.path.isfile(path):
            os.remove(path)
else:
    os.makedirs(AUDIO_PATH)

print("Copying selected tracks from database to:", AUDIO_PATH)
for name in track_names:
    path = getFilePath(name.lower())
    if not path:
        print(f"⚠ No file found in DB for {name}, skipping.")
        continue
    dest = os.path.join(AUDIO_PATH, os.path.basename(path))
    shutil.copyfile(path, dest)
    print("Copied:", dest)
shutil.copytree(AUDIO_PATH,TRAINING_SET,dirs_exist_ok=True)
# ------------------------------------------------------------------
# Step 4 – Convert to WAV
# ------------------------------------------------------------------
def convert_to_wav(src_dir, delete_original=True):
    print("Converting to WAV in:", src_dir)
    for f in os.listdir(src_dir):
        full = os.path.join(src_dir, f)
        if os.path.isdir(full):
            continue
        if not f.lower().endswith(".wav"):
            audio = AudioSegment.from_file(full)
            wav_path = os.path.splitext(full)[0] + ".wav"
            audio.export(wav_path, format="wav")
            if delete_original:
                os.remove(full)
            print("Converted:", os.path.basename(wav_path))
        else:
            print("Already WAV:", f)

convert_to_wav(AUDIO_PATH)

# ------------------------------------------------------------------
# Step 5 – Run predictions
# ------------------------------------------------------------------
print("Running predict_multiple_songs() ...")
predict_multiple_songs(AUDIO_PATH)
print("Predictions saved to:", PREDICTION_DIR)
print("Generating Edits")
editor()
print("Applying Edits")
edit()
print("Pipeline complete.")

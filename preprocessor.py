
import os
import json
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from matplotlib import pyplot as plt

# ========== CONFIG ==========
DATASET = "Data/genres_original"
OUTPUT_CSV = "datasets/features_dataset.csv"
TARGET_SR = 22050
SEGMENT_DURATION = 15.0  # seconds
HOP_DURATION = 15.0      # step between segments (set smaller for overlap)
N_MFCC = 13
N_MELS = 64
N_FFT = 2048
HOP_LENGTH = 512
SILENCE_TOP_DB = 25
# ============================


# ---------- Preprocess ----------
def preprocess(file_path, target_sr=TARGET_SR):
    y, sr = librosa.load(file_path, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    y_trimmed, _ = librosa.effects.trim(y, top_db=SILENCE_TOP_DB)
    if np.max(np.abs(y_trimmed)) > 0:
        y_norm = y_trimmed / np.max(np.abs(y_trimmed))
    else:
        y_norm = y_trimmed
    return y_norm, sr


# ---------- Feature Extraction ----------
def extract_features(y, sr):
    """Extracts DSP-based features for one segment."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=N_MFCC)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    contrast = np.mean(librosa.feature.spectral_contrast(S=mel, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    y_harm, y_perc = librosa.effects.hpss(y)
    harmonic_ratio = np.mean(np.abs(y_harm)) / (np.mean(np.abs(y_perc)) + 1e-6)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))

    features = {
        "rms": float(rms),
        "zcr": float(zcr),
        "tempo": float(tempo),
        "spectral_centroid": float(centroid),
        "spectral_bandwidth": float(bandwidth),
        "spectral_contrast": float(contrast),
        "spectral_rolloff": float(rolloff),
        "harmonic_to_percussive_ratio": float(harmonic_ratio),
        "chroma_mean": float(chroma)
    }

    for i in range(mfcc.shape[0]):
        features[f"mfcc_mean_{i+1}"] = float(np.mean(mfcc[i]))
        features[f"mfcc_std_{i+1}"] = float(np.std(mfcc[i]))

    for i in range(mfcc_delta.shape[0]):
        features[f"mfcc_delta_mean_{i+1}"] = float(np.mean(mfcc_delta[i]))
        features[f"mfcc_delta_std_{i+1}"] = float(np.std(mfcc_delta[i]))
        features[f"mfcc_delta2_mean_{i+1}"] = float(np.mean(mfcc_delta2[i]))
        features[f"mfcc_delta2_std_{i+1}"] = float(np.std(mfcc_delta2[i]))

    return features


# ---------- Main Processing ----------
def process_dataset():
    rows = []

    for genre in os.listdir(DATASET):
        genre_path = os.path.join(DATASET, genre)
        if not os.path.isdir(genre_path):
            continue

        print(f"\n🎵 Processing genre: {genre}")
        for file in os.listdir(genre_path):
            if not file.endswith(".wav"):
                continue

            file_path = os.path.join(genre_path, file)
            try:
                y, sr = preprocess(file_path)
            except Exception as e:
                print(f"⚠️ Failed to load {file_path}: {e}")
                continue

            duration = librosa.get_duration(y=y, sr=sr)
            print(f"→ {file} ({duration:.2f}s)")

            for start in np.arange(0, duration, HOP_DURATION):
                end = start + SEGMENT_DURATION
                if end > duration:
                    break  # skip incomplete last segment
                segment = y[int(start * sr): int(end * sr)]

                feats = extract_features(segment, sr)
                feats.update({
                    "genre": genre,
                    "file_name": file,
                    "start": round(float(start), 2),
                    "end": round(float(end), 2)
                })
                rows.append(feats)

    # Save dataset
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Feature dataset saved to: {OUTPUT_CSV}")
    print(f"Total segments extracted: {len(df)}")

    return df


# ---------- MAIN ----------
if __name__ == "__main__":
    process_dataset()

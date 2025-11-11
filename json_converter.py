import os 
import json
import numpy as np
import librosa
np.complex=complex

OUTPUT="output"
SEGMENT_WINDOW=15.0 #SEC
HOP_SIZE=1.5 #SEC
N_MELS=64
N_MFCC=13
AUDIO_DIR="audio"
def extract_features(y,sr):
    mel=librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db=librosa.power_to_db(mel) # converts to decibel units
    mfcc=librosa.feature.mfcc(S=mel_db, n_mfcc=N_MFCC)
    spectral_centroid=np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))
    dominant_freq = np.argmax(np.mean(mel, axis=1)) * (sr / (2 * mel.shape[0]))
    energy = np.mean(mel_db)
    return {
        "energy_level": float(np.clip((energy + 80) / 80, 0, 1)),  # normalize [-80, 0] → [0, 1]
        "spectral_centroid": float(spectral_centroid),
        "spectral_bandwidth": float(spectral_bandwidth),
        "zero_crossing_rate": float(zero_crossing_rate),
        "rms_energy": float(rms),
        "dominant_frequency": float(dominant_freq),
        "mfcc_mean": [float(x) for x in np.mean(mfcc, axis=1)]
    }

def segment_audio(file_path):
    window=SEGMENT_WINDOW
    hop=HOP_SIZE
    y,sr=librosa.load(file_path, sr=None,mono=True)
    total_duration=librosa.get_duration(y=y,sr=sr)
    segments=[]
    segment_id=1
    for start in np.arange(0,total_duration-window, hop):
        end=start+window
        y_seg=y[int(start*sr):int(end*sr)]
        features=extract_features(y_seg,sr)
        if features["energy_level"]>0.6:
            mood="high_energy"
        else:
            mood="low_energy"
        if features['zero_crossing_rate']>0.05:
            activity="speech"
        else:
            activity="music"
        segments.append({
            "track_id": f"segment_{segment_id:03d}",
            "start": float(start),
            "end": float(end),
            "duration": float(end - start),
            "features": features,
            "labels": {
                "mood": mood,
                "activity": activity,
            }
        })
        segment_id += 1

    result = {
        "metadata": {
            "file_name": os.path.basename(file_path),
            "sample_rate": sr,
            "total_duration": float(total_duration),
            "model_version": "music-segmenter-v1"
        },
        "segments": segments
    }

    return result
def process_all_files(audio_dir=AUDIO_DIR, output_dir=OUTPUT):
    """Processes all audio files in the directory and creates JSON dataset."""
    os.makedirs(output_dir, exist_ok=True)
    dataset_index = []

    for file_name in os.listdir(audio_dir):
        if not file_name.lower().endswith((".wav", ".mp3")):
            continue

        file_path = os.path.join(audio_dir, file_name)
        print(f"🎵 Processing: {file_name} ...")

        json_data = segment_audio(file_path)
        print(json_data)
        json_path = os.path.join(output_dir, file_name.rsplit(".", 1)[0] + ".json")

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        dataset_index.append({
            "file_name": file_name,
            "json_path": json_path,
            "num_segments": len(json_data["segments"]),
            "duration": json_data["metadata"]["total_duration"]
        })

    # Write dataset index
    index_path = os.path.join(output_dir, "dataset_index.json")
    with open(index_path, "w") as f:
        json.dump(dataset_index, f, indent=2)

    print(f"\n✅ Dataset generation complete!")
    print(f"JSON files saved to: {output_dir}")
    print(f"Index file: {index_path}")


if __name__ == "__main__":
    process_all_files()
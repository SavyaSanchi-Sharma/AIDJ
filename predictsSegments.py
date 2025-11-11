import os, json, joblib, numpy as np, pandas as pd
from scipy.ndimage import uniform_filter1d
from preprocessor import extract_features_from_file

MODELS_DIR = "models"
OUTPUT_DIR = "predictions"
SMOOTHING_WINDOW = 3  # number of segments for rolling average

def load_models():
    reg = joblib.load(f"{MODELS_DIR}/energy_regressor.joblib")
    cls = joblib.load(f"{MODELS_DIR}/energy_classifier.joblib")
    mood = joblib.load(f"{MODELS_DIR}/mood_classifier.joblib")
    return reg, cls, mood

def predict_song(audio_path, reg, cls, mood):
    segments = extract_features_from_file(audio_path)
    if not segments:
        print(f"⚠️ No valid segments for {audio_path}")
        return []

    preds = []
    for seg in segments:
        feature_dict = {k: v for k, v in seg.items() if k not in ["start", "end"]}
        X = pd.DataFrame([feature_dict], columns=reg.feature_names_in_)
        energy_level = float(reg.predict(X)[0])
        energy_class = int(cls.predict(X)[0])
        mood_pred = str(mood.predict(X)[0])
        preds.append({
            "start": seg["start"],
            "end": seg["end"],
            "energy_level": energy_level,
            "energy_class": "high" if energy_class == 1 else "low",
            "mood": mood_pred
        })

    energy_levels = np.array([p["energy_level"] for p in preds])
    if len(energy_levels) >= SMOOTHING_WINDOW:
        smoothed = uniform_filter1d(energy_levels, size=SMOOTHING_WINDOW)
        for i, p in enumerate(preds):
            p["energy_level"] = float(smoothed[i])

    return preds

def predict_multiple_songs(audio_dir_or_list):
    reg, cls, mood = load_models()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if isinstance(audio_dir_or_list, str) and os.path.isdir(audio_dir_or_list):
        audio_files = [os.path.join(audio_dir_or_list, f)
                       for f in os.listdir(audio_dir_or_list)
                       if f.lower().endswith((".wav", ".mp3", ".flac"))]
    elif isinstance(audio_dir_or_list, (list, tuple)):
        audio_files = [f for f in audio_dir_or_list if os.path.exists(f)]
    else:
        print("⚠️ Invalid input: provide a directory or list of file paths.")
        return

    all_results = {}

    for file_path in audio_files:
        print(f"🎵 Processing: {file_path}")
        preds = predict_song(file_path, reg, cls, mood)
        song_key = os.path.splitext(os.path.basename(file_path))[0]
        all_results[song_key] = preds

        indiv_out = os.path.join(OUTPUT_DIR, song_key + "_predictions.json")
        with open(indiv_out, "w") as f:
            json.dump(preds, f, indent=2)
        print(f"✅ Saved individual predictions → {indiv_out}")

    combined_out = os.path.join(OUTPUT_DIR, "combined_predictions.json")
    with open(combined_out, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n📦 Combined predictions saved to: {combined_out}")
    return all_results


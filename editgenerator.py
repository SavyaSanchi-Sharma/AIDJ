import os
import json
import glob
from dotenv import load_dotenv
import google.generativeai as genai

PREDICTIONS_DIR = "predictions"
OUTPUT_DIR = "edited_predictions"
MODEL_NAME = "gemini-2.5-flash"

def load_api_key():
    load_dotenv("project.env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("❌ GEMINI_API_KEY not found in project.env")
    genai.configure(api_key=api_key)

def call_gemini_for_editing(track_name: str, data: list):
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = f"""
You are an intelligent audio editing assistant.
You receive a JSON describing the energy and mood segments of a track.

Your task:
- Identify natural editing points for transitions, cuts, or mood changes.
- Output a *static JSON format* ready for a video or music editing script.

Return strictly in the following format:
{{
  "track": "<track_name>",
  "sections": [
    {{
      "start": <float>,
      "end": <float>,
      "energy_level": <float>,
      "mood": "<string>",
      "edit_action": "<KEEP|CUT|TRANSITION>",
      "notes": "<short reason>"
    }}
  ]
}}

Here is the track data for "{track_name}":
{json.dumps(data, indent=2)}
"""
    response = model.generate_content(prompt)
    try:
        return json.loads(response.text)
    except Exception:
        print(f"⚠️ Failed to parse Gemini output for {track_name}, saving raw text.")
        return {"track": track_name, "raw_response": response.text}

def editor():
    load_api_key()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_files = glob.glob(os.path.join(PREDICTIONS_DIR, "*_predictions.json"))

    for file_path in json_files:
        track_name = os.path.splitext(os.path.basename(file_path))[0].replace("_predictions", "")
        print(f"🎵 Processing {track_name} ...")
        with open(file_path, "r") as f:
            track_data = json.load(f)

        edited_json = call_gemini_for_editing(track_name, track_data)
        output_path = os.path.join(OUTPUT_DIR, f"{track_name}_edited.json")
        with open(output_path, "w") as f:
            json.dump(edited_json, f, indent=2)
        print(f"✅ Saved edited JSON → {output_path}")


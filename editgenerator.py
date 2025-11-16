import os
import re
import json
import glob
from dotenv import load_dotenv
import google.generativeai as genai

PREDICTIONS_DIR = "predictions"
OUTPUT_DIR = "edited_predictions"
MODEL_NAME = "gemini-2.5-flash"
FINAL_JSON_NAME = "remix_edit_plan.json"


def load_api_key():
    load_dotenv("project.env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("❌ GEMINI_API_KEY not found in project.env")
    genai.configure(api_key=api_key)


def extract_json(text: str):
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()
    match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
    if not match:
        raise ValueError("No JSON detected.")
    json_text = match.group(1)
    json_text = re.sub(r",\s*([}\]])", r"\1", json_text)
    try:
        return json.loads(json_text)
    except:
        json_text = json_text.replace("'", "\"")
        return json.loads(json_text)


def call_gemini_for_remix(all_tracks: list):
    model = genai.GenerativeModel(MODEL_NAME)

    prompt = f"""
You are an advanced **DJ mashup + audio editing assistant**.

You are given multiple tracks with energy/mood timelines.
Your job is to create a **creative mix–match remix**, combining pieces of different songs.

### Your goals:
- Mix-match songs in ANY order.
- You can return to a track multiple times (A → B → A → C).
- Choose the BEST, highest-energy, most compatible sections.
- Create a club-style mashup using CUTS and TRANSITIONS.
- Use TRANSITION where two different tracks connect.
- KEEP only the most impactful parts.

### Allowed actions:
- KEEP → include normally  
- CUT → skip section  
- TRANSITION → fade in/out when switching tracks  

### Required Output Format:
Your final answer MUST be a JSON array:

[
  {{
    "track": "<track_name>",
    "sections": [
      {{
        "start": <float>,
        "end": <float>,
        "energy_level": <float>,
        "mood": "<string>",
        "edit_action": "<KEEP|CUT|TRANSITION>",
        "notes": "<why this part was chosen>"
      }}
    ]
  }}
]

### Important:
- You are creating a **real DJ mashup**, not a simple playlist.
- Rearranging songs is allowed.
- Repeating a track later in the remix is allowed.
- Mixing small sections across different songs is encouraged.
- Prioritize strong build-ups, drops, stable grooves, and energetic parts.

### Input Songs:
Here is all the available track data:
{json.dumps(all_tracks, indent=2)}
"""

    response = model.generate_content(prompt)
    return extract_json(response.text)


def editor():
    load_api_key()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load all prediction JSONs
    all_tracks = []
    json_files = glob.glob(os.path.join(PREDICTIONS_DIR, "*_predictions.json"))

    for file_path in json_files:
        track_name = os.path.splitext(os.path.basename(file_path))[0].replace("_predictions", "")

        with open(file_path, "r") as f:
            track_data = json.load(f)

        all_tracks.append({
            "track": track_name,
            "data": track_data
        })

    print(f"🎧 Sending {len(all_tracks)} tracks to Gemini for MIX–MATCH remixing...")

    remix_json = call_gemini_for_remix(all_tracks)

    out_path = os.path.join(OUTPUT_DIR, FINAL_JSON_NAME)
    with open(out_path, "w") as f:
        json.dump(remix_json, f, indent=2)

    print(f"✨ MIX–MATCH REMIX JSON saved → {out_path}")



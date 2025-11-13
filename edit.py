#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pydub import AudioSegment

AUDIO_DIR = "audio"
EDIT_JSON_DIR = "edited_predictions"
OUTPUT_DIR = "edited_audio"
FADE_DURATION_MS = 1000  # fade-in/out duration for transitions

def apply_edits_to_track(audio_path, edit_json_path):
    print(f"🎵 Processing {os.path.basename(audio_path)} using {os.path.basename(edit_json_path)}")

    audio = AudioSegment.from_file(audio_path)
    with open(edit_json_path, "r") as f:
        edit_plan = json.load(f)

    edited_segments = []

    for section in edit_plan.get("sections", []):
        start_ms = int(section["start"] * 1000)
        end_ms = int(section["end"] * 1000)
        action = section.get("edit_action", "KEEP").upper()

        if start_ms >= len(audio):
            continue
        end_ms = min(end_ms, len(audio))

        segment = audio[start_ms:end_ms]

        if action == "CUT":
            print(f"⏭️  Skipping {start_ms/1000:.1f}s–{end_ms/1000:.1f}s ({section.get('notes', '')})")
            continue
        elif action == "TRANSITION":
            segment = segment.fade_in(FADE_DURATION_MS).fade_out(FADE_DURATION_MS)
            print(f"🔄 Transition: {start_ms/1000:.1f}s–{end_ms/1000:.1f}s")
        else:
            print(f"🎬 Keeping: {start_ms/1000:.1f}s–{end_ms/1000:.1f}s")

        edited_segments.append(segment)

    if not edited_segments:
        print("⚠️ No segments kept — skipping export.")
        return

    final_track = sum(edited_segments)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    track_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{track_name}_edited.wav")
    final_track.export(output_path, format="wav")

    print(f"✅ Saved edited track: {output_path}")

def edit():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_files = [f for f in os.listdir(EDIT_JSON_DIR) if f.endswith("_edited.json")]

    for json_file in json_files:
        track_name = json_file.replace("_edited.json", "")
        audio_path = os.path.join(AUDIO_DIR, f"{track_name}.wav")
        edit_json_path = os.path.join(EDIT_JSON_DIR, json_file)

        if not os.path.exists(audio_path):
            print(f"⚠️ Audio not found for {track_name}, skipping.")
            continue

        apply_edits_to_track(audio_path, edit_json_path)

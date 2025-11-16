#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pydub import AudioSegment

AUDIO_DIR = "audio_files"
EDIT_JSON_DIR = "edited_predictions"
OUTPUT_DIR = "edited_audio"

FADE_DURATION_MS = 1000  # fade-in/out for TRANSITION
FINAL_OUTPUT_NAME = "final_remix.wav"


def safe_filename(track_name: str):
    """
    Convert the LLM track name directly to a .wav filename.
    Your audio files must match this convention.
    """
    return track_name.strip() + ".wav"


def apply_edits(audio_path, sections):
    """
    Apply KEEP / CUT / TRANSITION edits to a single track.
    Returns an AudioSegment containing the edited audio.
    """

    audio = AudioSegment.from_file(audio_path)
    edited_segments = []

    for section in sections:
        start_ms = int(section["start"] * 1000)
        end_ms = int(section["end"] * 1000)
        action = section.get("edit_action", "KEEP").upper()
        notes = section.get("notes", "")

        if start_ms >= len(audio):
            continue

        end_ms = min(end_ms, len(audio))
        segment = audio[start_ms:end_ms]

        if action == "CUT":
            print(f"⏭️  CUT {start_ms/1000:.1f}s–{end_ms/1000:.1f}s ({notes})")
            continue

        elif action == "TRANSITION":
            print(f"🔄 TRANSITION {start_ms/1000:.1f}s–{end_ms/1000:.1f}s")
            segment = segment.fade_in(FADE_DURATION_MS).fade_out(FADE_DURATION_MS)

        else:
            print(f"🎬 KEEP {start_ms/1000:.1f}s–{end_ms/1000:.1f}s ({notes})")

        edited_segments.append(segment)

    if not edited_segments:
        print("⚠️ No segments kept for this track.")
        return None

    return sum(edited_segments)


def create_remix(json_path):
    """
    Creates ONE combined remix from all tracks inside the JSON file.
    """

    print(f"\n📄 Loading multi-track edit file: {json_path}")

    with open(json_path, "r") as f:
        track_list = json.load(f)

    final_mix = AudioSegment.silent(duration=0)

    for entry in track_list:
        track_name = entry["track"]
        sections = entry["sections"]

        filename = safe_filename(track_name)
        audio_path = os.path.join(AUDIO_DIR, filename)

        if not os.path.exists(audio_path):
            print(f"⚠️ Missing audio file for: {track_name} ({filename})")
            continue

        print(f"\n🎧 Editing track: {track_name}")
        edited_track = apply_edits(audio_path, sections)

        if edited_track:
            final_mix += edited_track  # append to remix

    if len(final_mix) == 0:
        print("❌ No audio could be mixed. Nothing exported.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, FINAL_OUTPUT_NAME)
    final_mix.export(out_path, format="wav")

    print(f"\n🎉 ALL DONE!")
    print(f"🔥 Remix saved → {out_path}")


def edit():
    """
    Runs the editor on ALL multi-track JSONs in EDIT_JSON_DIR,
    producing one remix per JSON.
    """
    for filename in os.listdir(EDIT_JSON_DIR):
        if filename.endswith(".json"):
            create_remix(os.path.join(EDIT_JSON_DIR, filename))


if __name__ == "__main__":
    edit()

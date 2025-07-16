<img align="left" width="308" height="308" alt="iconW" src="https://github.com/user-attachments/assets/57e6dc0f-90dc-46bb-87db-26d226cf3cc2" />

# ANISEME
Animate Visemes and Lip Sync with driving clips from audio.

## Overview

**ANISEME** is a tool for animators and technical artists that automatically analyzes spoken audio and generates viseme tracks for lip sync animation. It uses OpenAI Whisper for speech-to-text transcription and phoneme timing, then maps phonemes to visemes and splits the original audio into separate WAV files for each viseme.

## How It Works

1. **Transcribe Audio:** ANISEME uses Whisper to transcribe a WAV file and extract word-level timestamps.
2. **Phoneme Extraction:** Each word is broken down into phonemes using the CMU Pronouncing Dictionary.
3. **Viseme Mapping:** Phonemes are mapped to visemes (mouth shapes) using a customizable mapping.
4. **Track Generation:** For each viseme, a time track is created indicating when that viseme is active.
5. **Audio Splitting:** The original audio is split into separate WAV files for each viseme, with soft fades for smooth transitions.
6. **Export:** Viseme tracks are saved as `.npz` (NumPy) and `.xml` files, and individual viseme audio clips are exported to the `output` folder.

## Use Case for Animators

- **Lip Sync Animation:** Import the viseme tracks and split audio clips into Blender or any animation software that supports audio as a driver.
- **Audio-Driven Animation:** Use the individual viseme WAV files to drive mouth shapes or blendshapes, enabling precise, automatic lip sync.
- **Customization:** Adjust viseme mappings or timing to fit your character rig.

## Getting Started

1. Install dependencies from `requirements.txt`.
2. Run `main.py` and select a WAV file.
3. The tool will generate viseme tracks and split audio in the `output` folder.

## Output Files

- `output/viseme_tracks.npz`: NumPy arrays for each viseme's activation over time.
- `output/viseme_tracks.xml`: XML file describing viseme timing for import into animation tools.
- `output/*.wav`: Individual WAV files for each viseme, usable as audio drivers.

## Example Workflow

1. Record or obtain a clean WAV file of dialogue.
2. Use ANISEME to process the audio.
3. Import the viseme XML and audio clips into Blender.
4. Use the audio clips to drive mouth shapes via Blender's audio-to-F-curve feature or custom drivers.

## Why?

I've been trying to find a solution like this for 5 years at this point.
I didn't find one.

<img align="center" width="300" height="300" alt="iconW" src="https://raw.githubusercontent.com/F1dg3tXD/ANISEME/refs/heads/main/res/iconW.png" />

# ANISEME
Animate Visemes and Lip Sync with driving clips from audio.

## Overview

**ANISEME** is a tool for animators and technical artists that automatically analyzes speech or sung vocals and generates viseme tracks for lip sync animation. It uses OpenAI Whisper for transcription and word timing, then maps phonemes to visemes and splits the original audio into separate WAV files for each viseme.

## How It Works

1. **Transcribe Audio:** ANISEME uses Whisper to transcribe an audio file and extract word-level timestamps.
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

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2. Install system `ffmpeg`, which Whisper requires:

```bash
brew install ffmpeg
```

3. Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

4. Run the app:

```bash
python main.py
```

5. Select an audio file, choose a Whisper profile, and run the analysis. The tool will generate viseme tracks and split audio in the `output` folder.

## Model Profiles

- **Speech (Fast, Recommended):** Uses Whisper `turbo` for quick speech transcription.
- **Speech (Best Accuracy):** Uses Whisper `large-v3` for the strongest official accuracy.
- **Lyrics / Singing:** Uses `large-v3` and works best when you paste the lyrics into the hint box before running.
- **Custom Whisper Checkpoint:** Lets you point ANISEME at a local Whisper-compatible `.pt` file if you have a song-specific fine-tuned model.

Whisper models are downloaded automatically on first use and then reused from the local cache.

## Panels

- **Jobs:** Tracks model downloads and transcription work in the background, including stage and progress.
- **Settings:** Lets you change the model cache directory and register extra local Whisper checkpoints so they appear in the model picker.

## Install Notes

- `requirements.txt` is intentionally minimal. The old dependency list included unrelated plotting/alignment packages such as `matplotlib`, which triggered failing source builds on Python 3.14.
- ANISEME now enables the system trust store at app startup via `truststore`, which helps model downloads succeed on macOS setups where Python's certificate bundle is incomplete.
- Apple Silicon is supported through PyTorch's `mps` backend, so the app now checks `CUDA`, then `MPS`, then falls back to CPU.

## Output Files

- `output/viseme_tracks.npz`: NumPy arrays for each viseme's activation over time.
- `output/viseme_tracks.xml`: XML file describing viseme timing for import into animation tools.
- `output/*.wav`: Individual WAV files for each viseme, usable as audio drivers.

## Example Workflow

1. Record or obtain a clean dialogue clip or song vocal.
2. Use ANISEME to process the audio.
3. Import the viseme XML and audio clips into Blender.
4. Use the audio clips to drive mouth shapes via Blender's audio-to-F-curve feature or custom drivers.

## Why?

I've been trying to find a solution like this for 5 years at this point.
I didn't find one.

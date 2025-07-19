import whisper
import pronouncing
import numpy as np
import torch
import sys
import os
from viseme_map import PHONEME_TO_VISEME
from PyQt6 import QtWidgets, QtCore, QtGui
import re
import xml.etree.ElementTree as ET
from pydub import AudioSegment

class WhisperVisemeApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ANISEME")
        self.setGeometry(100, 100, 600, 500)
        self.setWindowIcon(QtGui.QIcon("res/iconw.png"))

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QtWidgets.QVBoxLayout(self.central_widget)

        self.file_button = QtWidgets.QPushButton("Select WAV File")
        self.file_button.clicked.connect(self.select_audio)
        layout.addWidget(self.file_button)

        self.transcript_label = QtWidgets.QLabel("Optional Transcript:")
        layout.addWidget(self.transcript_label)
        self.transcript_input = QtWidgets.QTextEdit()
        self.transcript_input.setPlaceholderText("Paste transcript here to improve accuracy...")
        layout.addWidget(self.transcript_input)

        self.console = QtWidgets.QTextEdit()
        self.console.setReadOnly(True)
        layout.addWidget(self.console)

        self.process_button = QtWidgets.QPushButton("Run Whisper Viseme Analysis")
        self.process_button.clicked.connect(self.run_whisper)
        layout.addWidget(self.process_button)

        self.audio_path = ""

    def log(self, msg):
        self.console.append(msg)
        QtWidgets.QApplication.processEvents()

    def select_audio(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select WAV File", "", "WAV files (*.wav)")
        if path:
            self.audio_path = path
            self.log(f"Selected: {path}")

    def run_whisper(self):
        if not self.audio_path:
            self.log("No audio file selected.")
            return

        self.log("Loading Whisper model...")
        model = whisper.load_model("large-v3", device="cuda" if torch.cuda.is_available() else "cpu")

        transcript = self.transcript_input.toPlainText().strip()
        if transcript:
            self.log("Transcribing with provided transcript hint...")
            result = model.transcribe(self.audio_path, word_timestamps=True, language='en', initial_prompt=transcript)
        else:
            self.log("Transcribing without transcript hint...")
            result = model.transcribe(self.audio_path, word_timestamps=True, language='en')

        words = []
        for seg in result["segments"]:
            for w in seg.get("words", []):
                words.append((w["word"], w["start"], w["end"]))

        self.log(f"Found {len(words)} words.")
        self.extract_visemes(words)

    def save_viseme_xml(self, viseme_tracks, fps=100):
        root = ET.Element("visemes", framerate=str(fps))

        for viseme, track in viseme_tracks.items():
            viseme_elem = ET.SubElement(root, "track", name=viseme)
            active = False
            start_frame = 0
            for i, value in enumerate(track):
                if value > 0 and not active:
                    start_frame = i
                    active = True
                elif value == 0 and active:
                    end_frame = i
                    active = False
                    ET.SubElement(viseme_elem, "spike", start=f"{start_frame / fps:.2f}", end=f"{end_frame / fps:.2f}")

            if active:
                ET.SubElement(viseme_elem, "spike", start=f"{start_frame / fps:.2f}", end=f"{len(track) / fps:.2f}")

        tree = ET.ElementTree(root)
        tree.write("output/viseme_tracks.xml", encoding="utf-8", xml_declaration=True)

    def extract_visemes(self, words):
        duration = max(end for _, _, end in words)
        viseme_tracks = {v: np.zeros(int(duration * 100), dtype=np.float32) for v in PHONEME_TO_VISEME.values()}
        viseme_tracks["sil"] = np.zeros(int(duration * 100), dtype=np.float32)

        VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}

        for word, start, end in words:
            phonemes = self.get_phonemes(word)
            if not phonemes:
                self.log(f"No phonemes found for '{word}', defaulting to 'sil'.")
                phonemes = ['sil']

            # Assign higher weight to vowels
            weights = []
            for phon in phonemes:
                phon_root = re.sub(r"\d", "", phon.upper())
                weights.append(2.0 if phon_root in VOWELS else 1.0)

            total_weight = sum(weights)
            if total_weight == 0:
                continue

            word_duration = end - start
            current_time = start

            # Assign a minimum duration per phoneme to avoid losing any
            min_dur = 0.001  # 1ms minimum for very short phonemes
            total_assigned = 0.0
            durations = []

            for weight in weights:
                ideal_duration = word_duration * (weight / total_weight)
                dur = max(min_dur, ideal_duration)
                durations.append(dur)
                total_assigned += dur

            # Adjust durations if total goes over word duration
            if total_assigned > word_duration:
                scale = word_duration / total_assigned
                durations = [d * scale for d in durations]

            for phon, dur in zip(phonemes, durations):
                t_start = int(current_time * 100)
                t_end = int((current_time + dur) * 100)
                current_time += dur

                viseme = PHONEME_TO_VISEME.get(phon.upper(), "sil")
                viseme_tracks[viseme][t_start:t_end] = 1.0

        os.makedirs("output", exist_ok=True)
        np.savez("output/viseme_tracks.npz", **viseme_tracks)
        self.log("Saved viseme tracks to output/viseme_tracks.npz")

        self.save_viseme_xml(viseme_tracks)
        self.log("Saved viseme XML to output/viseme_tracks.xml")

        self.log("Splitting audio by viseme...")
        self.split_audio_by_viseme(viseme_tracks, duration)
        self.log("Saved individual viseme WAV files.")

    def split_audio_by_viseme(self, viseme_tracks, duration):
        full_audio = AudioSegment.from_wav(self.audio_path)
        total_ms = int(duration * 1000)
        fade_ms = 20

        for viseme, track in viseme_tracks.items():
            output_audio = AudioSegment.silent(duration=total_ms)
            active = False
            t_start = 0
            for i, v in enumerate(track):
                if v > 0 and not active:
                    t_start = i
                    active = True
                elif v == 0 and active:
                    t_end = i
                    active = False
                    ms_start = int(t_start * 10)
                    ms_end = int(t_end * 10)

                    spike = full_audio[ms_start:ms_end]
                    if len(spike) >= 2 * fade_ms:
                        spike = spike.fade_in(fade_ms).fade_out(fade_ms)

                    output_audio = output_audio.overlay(spike, position=ms_start)

            if active:
                ms_start = int(t_start * 10)
                ms_end = int(len(track) * 10)
                spike = full_audio[ms_start:ms_end]
                if len(spike) >= 2 * fade_ms:
                    spike = spike.fade_in(fade_ms).fade_out(fade_ms)

                output_audio = output_audio.overlay(spike, position=ms_start)

            output_path = os.path.join("output", f"{viseme}.wav")
            output_audio.export(output_path, format="wav")

    def get_phonemes(self, word):
        word_clean = re.sub(r"[^a-zA-Z]", "", word).lower()
        phones = pronouncing.phones_for_word(word_clean)
        if phones:
            return phones[0].split()

        if "'" in word.lower():
            parts = word.lower().replace("'", " ").split()
            phonemes = []
            for part in parts:
                sub_phones = pronouncing.phones_for_word(part)
                if sub_phones:
                    phonemes.extend(sub_phones[0].split())
            if phonemes:
                return phonemes

        return []

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    win = WhisperVisemeApp()
    win.show()
    app.exec()

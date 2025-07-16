import whisper
import pronouncing
import numpy as np
import os
from viseme_map import PHONEME_TO_VISEME
from PyQt6 import QtWidgets, QtCore, QtGui  # <-- Add QtGui import
import re
import xml.etree.ElementTree as ET
from pydub import AudioSegment

class WhisperVisemeApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ANISEME")
        self.setGeometry(100, 100, 600, 400)
        self.setWindowIcon(QtGui.QIcon("res/iconw.png"))  # <-- Set window icon

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QtWidgets.QVBoxLayout(self.central_widget)

        self.file_button = QtWidgets.QPushButton("Select WAV File")
        self.file_button.clicked.connect(self.select_audio)
        layout.addWidget(self.file_button)

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
        model = whisper.load_model("base")
        self.log("Transcribing...")

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

            # Handle trailing spike
            if active:
                ET.SubElement(viseme_elem, "spike", start=f"{start_frame / fps:.2f}", end=f"{len(track) / fps:.2f}")

        tree = ET.ElementTree(root)
        tree.write("output/viseme_tracks.xml", encoding="utf-8", xml_declaration=True)

    def extract_visemes(self, words):
        duration = max(end for _, _, end in words)
        viseme_tracks = {v: np.zeros(int(duration * 100), dtype=np.float32) for v in PHONEME_TO_VISEME.values()}
        viseme_tracks["sil"] = np.zeros(int(duration * 100), dtype=np.float32)

        for word, start, end in words:
            phonemes = self.get_phonemes(word)
            if not phonemes:
                self.log(f"No phonemes found for '{word}', defaulting to 'sil'.")
                phonemes = ['sil']

            phoneme_duration = (end - start) / len(phonemes)
            for i, phon in enumerate(phonemes):
                t_start = int((start + i * phoneme_duration) * 100)
                t_end = int((start + (i + 1) * phoneme_duration) * 100)
                viseme = PHONEME_TO_VISEME.get(phon.upper(), "sil")
                viseme_tracks[viseme][t_start:t_end] = 1.0

        os.makedirs("output", exist_ok=True)
        np.savez("output/viseme_tracks.npz", **viseme_tracks)
        self.log("Saved viseme tracks to output/viseme_tracks.npz")

        self.save_viseme_xml(viseme_tracks)
        self.log("Saved viseme XML to output/viseme_tracks.xml")

        # Audio splitting
        self.log("Splitting audio by viseme...")
        self.split_audio_by_viseme(viseme_tracks, duration)
        self.log("Saved individual viseme WAV files.")

    def split_audio_by_viseme(self, viseme_tracks, duration):
    
        full_audio = AudioSegment.from_wav(self.audio_path)
        total_ms = int(duration * 1000)
        fade_ms = 20  # 20ms soft fade

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

            # Trailing viseme region
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
        # Normalize the word
        word_clean = re.sub(r"[^a-zA-Z]", "", word).lower()

        # Use pronouncing to find phonemes
        phones = pronouncing.phones_for_word(word_clean)
        if phones:
            return phones[0].split()

        # Optional fallback: try to split contractions (like "that's" -> "that", "is")
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

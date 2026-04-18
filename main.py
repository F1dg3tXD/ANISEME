import hashlib
import io
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import cmudict
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from pydub import AudioSegment
import torch
import whisper

from viseme_map import PHONEME_TO_VISEME

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR / "output"
CMU_DICT = cmudict.dict()
VOWELS = {
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "EH",
    "ER",
    "EY",
    "IH",
    "IY",
    "OW",
    "OY",
    "UH",
    "UW",
}

MODEL_PRESETS = [
    {
        "kind": "official",
        "label": "Speech (Fast, Recommended)",
        "model": "turbo",
        "notes": "Fastest official OpenAI Whisper preset for everyday English speech.",
    },
    {
        "kind": "official",
        "label": "Speech (Best Accuracy)",
        "model": "large-v3",
        "notes": "Highest-accuracy official Whisper preset, but much larger and slower.",
    },
    {
        "kind": "official",
        "label": "Lyrics / Singing",
        "model": "large-v3",
        "notes": "Best default for songs. Paste lyrics below to improve recognition on vocals.",
    },
]

ONE_OFF_MODEL_ENTRY = {
    "kind": "ad_hoc",
    "label": "One-Off Local Checkpoint",
    "model": "custom",
    "notes": "Use a local Whisper-compatible checkpoint once without adding it to Settings.",
}


def enable_system_trust_store():
    try:
        import truststore

        truststore.inject_into_ssl()
        return True
    except Exception:
        return False


def detect_best_device():
    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and mps_backend.is_available():
        return "mps"

    return "cpu"


def ensure_directory(path: Path, fallback: Path | None = None):
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError:
        if fallback is None:
            raise
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def default_cache_dir():
    cache_root = QtCore.QStandardPaths.writableLocation(
        QtCore.QStandardPaths.StandardLocation.CacheLocation
    )
    if cache_root:
        candidate = Path(cache_root) / "whisper"
    else:
        candidate = Path.home() / ".cache" / "ANISEME" / "whisper"

    return ensure_directory(candidate, BASE_DIR / ".cache" / "whisper")


def settings_file_path():
    app_data_root = QtCore.QStandardPaths.writableLocation(
        QtCore.QStandardPaths.StandardLocation.AppDataLocation
    )
    if app_data_root:
        settings_root = Path(app_data_root)
    else:
        settings_root = BASE_DIR / ".appdata"

    settings_root = ensure_directory(settings_root, BASE_DIR / ".appdata")
    return settings_root / "settings.json"


def normalize_settings(data):
    if not isinstance(data, dict):
        data = {}

    cache_dir = data.get("cache_dir", "")
    if not isinstance(cache_dir, str):
        cache_dir = ""

    additional_models = []
    for entry in data.get("additional_models", []):
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        path = str(entry.get("path", "")).strip()
        if name and path:
            additional_models.append({"name": name, "path": path})

    return {"cache_dir": cache_dir, "additional_models": additional_models}


def load_settings(path: Path):
    if not path.exists():
        return normalize_settings({})

    try:
        return normalize_settings(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return normalize_settings({})


def save_settings(path: Path, data):
    normalized = normalize_settings(data)
    path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")


def resolve_cache_dir(configured_path: str):
    if configured_path:
        preferred = Path(configured_path).expanduser()
        return ensure_directory(preferred, BASE_DIR / ".cache" / "whisper")
    return default_cache_dir()


def lookup_phonemes(word):
    word_clean = re.sub(r"[^a-zA-Z]", "", word).lower()
    phones = CMU_DICT.get(word_clean)
    if phones:
        return phones[0]

    if "'" in word.lower():
        parts = word.lower().replace("'", " ").split()
        phonemes = []
        for part in parts:
            sub_phones = CMU_DICT.get(part)
            if sub_phones:
                phonemes.extend(sub_phones[0])
        if phonemes:
            return phonemes

    return []


def build_output_dir(audio_path: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(audio_path).stem or "audio"
    output_dir = OUTPUT_ROOT / f"{stem}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_viseme_xml(viseme_tracks, output_dir: Path, fps=100):
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
                active = False
                ET.SubElement(
                    viseme_elem,
                    "spike",
                    start=f"{start_frame / fps:.2f}",
                    end=f"{i / fps:.2f}",
                )

        if active:
            ET.SubElement(
                viseme_elem,
                "spike",
                start=f"{start_frame / fps:.2f}",
                end=f"{len(track) / fps:.2f}",
            )

    ET.ElementTree(root).write(
        output_dir / "viseme_tracks.xml",
        encoding="utf-8",
        xml_declaration=True,
    )


def split_audio_by_viseme(
    viseme_tracks,
    duration,
    audio_path,
    output_dir: Path,
    progress_callback=None,
):
    full_audio = AudioSegment.from_file(audio_path)
    total_ms = int(duration * 1000)
    fade_ms = 20
    visemes = list(viseme_tracks.items())
    total_visemes = max(len(visemes), 1)

    for index, (viseme, track) in enumerate(visemes, start=1):
        output_audio = AudioSegment.silent(duration=total_ms)
        active = False
        t_start = 0

        for i, value in enumerate(track):
            if value > 0 and not active:
                t_start = i
                active = True
            elif value == 0 and active:
                active = False
                ms_start = int(t_start * 10)
                ms_end = int(i * 10)
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

        output_audio.export(output_dir / f"{viseme}.wav", format="wav")

        if progress_callback:
            progress_callback(index / total_visemes)


def create_viseme_artifacts(words, audio_path, output_dir, stage_callback=None, progress_callback=None):
    if not words:
        raise ValueError("No words were available for viseme extraction.")

    duration = max(end for _, _, end in words)
    frame_count = max(int(duration * 100), 1)
    viseme_tracks = {v: np.zeros(frame_count, dtype=np.float32) for v in PHONEME_TO_VISEME.values()}
    viseme_tracks["sil"] = np.zeros(frame_count, dtype=np.float32)

    if stage_callback:
        stage_callback("Generating viseme tracks")

    total_words = max(len(words), 1)
    for index, (word, start, end) in enumerate(words, start=1):
        phonemes = lookup_phonemes(word)
        if not phonemes:
            phonemes = ["sil"]

        weights = []
        for phon in phonemes:
            phon_root = re.sub(r"\d", "", phon.upper())
            weights.append(2.0 if phon_root in VOWELS else 1.0)

        total_weight = sum(weights)
        if total_weight == 0:
            continue

        word_duration = end - start
        current_time = start
        min_duration = 0.001
        total_assigned = 0.0
        durations = []

        for weight in weights:
            ideal_duration = word_duration * (weight / total_weight)
            duration_slice = max(min_duration, ideal_duration)
            durations.append(duration_slice)
            total_assigned += duration_slice

        if total_assigned > word_duration and word_duration > 0:
            scale = word_duration / total_assigned
            durations = [duration_slice * scale for duration_slice in durations]

        for phon, duration_slice in zip(phonemes, durations):
            t_start = int(current_time * 100)
            t_end = int((current_time + duration_slice) * 100)
            current_time += duration_slice

            viseme = PHONEME_TO_VISEME.get(phon.upper(), "sil")
            viseme_tracks[viseme][t_start:t_end] = 1.0

        if progress_callback and index % 5 == 0:
            progress_callback(index / total_words * 0.5)

    np.savez(output_dir / "viseme_tracks.npz", **viseme_tracks)
    save_viseme_xml(viseme_tracks, output_dir)

    if stage_callback:
        stage_callback("Exporting per-viseme audio")

    split_audio_by_viseme(
        viseme_tracks,
        duration,
        audio_path,
        output_dir,
        progress_callback=lambda ratio: progress_callback(0.5 + ratio * 0.5) if progress_callback else None,
    )


class SignalProgressTqdm:
    def __init__(self, start_progress, end_progress, progress_handler, total=None, disable=False, **kwargs):
        self.start_progress = start_progress
        self.end_progress = end_progress
        self.progress_handler = progress_handler
        self.total = total or 0
        self.disable = disable
        self.n = 0

    def __enter__(self):
        if not self.disable:
            self.progress_handler(self.start_progress)
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.disable:
            self.progress_handler(self.end_progress)

    def update(self, amount):
        self.n += amount
        if self.disable:
            return
        if self.total:
            ratio = min(max(self.n / self.total, 0.0), 1.0)
            progress = self.start_progress + int(
                (self.end_progress - self.start_progress) * ratio
            )
            self.progress_handler(progress)


def download_official_model(model_name, download_root, stage_handler, progress_handler):
    url = whisper._MODELS[model_name]
    expected_sha256 = url.split("/")[-2]
    target = Path(download_root) / Path(url).name
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not target.is_file():
        raise RuntimeError(f"{target} exists and is not a regular file")

    if target.is_file():
        existing_bytes = target.read_bytes()
        if hashlib.sha256(existing_bytes).hexdigest() == expected_sha256:
            stage_handler("Using cached model")
            progress_handler(30)
            return target

    stage_handler("Downloading model")
    downloaded = 0
    hasher = hashlib.sha256()
    with urllib.request.urlopen(url) as source, target.open("wb") as output:
        total_size = int(source.info().get("Content-Length") or 0)
        while True:
            buffer = source.read(8192)
            if not buffer:
                break
            output.write(buffer)
            hasher.update(buffer)
            downloaded += len(buffer)
            if total_size:
                progress_handler(int(downloaded / total_size * 30))

    if hasher.hexdigest() != expected_sha256:
        raise RuntimeError("Downloaded model checksum did not match. Please retry.")

    progress_handler(30)
    return target


def load_whisper_model_with_progress(model_name, device, download_root, stage_handler, progress_handler):
    checkpoint_path = None
    alignment_heads = None

    if model_name in whisper._MODELS:
        checkpoint_path = download_official_model(
            model_name,
            download_root,
            stage_handler,
            progress_handler,
        )
        alignment_heads = whisper._ALIGNMENT_HEADS[model_name]
    else:
        checkpoint_path = Path(model_name)
        if not checkpoint_path.is_file():
            raise RuntimeError(f"Model not found: {checkpoint_path}")
        stage_handler("Loading local checkpoint")
        progress_handler(25)

    stage_handler("Loading model weights")
    progress_handler(35)
    with checkpoint_path.open("rb") as fp:
        kwargs = {"weights_only": True} if torch.__version__ >= "1.13" else {}
        checkpoint = torch.load(fp, map_location=device, **kwargs)

    stage_handler("Preparing model")
    progress_handler(40)
    dims = whisper.ModelDimensions(**checkpoint["dims"])
    model = whisper.Whisper(dims)
    model.load_state_dict(checkpoint["model_state_dict"])

    if alignment_heads is not None:
        model.set_alignment_heads(alignment_heads)

    return model.to(device)


def transcribe_with_progress(model, audio_path, options, progress_handler, stage_handler):
    stage_handler("Transcribing audio")
    progress_handler(40)
    tqdm_module = whisper.transcribe.__globals__["tqdm"]
    original_tqdm = tqdm_module.tqdm

    def tqdm_factory(*args, **kwargs):
        return SignalProgressTqdm(
            start_progress=40,
            end_progress=80,
            progress_handler=progress_handler,
            **kwargs,
        )

    tqdm_module.tqdm = tqdm_factory
    try:
        return model.transcribe(audio_path, verbose=False, **options)
    finally:
        tqdm_module.tqdm = original_tqdm


class WhisperJobWorker(QtCore.QObject):
    progress_changed = QtCore.pyqtSignal(str, int)
    stage_changed = QtCore.pyqtSignal(str, str)
    status_changed = QtCore.pyqtSignal(str, str)
    log_message = QtCore.pyqtSignal(str, str)
    completed = QtCore.pyqtSignal(str, object)
    failed = QtCore.pyqtSignal(str, str)
    finished = QtCore.pyqtSignal(str)

    def __init__(
        self,
        job_id,
        mode,
        model_entry,
        cache_dir,
        device,
        audio_path="",
        transcript="",
    ):
        super().__init__()
        self.job_id = job_id
        self.mode = mode
        self.model_entry = dict(model_entry)
        self.cache_dir = Path(cache_dir)
        self.device = device
        self.audio_path = audio_path
        self.transcript = transcript

    def emit_progress(self, value):
        self.progress_changed.emit(self.job_id, int(max(0, min(value, 100))))

    def emit_stage(self, text):
        self.stage_changed.emit(self.job_id, text)
        self.log_message.emit(self.job_id, text)

    @QtCore.pyqtSlot()
    def run(self):
        enable_system_trust_store()
        self.status_changed.emit(self.job_id, "Running")
        try:
            if self.mode == "download":
                result = self.run_download_job()
            else:
                result = self.run_transcription_job()

            self.emit_progress(100)
            self.status_changed.emit(self.job_id, "Completed")
            self.stage_changed.emit(self.job_id, "Completed")
            self.completed.emit(self.job_id, result)
        except Exception as exc:
            self.status_changed.emit(self.job_id, "Failed")
            self.failed.emit(self.job_id, str(exc))
        finally:
            self.finished.emit(self.job_id)

    def run_download_job(self):
        model_name = self.model_entry["model"]
        model_kind = self.model_entry["kind"]
        if model_kind == "official":
            target = download_official_model(
                model_name,
                self.cache_dir,
                self.emit_stage,
                self.emit_progress,
            )
            return {"message": f"Model ready: {target}", "path": str(target)}

        checkpoint = Path(model_name)
        if not checkpoint.is_file():
            raise RuntimeError(f"Model not found: {checkpoint}")
        self.emit_stage("Validated local checkpoint")
        self.emit_progress(100)
        return {"message": f"Local checkpoint available: {checkpoint}", "path": str(checkpoint)}

    def run_transcription_job(self):
        model_name = self.model_entry["model"]
        model = load_whisper_model_with_progress(
            model_name,
            self.device,
            self.cache_dir,
            self.emit_stage,
            self.emit_progress,
        )

        transcribe_options = {
            "word_timestamps": True,
            "language": "en",
            "fp16": self.device == "cuda",
        }
        if self.transcript:
            transcribe_options["initial_prompt"] = self.transcript
            self.log_message.emit(self.job_id, "Using transcript / lyric hint.")

        result = transcribe_with_progress(
            model,
            self.audio_path,
            transcribe_options,
            self.emit_progress,
            self.emit_stage,
        )

        words = []
        for segment in result.get("segments", []):
            for word in segment.get("words", []):
                word_text = word.get("word", "").strip()
                if word_text:
                    words.append((word_text, word["start"], word["end"]))

        if not words:
            raise RuntimeError(
                "Whisper returned no word timestamps. Try a larger model or paste transcript / lyrics hints."
            )

        output_dir = build_output_dir(self.audio_path)
        self.emit_stage("Creating viseme exports")
        create_viseme_artifacts(
            words,
            self.audio_path,
            output_dir,
            stage_callback=self.emit_stage,
            progress_callback=lambda ratio: self.emit_progress(80 + int(ratio * 19)),
        )

        return {
            "message": f"Saved viseme exports to {output_dir}",
            "output_dir": str(output_dir),
            "word_count": len(words),
        }


class WhisperVisemeApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ANISEME")
        self.setGeometry(100, 100, 1080, 680)
        self.setWindowIcon(QtGui.QIcon(str(BASE_DIR / "res" / "iconW.png")))

        self.ssl_patched = enable_system_trust_store()
        self.device = detect_best_device()
        self.audio_path = ""
        self.job_counter = 0
        self.job_rows = {}
        self.job_threads = {}
        self.active_transcription_jobs = set()

        self.settings_path = settings_file_path()
        self.settings = load_settings(self.settings_path)
        self.cache_dir = resolve_cache_dir(self.settings.get("cache_dir", ""))

        self.build_main_ui()
        self.build_jobs_panel()
        self.build_settings_panel()
        self.refresh_model_combo()
        self.populate_settings_widgets()

        self.log(f"Whisper cache: {self.cache_dir}")
        self.log(f"Detected device: {self.device}")
        if self.ssl_patched:
            self.log("System trust store enabled for HTTPS downloads.")

    def build_main_ui(self):
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QtWidgets.QVBoxLayout(self.central_widget)

        self.file_button = QtWidgets.QPushButton("Select Audio File")
        self.file_button.clicked.connect(self.select_audio)
        layout.addWidget(self.file_button)

        self.file_label = QtWidgets.QLabel("No audio file selected yet.")
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)

        self.model_label = QtWidgets.QLabel("Whisper Model Profile:")
        layout.addWidget(self.model_label)

        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        layout.addWidget(self.model_combo)

        self.model_notes = QtWidgets.QLabel()
        self.model_notes.setWordWrap(True)
        layout.addWidget(self.model_notes)

        custom_model_layout = QtWidgets.QHBoxLayout()
        self.custom_model_input = QtWidgets.QLineEdit()
        self.custom_model_input.setPlaceholderText("Optional: select a local Whisper checkpoint (.pt)")
        custom_model_layout.addWidget(self.custom_model_input)

        self.custom_model_button = QtWidgets.QPushButton("Browse Model")
        self.custom_model_button.clicked.connect(self.select_custom_model)
        custom_model_layout.addWidget(self.custom_model_button)

        self.custom_model_widget = QtWidgets.QWidget()
        self.custom_model_widget.setLayout(custom_model_layout)
        layout.addWidget(self.custom_model_widget)

        self.runtime_notes = QtWidgets.QLabel()
        self.runtime_notes.setWordWrap(True)
        layout.addWidget(self.runtime_notes)

        self.transcript_label = QtWidgets.QLabel("Optional Transcript / Lyrics Hint:")
        layout.addWidget(self.transcript_label)

        self.transcript_input = QtWidgets.QTextEdit()
        self.transcript_input.setPlaceholderText(
            "Paste a clean transcript or song lyrics here to improve recognition..."
        )
        layout.addWidget(self.transcript_input)

        button_row = QtWidgets.QHBoxLayout()
        self.download_button = QtWidgets.QPushButton("Download Selected Model")
        self.download_button.clicked.connect(self.start_model_download)
        button_row.addWidget(self.download_button)

        self.process_button = QtWidgets.QPushButton("Run Whisper Viseme Analysis")
        self.process_button.clicked.connect(self.run_whisper)
        button_row.addWidget(self.process_button)
        layout.addLayout(button_row)

        self.console = QtWidgets.QTextEdit()
        self.console.setReadOnly(True)
        layout.addWidget(self.console)

    def build_jobs_panel(self):
        self.jobs_dock = QtWidgets.QDockWidget("Jobs", self)
        self.jobs_dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )

        jobs_widget = QtWidgets.QWidget()
        jobs_layout = QtWidgets.QVBoxLayout(jobs_widget)

        self.jobs_table = QtWidgets.QTableWidget(0, 6)
        self.jobs_table.setHorizontalHeaderLabels(
            ["Job", "Type", "Target", "Stage", "Progress", "Status"]
        )
        self.jobs_table.verticalHeader().setVisible(False)
        self.jobs_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.jobs_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        header = self.jobs_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        jobs_layout.addWidget(self.jobs_table)

        self.jobs_dock.setWidget(jobs_widget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.jobs_dock)

    def build_settings_panel(self):
        self.settings_dock = QtWidgets.QDockWidget("Settings", self)
        self.settings_dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )

        settings_widget = QtWidgets.QWidget()
        settings_layout = QtWidgets.QVBoxLayout(settings_widget)

        cache_group = QtWidgets.QGroupBox("Model Cache")
        cache_layout = QtWidgets.QVBoxLayout(cache_group)
        cache_path_row = QtWidgets.QHBoxLayout()
        self.cache_dir_input = QtWidgets.QLineEdit()
        cache_path_row.addWidget(self.cache_dir_input)

        self.cache_dir_browse_button = QtWidgets.QPushButton("Browse")
        self.cache_dir_browse_button.clicked.connect(self.browse_cache_dir)
        cache_path_row.addWidget(self.cache_dir_browse_button)
        cache_layout.addLayout(cache_path_row)

        self.cache_dir_save_button = QtWidgets.QPushButton("Apply Cache Directory")
        self.cache_dir_save_button.clicked.connect(self.apply_cache_dir_setting)
        cache_layout.addWidget(self.cache_dir_save_button)
        settings_layout.addWidget(cache_group)

        models_group = QtWidgets.QGroupBox("Additional Models")
        models_layout = QtWidgets.QVBoxLayout(models_group)

        self.saved_models_table = QtWidgets.QTableWidget(0, 2)
        self.saved_models_table.setHorizontalHeaderLabels(["Name", "Checkpoint Path"])
        self.saved_models_table.verticalHeader().setVisible(False)
        self.saved_models_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.saved_models_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.saved_models_table.itemSelectionChanged.connect(
            self.on_saved_model_selection_changed
        )
        saved_header = self.saved_models_table.horizontalHeader()
        saved_header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        saved_header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        models_layout.addWidget(self.saved_models_table)

        self.model_name_input = QtWidgets.QLineEdit()
        self.model_name_input.setPlaceholderText("Friendly name, for example Song Model")
        models_layout.addWidget(self.model_name_input)

        model_path_row = QtWidgets.QHBoxLayout()
        self.model_path_input = QtWidgets.QLineEdit()
        self.model_path_input.setPlaceholderText("Path to a Whisper-compatible .pt checkpoint")
        model_path_row.addWidget(self.model_path_input)

        self.model_path_browse_button = QtWidgets.QPushButton("Browse")
        self.model_path_browse_button.clicked.connect(self.browse_saved_model_path)
        model_path_row.addWidget(self.model_path_browse_button)
        models_layout.addLayout(model_path_row)

        model_button_row = QtWidgets.QHBoxLayout()
        self.save_model_button = QtWidgets.QPushButton("Add / Update Model")
        self.save_model_button.clicked.connect(self.save_additional_model)
        model_button_row.addWidget(self.save_model_button)

        self.remove_model_button = QtWidgets.QPushButton("Remove Selected Model")
        self.remove_model_button.clicked.connect(self.remove_selected_model)
        model_button_row.addWidget(self.remove_model_button)
        models_layout.addLayout(model_button_row)
        settings_layout.addWidget(models_group)
        settings_layout.addStretch(1)

        self.settings_dock.setWidget(settings_widget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.settings_dock)

    def log(self, message):
        self.console.append(message)
        QtWidgets.QApplication.processEvents()

    def populate_settings_widgets(self):
        self.cache_dir_input.setText(str(self.cache_dir))
        self.refresh_saved_models_table()

    def refresh_model_combo(self):
        previous_data = self.model_combo.currentData()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()

        for entry in MODEL_PRESETS:
            self.model_combo.addItem(entry["label"], dict(entry))

        saved_models = self.settings.get("additional_models", [])
        if saved_models:
            self.model_combo.insertSeparator(self.model_combo.count())
            for entry in saved_models:
                item = {
                    "kind": "saved_custom",
                    "label": entry["name"],
                    "model": entry["path"],
                    "notes": f"Saved custom checkpoint:\n{entry['path']}",
                }
                self.model_combo.addItem(entry["name"], item)

        self.model_combo.insertSeparator(self.model_combo.count())
        self.model_combo.addItem(ONE_OFF_MODEL_ENTRY["label"], dict(ONE_OFF_MODEL_ENTRY))

        if previous_data:
            for index in range(self.model_combo.count()):
                item = self.model_combo.itemData(index)
                if not item:
                    continue
                if (
                    item.get("kind") == previous_data.get("kind")
                    and item.get("model") == previous_data.get("model")
                ):
                    self.model_combo.setCurrentIndex(index)
                    break

        if self.model_combo.currentIndex() < 0:
            self.model_combo.setCurrentIndex(0)

        self.model_combo.blockSignals(False)
        self.on_model_changed()

    def refresh_saved_models_table(self):
        saved_models = self.settings.get("additional_models", [])
        self.saved_models_table.setRowCount(len(saved_models))
        for row, entry in enumerate(saved_models):
            self.saved_models_table.setItem(row, 0, QtWidgets.QTableWidgetItem(entry["name"]))
            self.saved_models_table.setItem(row, 1, QtWidgets.QTableWidgetItem(entry["path"]))

    def current_model_entry(self):
        return self.model_combo.currentData()

    def on_model_changed(self):
        entry = self.current_model_entry()
        if not entry:
            return

        is_one_off = entry["kind"] == "ad_hoc"
        self.custom_model_widget.setVisible(is_one_off)
        self.download_button.setEnabled(entry["kind"] == "official")
        self.model_notes.setText(entry["notes"])
        self.runtime_notes.setText(
            f"Runtime device: {self.device} | Model cache: {self.cache_dir}"
        )

    def select_audio(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio files (*.wav *.mp3 *.flac *.m4a *.ogg *.aac);;All files (*)",
        )
        if path:
            self.audio_path = path
            self.file_label.setText(path)
            self.log(f"Selected: {path}")

    def select_custom_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Whisper Checkpoint",
            "",
            "Whisper checkpoints (*.pt);;All files (*)",
        )
        if path:
            self.custom_model_input.setText(path)
            self.log(f"One-off custom model selected: {path}")

    def browse_cache_dir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Whisper Cache Directory",
            self.cache_dir_input.text() or str(BASE_DIR),
        )
        if path:
            self.cache_dir_input.setText(path)

    def apply_cache_dir_setting(self):
        self.cache_dir = resolve_cache_dir(self.cache_dir_input.text().strip())
        self.settings["cache_dir"] = str(self.cache_dir)
        save_settings(self.settings_path, self.settings)
        self.runtime_notes.setText(
            f"Runtime device: {self.device} | Model cache: {self.cache_dir}"
        )
        self.log(f"Updated Whisper cache directory to {self.cache_dir}")

    def browse_saved_model_path(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Whisper Checkpoint",
            "",
            "Whisper checkpoints (*.pt);;All files (*)",
        )
        if path:
            self.model_path_input.setText(path)

    def on_saved_model_selection_changed(self):
        row = self.saved_models_table.currentRow()
        if row < 0:
            return

        saved_models = self.settings.get("additional_models", [])
        if row >= len(saved_models):
            return

        entry = saved_models[row]
        self.model_name_input.setText(entry["name"])
        self.model_path_input.setText(entry["path"])

    def save_additional_model(self):
        name = self.model_name_input.text().strip()
        path = self.model_path_input.text().strip()

        if not name or not path:
            self.log("Model name and checkpoint path are required.")
            return

        checkpoint = Path(path).expanduser()
        if not checkpoint.is_file():
            self.log(f"Model checkpoint not found: {checkpoint}")
            return

        entry = {"name": name, "path": str(checkpoint)}
        saved_models = self.settings.get("additional_models", [])

        row = self.saved_models_table.currentRow()
        if 0 <= row < len(saved_models):
            saved_models[row] = entry
            action = "Updated"
        else:
            saved_models.append(entry)
            action = "Added"

        self.settings["additional_models"] = saved_models
        save_settings(self.settings_path, self.settings)
        self.refresh_saved_models_table()
        self.refresh_model_combo()
        self.log(f"{action} saved model '{name}'.")
        self.model_name_input.clear()
        self.model_path_input.clear()
        self.saved_models_table.clearSelection()

    def remove_selected_model(self):
        row = self.saved_models_table.currentRow()
        saved_models = self.settings.get("additional_models", [])
        if row < 0 or row >= len(saved_models):
            self.log("Select a saved model to remove.")
            return

        removed = saved_models.pop(row)
        self.settings["additional_models"] = saved_models
        save_settings(self.settings_path, self.settings)
        self.refresh_saved_models_table()
        self.refresh_model_combo()
        self.log(f"Removed saved model '{removed['name']}'.")

    def selected_model_for_job(self):
        entry = self.current_model_entry()
        if not entry:
            raise ValueError("Select a model first.")

        if entry["kind"] != "ad_hoc":
            return dict(entry)

        custom_path = self.custom_model_input.text().strip()
        if not custom_path:
            raise ValueError("Select a local checkpoint for the one-off model entry first.")

        checkpoint = Path(custom_path).expanduser()
        if not checkpoint.is_file():
            raise ValueError(f"Custom model not found: {checkpoint}")

        return {
            "kind": "saved_custom",
            "label": checkpoint.name,
            "model": str(checkpoint),
            "notes": f"One-off local checkpoint:\n{checkpoint}",
        }

    def next_job_id(self):
        self.job_counter += 1
        return f"job-{self.job_counter:03d}"

    def add_job_row(self, job_id, job_type, target):
        row = self.jobs_table.rowCount()
        self.jobs_table.insertRow(row)
        self.jobs_table.setItem(row, 0, QtWidgets.QTableWidgetItem(job_id))
        self.jobs_table.setItem(row, 1, QtWidgets.QTableWidgetItem(job_type))
        self.jobs_table.setItem(row, 2, QtWidgets.QTableWidgetItem(target))
        self.jobs_table.setItem(row, 3, QtWidgets.QTableWidgetItem("Queued"))

        progress_bar = QtWidgets.QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        progress_bar.setFormat("%p%")
        self.jobs_table.setCellWidget(row, 4, progress_bar)

        self.jobs_table.setItem(row, 5, QtWidgets.QTableWidgetItem("Queued"))
        self.job_rows[job_id] = {"row": row, "progress": progress_bar}
        self.jobs_dock.raise_()

    def update_job_stage(self, job_id, stage):
        row_info = self.job_rows.get(job_id)
        if not row_info:
            return
        self.jobs_table.item(row_info["row"], 3).setText(stage)

    def update_job_progress(self, job_id, value):
        row_info = self.job_rows.get(job_id)
        if not row_info:
            return
        row_info["progress"].setValue(value)

    def update_job_status(self, job_id, status):
        row_info = self.job_rows.get(job_id)
        if not row_info:
            return
        self.jobs_table.item(row_info["row"], 5).setText(status)

    def handle_job_log(self, job_id, message):
        self.log(f"[{job_id}] {message}")

    def handle_job_completed(self, job_id, result):
        message = result.get("message", "Job completed.")
        self.log(f"[{job_id}] {message}")
        if "output_dir" in result:
            self.log(f"[{job_id}] Output folder: {result['output_dir']}")

    def handle_job_failed(self, job_id, error_message):
        self.log(f"[{job_id}] Failed: {error_message}")

    def cleanup_job(self, job_id):
        thread_info = self.job_threads.pop(job_id, None)
        if thread_info:
            thread, worker = thread_info
            worker.deleteLater()
            thread.deleteLater()
        self.active_transcription_jobs.discard(job_id)

    def start_job(self, mode, model_entry, audio_path="", transcript=""):
        job_id = self.next_job_id()
        job_type = "Download" if mode == "download" else "Transcribe"
        target_text = model_entry["label"]
        if audio_path:
            target_text = f"{Path(audio_path).name} | {model_entry['label']}"

        self.add_job_row(job_id, job_type, target_text)
        if mode == "transcribe":
            self.active_transcription_jobs.add(job_id)

        thread = QtCore.QThread(self)
        worker = WhisperJobWorker(
            job_id=job_id,
            mode=mode,
            model_entry=model_entry,
            cache_dir=self.cache_dir,
            device=detect_best_device(),
            audio_path=audio_path,
            transcript=transcript,
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress_changed.connect(self.update_job_progress)
        worker.stage_changed.connect(self.update_job_stage)
        worker.status_changed.connect(self.update_job_status)
        worker.log_message.connect(self.handle_job_log)
        worker.completed.connect(self.handle_job_completed)
        worker.failed.connect(self.handle_job_failed)
        worker.finished.connect(thread.quit)
        thread.finished.connect(lambda finished_job_id=job_id: self.cleanup_job(finished_job_id))

        self.job_threads[job_id] = (thread, worker)
        thread.start()

        self.log(f"[{job_id}] Started {job_type.lower()} job for {model_entry['label']}.")
        return job_id

    def start_model_download(self):
        try:
            model_entry = self.selected_model_for_job()
        except ValueError as exc:
            self.log(str(exc))
            return

        self.start_job("download", model_entry)

    def run_whisper(self):
        if not self.audio_path:
            self.log("No audio file selected.")
            return

        if self.active_transcription_jobs:
            self.log("A transcription job is already running. Wait for it to finish before starting another.")
            return

        try:
            model_entry = self.selected_model_for_job()
        except ValueError as exc:
            self.log(str(exc))
            return

        transcript = self.transcript_input.toPlainText().strip()
        self.start_job(
            "transcribe",
            model_entry,
            audio_path=self.audio_path,
            transcript=transcript,
        )


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    win = WhisperVisemeApp()
    win.show()
    app.exec()

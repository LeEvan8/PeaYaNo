"""
transcriber.py - The Final 'Smart v1' Architecture (Math Grid + Single Track)
"""

import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

import librosa
import torch
import numpy as np
import pretty_midi

SAMPLE_RATE = 16000
MODEL_URL = "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
MODEL_DIR = Path.home() / "peayano_models"
MODEL_PATH = MODEL_DIR / "CRNN_note_F1=0.9677_pedal_F1=0.9186.pth"
MODEL_MIN_BYTES = 10_000_000

ProgressCallback = Optional[Callable[[str], None]]

class TranscriptionError(Exception):
    pass

def _log(message: str, callback: ProgressCallback = None):
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line)
    if callback:
        callback(line)

def get_device() -> str:
    try:
        if torch.cuda.is_available():
            torch.zeros(1).cuda()
            return "cuda"
    except Exception:
        pass
    return "cpu"

def ensure_model_weights(progress_callback: ProgressCallback = None) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size >= MODEL_MIN_BYTES:
        return MODEL_PATH

    _log("Downloading model checkpoint (165MB)...", progress_callback)
    tmp_path = MODEL_PATH.with_suffix(".part")

    def _report_hook(block_num, block_size, total_size):
        if progress_callback and total_size > 0:
            progress_callback(f"Downloading: {min(100, block_num * block_size * 100 // total_size)}%")

    try:
        urllib.request.urlretrieve(MODEL_URL, tmp_path, _report_hook)
        tmp_path.replace(MODEL_PATH)
    except Exception as exc:
        raise TranscriptionError(f"Failed to download AI model: {exc}")
    
    return MODEL_PATH

def find_musescore() -> Optional[Path]:
    paths = [
        Path(r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"),
        Path(r"C:\Program Files\MuseScore Studio 4\bin\MuseScoreStudio.exe"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "MuseScore 4" / "bin" / "MuseScore4.exe"
    ]
    for p in paths:
        if p.exists(): return p
    return Path(shutil.which("MuseScore4.exe")) if shutil.which("MuseScore4.exe") else None

def clean_and_quantize_midi(raw_midi_path: str, clean_midi_path: str, detected_bpm: float, grid_division: int, gap_threshold_ms: float):
    pm = pretty_midi.PrettyMIDI(raw_midi_path)
    pm.tempo_changes = (np.array([0.0]), np.array([detected_bpm]))
    
    beat_duration = 60.0 / detected_bpm
    step_duration = beat_duration / (grid_division / 4.0)
    gap_threshold_sec = gap_threshold_ms / 1000.0
    
    raw_notes = []
    for inst in pm.instruments:
        raw_notes.extend(inst.notes)
        
    # Phase 1: Legato Gap Cleanup
    notes_by_pitch = {}
    for note in raw_notes:
        notes_by_pitch.setdefault(note.pitch, []).append(note)
        
    processed_notes = []
    for pitch, notes in notes_by_pitch.items():
        notes.sort(key=lambda n: n.start)
        for i in range(len(notes) - 1):
            curr_note = notes[i]
            next_note = notes[i + 1]
            gap = next_note.start - curr_note.end
            if 0 < gap <= gap_threshold_sec:
                curr_note.end = next_note.start
        processed_notes.extend(notes)
    
    processed_notes.sort(key=lambda n: (n.start, n.pitch))
    
    piano = pretty_midi.Instrument(program=0, name="Piano")

    # Phase 2: Strict Math Quantization
    for note in processed_notes:
        snapped_start = round(note.start / step_duration) * step_duration
        raw_duration = note.end - note.start
        snapped_duration = max(step_duration, round(raw_duration / step_duration) * step_duration)
        max_allowed_duration = beat_duration * 4.0
        snapped_duration = min(snapped_duration, max_allowed_duration)
        
        note.start = snapped_start
        note.end = snapped_start + snapped_duration
        piano.notes.append(note)
        
    # Assigning to a single track forces MuseScore to render exactly 1 Grand Staff (2 staves)
    pm.instruments = [piano]
    pm.write(clean_midi_path)

def engrave_musescore(midi_path: str, out_pdf: str, out_xml: str, ms_path: str, progress_callback: ProgressCallback = None):
    _log("Rendering single-track quantized MIDI via MuseScore CLI...", progress_callback)
    
    cmd_pdf = [str(ms_path), "-o", str(out_pdf), str(midi_path)]
    cmd_xml = [str(ms_path), "-o", str(out_xml), str(midi_path)]
    
    try:
        subprocess.run(cmd_pdf, check=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        subprocess.run(cmd_xml, check=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except Exception as exc:
        raise TranscriptionError(f"MuseScore rendering failed: {exc}")

def run_full_pipeline(audio_path: str, work_dir: str, musescore_path: str, device: str, bpm: float, grid_division: int, gap_threshold_ms: float, progress_callback: ProgressCallback = None) -> dict:
    from piano_transcription_inference import PianoTranscription
    
    wd = Path(work_dir)
    raw_midi = wd / "raw.mid"
    clean_midi = wd / "clean.mid"
    pdf_path = wd / "score.pdf"
    xml_path = wd / "score.musicxml"

    checkpoint = ensure_model_weights(progress_callback)
    
    _log("Resampling audio and estimating BPM...", progress_callback)
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    
    estimated_tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    actual_bpm = float(np.atleast_1d(estimated_tempo)[0]) if bpm == 120 else bpm
    _log(f"Using Tempo: {actual_bpm:.1f} BPM", progress_callback)
    
    _log("Running AI Inference...", progress_callback)
    resolved_device = get_device() if device == "auto" else device
    try:
        ai = PianoTranscription(device=resolved_device, checkpoint_path=str(checkpoint))
        ai.transcribe(audio, str(raw_midi))
    except Exception as exc:
        raise TranscriptionError(f"AI Model failed: {exc}")

    _log("Applying strict math quantization and gap cleanup...", progress_callback)
    clean_and_quantize_midi(str(raw_midi), str(clean_midi), actual_bpm, grid_division, gap_threshold_ms)
    
    engrave_musescore(str(clean_midi), str(pdf_path), str(xml_path), musescore_path, progress_callback)
    
    return {"midi": str(clean_midi), "pdf": str(pdf_path), "musicxml": str(xml_path)}
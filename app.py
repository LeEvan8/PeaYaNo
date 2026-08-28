import base64
import shutil
import tempfile
from pathlib import Path

import streamlit as st

from transcriber import (
    TranscriptionError,
    find_musescore,
    get_device,
    run_full_pipeline,
)

st.set_page_config(page_title="PeaYaNo - AI Piano Transcription", page_icon="🎹", layout="centered")

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac"}
MAX_DURATION_SECONDS = 5 * 60
SESSION_TMP_KEY = "peayano_tmp_dir"
LOG_TAIL_LINES = 14

def start_fresh_session_dir() -> Path:
    previous = st.session_state.get(SESSION_TMP_KEY)
    if previous and Path(previous).exists():
        shutil.rmtree(Path(previous), ignore_errors=True)
    new_dir = tempfile.mkdtemp(prefix="peayano_")
    st.session_state[SESSION_TMP_KEY] = new_dir
    return Path(new_dir)

def render_sidebar():
    with st.sidebar:
        st.header("Hardware Settings")
        device_choice = st.selectbox("Processing device", ["auto", "cuda", "cpu"], index=0)
        gpu_available = get_device() == "cuda"
        st.write(f"GPU detected: {'✅ Yes' if gpu_available else '❌ No (CPU)'}")
        
        auto_ms_path = find_musescore()
        manual_ms_path = st.text_input(
            "MuseScore 4 Path",
            value=str(auto_ms_path) if auto_ms_path else "",
        )

        st.divider()
        st.header("Math Quantization Settings")
        st.caption("Controls metronome snapping to generate clean Grand Staves.")
        
        bpm_input = st.number_input("Target Tempo (BPM)", min_value=30, max_value=300, value=120, step=1)
        
        grid_input = st.selectbox("Strict Snap Grid", [8, 16, 32], index=1, 
                                  format_func=lambda x: f"1/{x} Notes")
                                  
        gap_input = st.slider("Legato Gap Fill (ms)", min_value=0, max_value=200, value=50, step=10)

    resolved_ms_path = manual_ms_path.strip() or (str(auto_ms_path) if auto_ms_path else None)
    return device_choice, resolved_ms_path, bpm_input, grid_input, gap_input

def render_pdf_preview(pdf_bytes: bytes) -> None:
    try:
        b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="600" style="border:1px solid #444;"></iframe>',
            unsafe_allow_html=True,
        )
    except Exception:
        st.info("Inline preview unavailable.")

def main():
    st.title("🎹 PeaYaNo")
    st.caption("AI piano transcription — Strict Math Grid with Native 2-Staff Engraving.")

    device_choice, ms_path, bpm, grid, gap_threshold = render_sidebar()
    
    uploaded_file = st.file_uploader("Drop a piano recording here", type=["mp3", "wav", "m4a", "flac"])

    if not uploaded_file:
        return

    work_dir = start_fresh_session_dir()
    input_path = work_dir / f"input{Path(uploaded_file.name).suffix.lower()}"
    input_path.write_bytes(uploaded_file.getbuffer())

    st.audio(uploaded_file)
    
    if st.button("🎼 Engrave Clean PDF", type="primary", use_container_width=True):
        log_box = st.empty()
        logs = []

        def progress_callback(line: str):
            logs.append(line)
            log_box.code("\n".join(logs[-LOG_TAIL_LINES:]), language="text")

        try:
            results = run_full_pipeline(
                audio_path=str(input_path),
                work_dir=str(work_dir),
                musescore_path=ms_path,
                device=device_choice,
                bpm=bpm,
                grid_division=grid,
                gap_threshold_ms=gap_threshold,
                progress_callback=progress_callback,
            )
            
            st.success("Transcription complete! Enjoy your standard 2-staff sheet music.")
            
            pdf_path = Path(results["pdf"])
            if pdf_path.exists():
                pdf_bytes = pdf_path.read_bytes()
                st.download_button("⬇️ Download PDF", data=pdf_bytes, file_name="score.pdf", mime="application/pdf", use_container_width=True)
                render_pdf_preview(pdf_bytes)

            c1, c2 = st.columns(2)
            c1.download_button("⬇️ Download MusicXML", data=Path(results["musicxml"]).read_bytes(), file_name="score.musicxml", use_container_width=True)
            c2.download_button("⬇️ Download Clean MIDI", data=Path(results["midi"]).read_bytes(), file_name="clean_score.mid", use_container_width=True)

        except TranscriptionError as exc:
            st.error(f"Failed: {exc}")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")

if __name__ == "__main__":
    main()
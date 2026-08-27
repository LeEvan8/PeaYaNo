# PeaYaNo 🎹

AI piano transcription for Windows — drop in an audio recording, get back pre-quantized, engraved PDF sheet music, MusicXML, and MIDI.

## Step-by-Step Manual Setup

### Step 1: Install System Software
1. Download **Python 3.10 (64-bit)** from python.org.
2. Run the installer and check **"Add python.exe to PATH"** before clicking Install.
3. Open PowerShell and run: `winget install --id Gyan.FFmpeg -e`
4. Download and install **MuseScore 4** from musescore.org (leave the default installation path).
5. Restart your computer so Windows registers the new FFmpeg commands.

### Step 2: Create the Python Environment
Open PowerShell in your project folder and run the following commands one by one to create a clean, isolated environment:

```powershell
py -3.10 -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip


streamlit run app.py
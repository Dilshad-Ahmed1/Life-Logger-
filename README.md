# Life-Logger-
Smart Audio Assistant

A Streamlit-based web app to upload audio, transcribe it, extract tasks/reminders, store conversation history, and perform QA on audio.



# Features 

Upload audio files (.wav, .mp3, .m4a, .flac).

Transcribe audio using OpenAI Whisper.

Extract tasks and reminders from transcripts.

Store conversation history.

Ask questions about uploaded audio using QA module.

# Tech Stack

Python 3.10+

Streamlit – for interactive web UI.

OpenAI Whisper – for audio transcription.

SentenceTransformers – for embeddings in QA module.

Transformers (Hugging Face) – for question answering.

PyTorch – required for Whisper and Transformers models.

SoundFile / librosa – for audio file handling.

# Dependencies

Install Python packages using pip:

# Core web app
pip install streamlit

# Audio transcription
pip install openai-whisper
pip install torch --index-url https://download.pytorch.org/whl/cu118  # GPU optional

# QA module
pip install transformers
pip install sentence-transformers

# Audio handling
pip install librosa soundfile numpy

# Optional (for temp files)
pip install pathlib

Running the App

Clone the repository:

git clone <your-repo-url>
cd <your-project-folder>


Run Streamlit:

streamlit run app.py


Open the browser at the URL shown in the terminal.

Upload an audio file and interact with the app.

Notes

Only audio upload is supported in this version. Browser recording requires streamlit-webrtc integration (added in later versions).

GPU usage is optional but recommended for faster transcription.

QA module depends on Hugging Face Transformers models downloaded on first run.

This is the baseline version before we added:

Browser-based recording (streamlit-webrtc).

Beautified and colorful UI.

Speaker diarization using WhisperX.

Real-time conversation history updates.


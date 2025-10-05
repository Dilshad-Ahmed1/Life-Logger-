# app.py
import streamlit as st
import tempfile
import os
import time
from datetime import datetime
import json
import io
import matplotlib.pyplot as plt

# Try optional recording (streamlit-webrtc)
try:
    from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
    WEBSOCKET_AVAILABLE = True
except Exception:
    WEBSOCKET_AVAILABLE = False

# Import your backend functions (assumes the two files are in same dir)
from behavior_from_audio import (
    transcribe,
    extract_acoustic,
    enrich_with_text_rates,
    analyze_text,
    score_behavior,
    summarize_behavior,
)
from audio_conversation_qa import (
    answer_from_audio,
    transcribe_audio,
    extract_reminder_lines,
    chunk_segments,
    chunk_segments as _chunk_segments,  # alias if needed
)

st.set_page_config(page_title="Audio Insights — Transcript, QA & Behavior", layout="wide")

# --- Helpers ---
def save_uploaded_file(uploaded) -> str:
    suffix = "." + uploaded.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        return tmp.name

def add_history_entry(session, entry):
    if "history" not in session:
        session["history"] = []
    session["history"].insert(0, entry)
    # keep only last 50
    session["history"] = session["history"][:50]

def format_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# --- UI Layout ---
st.title("🎧 Audio Insights — Transcript · QA · Tasks · Behavior")
st.markdown("Upload an audio file (wav/mp3/m4a/flac/ogg/mp4). If you want in-browser recording, install `streamlit-webrtc`.")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    whisper_model = st.selectbox("Whisper model", ["tiny", "base", "small", "medium", "large"], index=2)
    language = st.text_input("Force language (optional, e.g. en)", value="")
    show_verbose = st.checkbox("Show raw transcript segments", value=False)
    st.markdown("---")
    st.markdown("Quick actions")
    clear_hist = st.button("Clear saved history")

if clear_hist:
    st.session_state.pop("history", None)
    st.success("History cleared.")

# Recording block (optional)
audio_path = None
st.subheader("1) Provide audio")
col1, col2 = st.columns([2,1])
with col1:
    uploaded = st.file_uploader("Upload audio file", type=["wav","mp3","m4a","flac","ogg","mp4"])
    if uploaded:
        audio_path = save_uploaded_file(uploaded)
        st.audio(audio_path)
    else:
        st.info("You can upload an audio file above. Optionally record in-browser if available.")

with col2:
    if WEBSOCKET_AVAILABLE:
        st.markdown("### Record (experimental)")
        ctx = st.empty()
        # Minimal example: capture 5 seconds then save - streamlit-webrtc advanced usage may be needed
        webrtc_ctx = webrtc_streamer(key="audio", mode=WebRtcMode.SENDONLY, 
                                    media_stream_constraints={"audio": True, "video": False})
        if webrtc_ctx and webrtc_ctx.state.playing:
            st.caption("Recording... press stop to finish and then use the 'Save recording' button.")
            if st.button("Save recording"):
                # Accessing recorded audio programmatically requires custom AudioProcessor.
                st.warning("Browser recording saving is only available with a custom AudioProcessor. Use upload for now.")
    else:
        st.caption("Browser recording disabled (install `streamlit-webrtc` to enable).")

# Tabs for functionality
tabs = st.tabs(["Conversation QA", "Behavior Analysis", "Tasks & Reminders", "History"])

# -------- Conversation QA Tab --------
with tabs[0]:
    st.header("Conversation QA")
    st.markdown("Ask questions about the uploaded conversation. Uses your `audio_conversation_qa.py` pipeline.")

    qa_question = st.text_input("Question about the audio", value="")
    top_k = st.slider("Top-k chunks to retrieve", min_value=1, max_value=10, value=5)
    gen_model = st.text_input("Generative fallback model (HF)", value="google/flan-t5-large")
    extractive_model = st.text_input("Extractive model (HF)", value="deepset/roberta-base-squad2")
    retriever = st.text_input("Retriever model (SentenceTransformer)", value="sentence-transformers/all-MiniLM-L6-v2")

    cola, colb = st.columns([1,1])
    with cola:
        run_qa = st.button("🔍 Run QA")
    with colb:
        run_transcribe = st.button("📝 Transcribe (for QA & tasks)")

    if run_transcribe and audio_path:
        with st.spinner("Transcribing with Whisper..."):
            # Uses the chunking-aware transcribe in your QA module to get segments
            segments = transcribe_audio(audio_path, whisper_model=whisper_model, language=(language or None))
            full_text = " ".join(s.text for s in segments)
            st.success("Transcription complete.")
            st.text_area("Full transcript", value=full_text, height=220)
            # Save transcript to session
            add_history_entry(st.session_state, {"type": "transcript", "time": format_timestamp(), "transcript": full_text, "audio_path": audio_path})

    if run_qa and audio_path and qa_question.strip():
        with st.spinner("Running retrieval + QA... (this may take a while)"):
            try:
                # answer_from_audio calls transcribe_audio internally, but we can pass same models
                result = answer_from_audio(
                    audio_path=audio_path,
                    question=qa_question,
                    whisper_model=whisper_model,
                    language=(language or None),
                    retriever_name=retriever,
                    extractive_model=extractive_model,
                    generative_model=gen_model,
                    top_k=top_k
                )
                st.subheader("Answer")
                st.write(result.get("answer") or "(no answer)")
                if result.get("confidence") is not None:
                    st.write("Confidence:", result["confidence"])
                st.subheader("Source timestamps")
                for i, src in enumerate(result.get("sources", []), 1):
                    win = src.get("window", ["?", "?"])
                    if "precise_span" in src:
                        ps = src["precise_span"]
                        st.write(f"{i}. Window {win[0]} → {win[1]}  |  Span {ps[0]} → {ps[1]}")
                    else:
                        st.write(f"{i}. Window {win[0]} → {win[1]}")
                # Save QA result to history
                add_history_entry(st.session_state, {"type": "qa", "time": format_timestamp(), "question": qa_question, "answer": result.get("answer"), "sources": result.get("sources")})
            except Exception as e:
                st.error(f"QA failed: {e}")

# -------- Behavior Analysis Tab --------
with tabs[1]:
    st.header("Behavior Analysis")
    st.markdown("Run the behavior analysis pipeline (pitch/energy/pauses + text features). Uses your `behavior_from_audio.py` functions.")

    run_behavior = st.button("🧠 Analyze Behavior")

    if run_behavior:
        if not audio_path:
            st.warning("Please upload an audio file first.")
        else:
            with st.spinner("Running behavior analysis..."):
                # Full ASR using transcribe() - lighter than the QA module
                transcript, duration_s = transcribe(audio_path, whisper_model=whisper_model, language=(language or None))
                ac = enrich_with_text_rates(extract_acoustic(audio_path), transcript)
                tx = analyze_text(transcript)
                scores = score_behavior(ac, tx)
                summary = summarize_behavior(scores, ac, tx, transcript)

                st.success("Behavior analysis complete.")
                st.subheader("Scores")
                # Nice horizontal bar chart using matplotlib
                score_items = [
                    ("Confidence", scores.confidence),
                    ("Stress/Tension", scores.stress_tension),
                    ("Empathy", scores.empathy),
                    ("Assertiveness", scores.assertiveness),
                    ("Politeness", scores.politeness),
                    ("Engagement", scores.engagement),
                ]
                labels, vals = zip(*score_items)
                fig, ax = plt.subplots(figsize=(8, 3))
                bars = ax.barh(labels, vals)
                ax.set_xlim(0, 100)
                ax.invert_yaxis()
                for i, b in enumerate(bars):
                    ax.text(b.get_width() + 1, b.get_y() + b.get_height()/2, f"{vals[i]:d}", va="center")
                st.pyplot(fig)

                st.subheader("Acoustic & Text Features")
                col1, col2 = st.columns(2)
                with col1:
                    st.json(ac.__dict__)
                with col2:
                    st.json(tx.__dict__)

                st.subheader("Explainable Summary")
                st.text_area("Summary", value=summary, height=260)

                # Offer download
                out = {
                    "time": format_timestamp(),
                    "transcript": transcript,
                    "scores": scores.__dict__,
                    "acoustic": ac.__dict__,
                    "text_features": tx.__dict__,
                    "summary": summary,
                }
                buf = io.BytesIO(json.dumps(out, indent=2).encode("utf-8"))
                st.download_button("Download analysis JSON", data=buf, file_name="behavior_analysis.json", mime="application/json")

                # Save to history
                add_history_entry(st.session_state, {"type": "behavior", "time": format_timestamp(), "summary": summary, "scores": scores.__dict__, "audio_path": audio_path})

# -------- Tasks & Reminders Tab --------
with tabs[2]:
    st.header("Tasks & Reminders")
    st.markdown("Extracts reminder/task-like lines from the transcript (uses extraction logic from `audio_conversation_qa.py`).")

    run_tasks = st.button("🔖 Extract tasks from transcript")
    if run_tasks:
        if not audio_path:
            st.warning("Please upload audio first.")
        else:
            with st.spinner("Transcribing & scanning for tasks..."):
                # Use the QA transcribe (timed segments) to extract reminders with timestamps
                segments = transcribe_audio(audio_path, whisper_model=whisper_model, language=(language or None))
                full_text = " ".join(s.text for s in segments)
                tasks = extract_reminder_lines(full_text)
                if tasks:
                    st.success(f"Found {len(tasks)} reminder/task(s).")
                    for i, t in enumerate(tasks, 1):
                        st.write(f"{i}. {t}")
                else:
                    st.info("No clear reminder/task lines found.")
                # Save to history
                add_history_entry(st.session_state, {"type": "tasks", "time": format_timestamp(), "tasks": tasks, "transcript": full_text})

# -------- History Tab --------
with tabs[3]:
    st.header("History")
    st.markdown("Recent analyses (transcripts, QA results, behaviors, tasks). Stored only in your browser session.")

    hist = st.session_state.get("history", [])
    if not hist:
        st.info("No history yet. Run an analysis to see entries.")
    else:
        for entry in hist:
            t = entry.get("time")
            etype = entry.get("type")
            with st.expander(f"{t} — {etype}"):
                if etype == "transcript":
                    st.text_area("Transcript", value=entry.get("transcript",""), height=180)
                elif etype == "qa":
                    st.write("Q:", entry.get("question"))
                    st.write("A:", entry.get("answer"))
                    st.write("Sources:", entry.get("sources"))
                elif etype == "behavior":
                    st.write("Scores:")
                    st.json(entry.get("scores", {}))
                    st.write("Summary:")
                    st.text(entry.get("summary",""))
                elif etype == "tasks":
                    st.write("Tasks:")
                    for i, x in enumerate(entry.get("tasks", []), 1):
                        st.write(f"{i}. {x}")
                # If audio path present, offer download link for original file
                if entry.get("audio_path") and os.path.exists(entry.get("audio_path")):
                    st.download_button("Download original audio", data=open(entry["audio_path"], "rb").read(), file_name=os.path.basename(entry["audio_path"]))

st.markdown("---")
st.caption("Built on your `behavior_from_audio.py` and `audio_conversation_qa.py`. Not clinical. All processing happens locally on your machine (unless you deploy elsewhere).")


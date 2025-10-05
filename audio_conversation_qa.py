.#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio → Conversation → Question Answering (Whisper + Lightweight RAG)

Usage:
  python audio_conversation_qa.py --audio "/path/to/audio_file.(wav|mp3|m4a|flac)" \
                                  --question "What did the speaker say about timelines?"

python audio_conversation_qa.py --audio "audio2.mp3" --question "explain iphone 17 pro"

What it does:
1) Transcribes the conversation from an audio file using OpenAI Whisper (offline).
2) Chunks the transcript with timestamps.
3) Retrieves the most relevant chunks for your question via semantic search.
4) Attempts extractive QA on each top chunk; if not confident, uses a generative fallback.
5) Prints the best answer and shows timestamped sources from the audio.

Dependencies (install if needed):   
  pip install openai-whisper transformers sentence-transformers torch numpy

Tip: For faster inference on supported hardware, install PyTorch with GPU support.
"""

import argparse #lets your script read command-line options like --input file.wav.
import re #“find/replace by pattern” (regular expressions).
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict #type hints to make code clearer.

import numpy as np #– fast math on arrays (vectors/matrices).
import torch #– PyTorch; runs neural nets on CPU/GPU.
import whisper #OpenAI’s speech-to-text (transcribes audio files).
from sentence_transformers import SentenceTransformer #turns sentences into vectors (embeddings) you can compare.
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


# ------------------------------ Utilities ------------------------------ #

def human_time(s: float) -> str:
    """Seconds → hh:mm:ss.mmm"""
    if s is None or np.isnan(s):
        return "?:?:?.???"
    s = max(0.0, float(s))
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}"


def device_auto() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ------------------------------ Data Types ------------------------------ #

@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Chunk:
    start: float
    end: float
    text: str
    seg_indices: List[int]


# ------------------------------ ASR (Whisper) ------------------------------ #

# def transcribe_audio(
#     audio_path: str,
#     whisper_model: str = "small",
#     language: Optional[str] = None,
#     device: Optional[str] = None,
# ) -> List[Segment]:
#     """
#     Transcribe audio to segments (start, end, text) using openai-whisper.
#     """
#     device = device or device_auto()
#     model = whisper.load_model(whisper_model, device=device)
#     # Better timestamps & punctuation handling
#     options = {"task": "transcribe", "fp16": (device == "cuda"), "language": language}
#     result = model.transcribe(audio_path, **{k: v for k, v in options.items() if v is not None})

#     segments = []
#     for seg in result.get("segments", []):
#         text = normalize_space(seg.get("text", ""))
#         if text:
#             segments.append(Segment(start=float(seg.get("start", 0.0)),
#                                     end=float(seg.get("end", 0.0)),
#                                     text=text))
#     return segments

# Assumes these exist/imported elsewhere:
# from typing import Optional, List
# import whisper
# from your_types import Segment           # dataclass with (start: float, end: float, text: str)
# from your_utils import device_auto, normalize_space  # helper functions you wrote

def transcribe_audio(
    audio_path: str,
    whisper_model: str = "small",
    language: Optional[str] = None,
    device: Optional[str] = None,
) -> List[Segment]:
    
    print(audio_path)
    print("vineet")

    """
    Transcribe audio to segments (start, end, text) using openai-whisper.
    Returns a list of Segment(start, end, text).
    """
   

    # Pick a compute device:
    # - use the provided `device` if given (e.g., "cuda", "cpu", "mps")
    # - otherwise auto-detect (e.g., prefer GPU if available)
    device = device or device_auto()

    # Load the Whisper model onto that device (sizes: tiny/base/small/medium/large-*)
    model = whisper.load_model(whisper_model, device=device)

    # Build decoding options:
    # task="transcribe" for speech->same-language text
    # fp16=True only if we're on CUDA (faster/less memory on NVIDIA GPUs)
    # language can be forced (e.g., "en"); if None, Whisper auto-detects
    options = {
        "task": "transcribe",
        "fp16": (device == "cuda"),
        "language": language
    }

    # Run transcription. We pass only non-None options to Whisper.
    result = model.transcribe(
        audio_path,
        **{k: v for k, v in options.items() if v is not None}
    

    )
   

    # Collect cleaned segments: (start time, end time, text)
    segments = []
    for seg in result.get("segments", []):
        # Normalize whitespace/punctuation quirks from ASR output
        text = normalize_space(seg.get("text", ""))
        # Only keep segments that actually have text
        if text:
            segments.append(
                Segment(
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    text=text
                )
            )
            
    full_text = " ".join(s.text for s in segments)
    # print(full_text)
    
    extract_reminder_lines(full_text)


    # print_reminders_schedules_tasks(full_text)

    return segments















import re
from typing import List, Tuple, Optional

# --- Optional: parse human dates to a tidy form if dateutil is available -----
try:
    from dateutil import parser as dtparse  # pip install python-dateutil
except Exception:
    dtparse = None  # graceful fallback; we’ll still show the time phrase text

# ----------------------------- Helpers ---------------------------------------
WS = re.compile(r"\s+")
def norm(s: str) -> str:
    return WS.sub(" ", s.strip())

# Time/Date phrase patterns (broad but practical)
TIME_PATTERNS = [
    r"\b(?:today|tomorrow|tmrw|tonight|this (?:morning|afternoon|evening|night))\b",
    r"\bnext (?:week|month|year|mon|tue(?:s)?|wed(?:nes)?|thu(?:rs)?|fri|sat(?:ur)?|sun)(?:day)?\b",
    r"\b(?:mon|tue(?:s)?|wed(?:nes)?|thu(?:rs)?|fri|sat(?:ur)?|sun)(?:day)?\b",
    r"\b(?:noon|midnight|eod|eom|eow|eoy)\b",
    r"\b(?:in|after)\s+\d+\s+(?:min(?:ute)?s?|hour(?:s)?|day(?:s)?|week(?:s)?|month(?:s)?|year(?:s)?)\b",
    r"\bby\s+(?:\d{1,2}(:\d{2})?\s?(?:am|pm)|noon|midnight|tomorrow|today|[A-Za-z]{3,9}(?:day)?)\b",
    r"\bat\s+\d{1,2}(:\d{2})?\s?(?:am|pm)\b",
    r"\b(?:\d{1,2}[:.]\d{2})\b",  # 24h or 12h without am/pm
    r"\b(?:\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?(?:\s+\d{2,4})?)\b",
    r"\b(?:\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|\d{1,2}[-/\.]\d{1,2}(?:[-/\.]\d{2,4})?)\b",
]
TIME_REGEX = re.compile("|".join(TIME_PATTERNS), re.IGNORECASE)

# Common “reminder-y” cues
TRIGGERS = [
    "remind me", "reminder", "remember to", "don't forget", "do not forget", "note to self",
    "to-do", "todo", "task", "action item", "deadline", "due",
    "schedule", "appointment", "meeting", "call", "follow up", "follow-up",
    "pay", "renew", "submit", "send", "email", "buy", "pick up", "collect", "book", "arrange",
]

BULLET = re.compile(r"^\s*(?:[-*•]|-\s*\[ \])\s+", re.IGNORECASE)

# ------------------------- Core extraction ------------------------------------
def split_sentences(text: str) -> List[str]:
    # Soft sentence splitter: periods, question/exclaim marks, or line breaks
    parts = re.split(r"(?<=[\.\!\?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]

def find_time_phrases(s: str) -> List[str]:
    matches = []
    for m in TIME_REGEX.finditer(s):
        chunk = norm(m.group(0))
        matches.append(chunk)
    # keep order & unique
    seen = set()
    uniq = []
    for x in matches:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(x)
    return uniq

def looks_like_reminder(s: str) -> bool:
    ls = s.lower()
    if BULLET.search(s):
        return True
    # has trigger OR (has a task-ish verb and a time phrase)
    has_trigger = any(t in ls for t in TRIGGERS)
    has_time = bool(TIME_REGEX.search(s))
    return has_trigger or has_time

def squeeze_task_text(s: str, time_chunks: List[str]) -> str:
    orig = " " + s + " "
    # Remove bullets
    s_ = BULLET.sub("", s).strip()

    # Anchor phrases that usually precede the actual task
    lead_patterns = [
        r"(?i)^.*?\b(?:remind me to|remember to|don'?t forget to|note to self to)\b\s*",
        r"(?i)^.*?\b(?:reminder\s*:\s*)",
        r"(?i)^\s*please\s+",
        r"(?i)^\s*can you\s+",
        r"(?i)^\s*kindly\s+",
    ]
    for pat in lead_patterns:
        s_ = re.sub(pat, "", s_, count=1)

    # If we still have “remind me/remember to/don't forget to” without the 'to', handle that:
    s_ = re.sub(r"(?i)\b(remind me|remember|don'?t forget)\b\s*", "", s_, count=1)

    # Remove obvious time prepositions when followed by a time phrase
    for ch in time_chunks:
        # include common preps near the chunk
        s_ = re.sub(rf"(?i)\b(on|at|by|before|after|this|next|coming)\s+{re.escape(ch)}", " ", s_)
        s_ = s_.replace(ch, " ")

    # If sentence starts with connectors like "to ", keep the verb phrase
    s_ = re.sub(r"(?i)^\s*(?:to\s+)", "", s_)

    # Trim filler pronouns/articles repeatedly
    s_ = re.sub(r"(?i)\b(my|the|a|an)\b\s+", " ", s_)
    s_ = norm(s_)
    # Keep it short
    if len(s_) > 100:
        s_ = s_[:97].rstrip() + "…"
    # Capitalize first letter
    if s_:
        s_ = s_[0].upper() + s_[1:]
    return s_

def tidy_when(chunks: List[str]) -> Optional[str]:
    if not chunks:
        return None
    pretty = " ".join(chunks)
    # Try to parse a single clear timestamp to a cleaner form, else keep text
    if dtparse and len(chunks) == 1:
        try:
            dt = dtparse.parse(chunks[0], fuzzy=True, dayfirst=True)  # dayfirst handles 10/12 vs 12/10
            # Show readable form without timezone assumptions
            pretty = dt.strftime("%a, %d %b %Y %I:%M %p").replace(" 12:00 AM", "")
        except Exception:
            pass
    return pretty

def extract_reminder_lines(text: str) -> List[str]:
    lines: List[str] = []
    seen = set()

    for s in split_sentences(text):
        if not looks_like_reminder(s):
            continue
        time_chunks = find_time_phrases(s)
        what = squeeze_task_text(s, time_chunks)
        when = tidy_when(time_chunks)
        if not what:  # fallback to original if extraction failed
            what = norm(s)
        short = f"{what} — {when} " if when else what
        key = short.lower()
        if key not in seen:
            seen.add(key)
            lines.append(short)

    # Also scan raw bullet lines that might not end with punctuation
    for raw in text.splitlines():
        if BULLET.search(raw):
            s = norm(BULLET.sub("", raw))
            if not s:
                continue
            time_chunks = find_time_phrases(s)
            what = squeeze_task_text(s, time_chunks)
            when = tidy_when(time_chunks)
            short = f"{what} — {when} " if when else what
            key = short.lower()
            if key not in seen:
                seen.add(key)
                lines.append(short)

    print("\n Reminder:")
    for idx, line in enumerate(lines, 1):
        print(f"{idx}. {line}")
    return lines





      



















# ------------------------------ Chunking ------------------------------ #

def chunk_segments(
    segments: List[Segment],
    max_chars: int = 1200,
    overlap_chars: int = 200
) -> List[Chunk]:
    """
    Merge consecutive segments into text windows ~max_chars with controlled overlap.
    """
    chunks: List[Chunk] = []
    buf = []
    buf_len = 0
    buf_start = None
    buf_indices = []

    def flush():
        nonlocal chunks, buf, buf_len, buf_start, buf_indices
        if not buf:
            return
        text = normalize_space(" ".join(buf))
        start = buf_start
        end = current_end if buf_indices else start
        chunks.append(Chunk(start=start, end=end, text=text, seg_indices=buf_indices.copy()))
        buf, buf_len, buf_start, buf_indices = [], 0, None, []

    current_end = 0.0
    for i, seg in enumerate(segments):
        seg_text = seg.text
        seg_len = len(seg_text)
        if buf_len == 0:
            buf_start = seg.start
        if buf_len + seg_len <= max_chars:
            buf.append(seg_text)
            buf_len += seg_len + 1
            buf_indices.append(i)
            current_end = seg.end
        else:
            flush()
            # Overlap: carry last overlap_chars chars into new buffer start
            if chunks:
                carry_text = chunks[-1].text[-overlap_chars:]
                # Re-anchor carry within new chunk without timestamps (just for text continuity)
                buf.append(carry_text)
                buf_len = len(carry_text)
                buf_start = seg.start  # new chunk timestamp starts at current seg
            else:
                buf_len = 0
                buf_start = seg.start
            buf.append(seg_text)
            buf_len += seg_len + 1
            buf_indices = [i]
            current_end = seg.end

    flush()
    return chunks


# ------------------------------ Retrieval ------------------------------ #

class Retriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        self.device = device_auto() if device is None else device
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return embs

    def top_k(self, query: str, chunks: List[Chunk], k: int = 5) -> List[Tuple[int, float]]:
        chunk_texts = [c.text for c in chunks]
        q = self.encode([query])  # (1, d)
        c = self.encode(chunk_texts)  # (n, d)
        sims = (c @ q.T).squeeze(-1)  # cosine sims since normalized
        top_idx = np.argsort(-sims)[:max(1, min(k, len(chunks)))]
        return [(int(i), float(sims[i])) for i in top_idx]


# ------------------------------ Readers (Extractive + Generative) ------------------------------ #

class QAReader:
    def __init__(
        self,
        extractive_model: str = "deepset/roberta-base-squad2",
        generative_model: str = "google/flan-t5-large",
        device: Optional[str] = None
    ):
        self.device = device_auto() if device is None else device

        # Extractive pipeline (fast and precise when answer span exists)
        self.qa = pipeline(
            "question-answering",
            model=extractive_model,
            device=0 if self.device == "cuda" else -1
        )

        # Generative fallback (handles synthesis across multiple chunks)
        self.gen_tokenizer = AutoTokenizer.from_pretrained(generative_model)
        self.gen_model = AutoModelForSeq2SeqLM.from_pretrained(generative_model)
        if self.device != "cpu":
            self.gen_model = self.gen_model.to(self.device)

    def extractive_best(
        self,
        question: str,
        candidates: List[Tuple[Chunk, float]]
    ) -> Optional[Dict]:
        """
        Try extractive QA over top chunks; return best if confident.
        """
        best = None
        for chunk, sim in candidates:
            try:
                out = self.qa(question=question, context=chunk.text)
                score = float(out.get("score", 0.0))
                answer = normalize_space(out.get("answer", ""))
                if not answer:
                    continue
                entry = {
                    "answer": answer,
                    "score": score,
                    "start_char": int(out.get("start", -1)),
                    "end_char": int(out.get("end", -1)),
                    "chunk": chunk,
                    "retrieval_sim": sim
                }
                if best is None or entry["score"] > best["score"]:
                    best = entry
            except Exception:
                continue
        # Heuristic threshold: accept only confident spans
        if best and best["score"] >= 0.35:
            return best
        return None

    def generative_synthesize(
        self,
        question: str,
        candidates: List[Tuple[Chunk, float]],
        max_context_chars: int = 3000,
        max_new_tokens: int = 256
    ) -> Dict:
        """
        Build a compact prompt from top chunks and synthesize an answer.
        """
        # Concatenate contexts until limit
        acc = []
        used = 0
        sources = []
        for chunk, sim in candidates:
            t = f"[{human_time(chunk.start)}–{human_time(chunk.end)}] {chunk.text}"
            if used + len(t) > max_context_chars and acc:
                break
            acc.append(t)
            used += len(t)
            sources.append((chunk.start, chunk.end))
        context = "\n\n".join(acc)

        prompt = (
            "You are an expert conversation analyst. Answer the question ONLY using the context from the transcript.\n"
            "If you are not sure, provide the best supported answer from the given excerpts. Be concise and precise.\n\n"
            f"Question: {question}\n\n"
            f"Transcript excerpts:\n{context}\n\n"
            "Answer:"
        )

        inputs = self.gen_tokenizer([prompt], return_tensors="pt", truncation=True, max_length=4096)
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            gen = self.gen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                #  temperature=0.2,
                #  top_p=0.95
              
            )
        ans = self.gen_tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        ans = ans.split("Answer:", 1)[-1].strip()
        return {"answer": ans, "score": None, "chunk": None, "sources": sources}


# ------------------------------ Orchestration ------------------------------ #

def map_span_to_timestamp(span_text: str, chunk: Chunk, segments: List[Segment]) -> Tuple[float, float]:
    """
    Approximate timestamps for an extractive span by locating it inside the chunk's segments.
    """
    # Simple heuristic: find the first segment inside chunk that contains a substantial part of the span.
    span = span_text.lower().strip()
    for idx in chunk.seg_indices:
        seg = segments[idx]
        txt = seg.text.lower()
        # Consider partial containment to be more robust
        if span in txt or any(w in txt for w in span.split()[:3]):
            return seg.start, seg.end
    # Fallback to chunk window
    return chunk.start, chunk.end


def answer_from_audio(
    audio_path: str,
    question: str,
    whisper_model: str = "small",
    language: Optional[str] = None,
    retriever_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    extractive_model: str = "deepset/roberta-base-squad2",
    generative_model: str = "google/flan-t5-large",
    top_k: int = 5
) -> Dict:
    # 1) ASR
    segments = transcribe_audio(audio_path, whisper_model=whisper_model, language=language)
  


    if not segments:
        return {"answer": "(No transcript produced.)", "sources": []}

    # 2) Chunking
    chunks = chunk_segments(segments, max_chars=1200, overlap_chars=200)

    # 3) Retrieval
    retriever = Retriever(model_name=retriever_name)
    ranked = retriever.top_k(question, chunks, k=top_k)
    candidates = [(chunks[i], sim) for i, sim in ranked]

    # 4) Readers
    reader = QAReader(extractive_model=extractive_model, generative_model=generative_model)

    # Try extractive first
    ext = reader.extractive_best(question, candidates)

    if ext:
        start_ts, end_ts = map_span_to_timestamp(ext["answer"], ext["chunk"], segments)
        return {
            "answer": ext["answer"],
            "confidence": round(float(ext["score"]), 4),
            "sources": [{
                "window": [human_time(ext["chunk"].start), human_time(ext["chunk"].end)],
                "precise_span": [human_time(start_ts), human_time(end_ts)]
            }]
        }

    # Fallback to generative synthesis
    gen = reader.generative_synthesize(question, candidates)
    return {
        "answer": gen["answer"],
        "confidence": None,
        "sources": [{
            "window": [human_time(s), human_time(e)]
        } for (s, e) in gen["sources"]]
    }


# ------------------------------ CLI ------------------------------ #

def parse_args():
    # Create a command-line parser with a short description.
    p = argparse.ArgumentParser(description="Ask questions about an audio conversation.")

    # File to analyze (wav, mp3, m4a, flac, etc.)
    p.add_argument("-a", "--audio", help="Path to audio file (wav, mp3, m4a, flac, etc.)")

    # The question you want to ask about that conversation.
    p.add_argument("-q", "--question", help="Your question about the conversation.")

    # Whisper ASR (speech-to-text) model size.
    p.add_argument("--whisper_model", default="small",
                   help="Whisper model size: tiny|base|small|medium|large-v2|large-v3 (default: small)")

    # Force a specific language (skip auto-detect), e.g., 'en' for English.
    p.add_argument("--language", default=None, help="Force language code (e.g., 'en'); otherwise auto-detect.")

    # How many relevant chunks to retrieve from the transcript for answering.
    p.add_argument("--top_k", type=int, default=5, help="Top K chunks to retrieve (default: 5)")

    # Model used for extractive QA (pulls exact spans from text).
    p.add_argument("--extractive_model", default="deepset/roberta-base-squad2", help="HF model for extractive QA")

    # Model used as a fallback for generative answers (when extractive isn’t enough).
    p.add_argument("--generative_model", default="google/flan-t5-large", help="HF model for generative fallback")

    # Sentence embedding model name (for retrieval).
    p.add_argument("--retriever", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer name")

    # Parse known flags; ignore anything unknown so it doesn’t crash.
    args, _ = p.parse_known_args()

    # ---------------- Interactive fallback ----------------
    # If user launched without flags (e.g., clicking "Run"), ask for inputs.
    if not args.audio:
        args.audio = input("Path to audio file: ").strip().strip('"')
    if not args.question:
        args.question = input("Your question: ").strip()

    return args


def main():
    # Get all command-line (or prompted) arguments.
    args = parse_args()

    # Call your core pipeline (you must implement `answer_from_audio` elsewhere).
    result = answer_from_audio(
        audio_path=args.audio,
        question=args.question,
        whisper_model=args.whisper_model,
        language=args.language,
        retriever_name=args.retriever,
        extractive_model=args.extractive_model,
        generative_model=args.generative_model,
        top_k=args.top_k
    )

    # ---------------- Output formatting ----------------
    print("\n=== Answer ===")
    # Safely print the answer (or a friendly message if missing).
    print(result.get("answer", "").strip() or "(No answer.)")

    # Print confidence if provided (e.g., 0.0–1.0).
    if result.get("confidence") is not None:
        print(f"\nConfidence: {result['confidence']:.4f}")

    # Show where in the audio/transcript the answer came from.
    print("\n=== from the audio 1) The timestamp that is the source of the text answer. 2) The timestamp in the audio where the answer appears. ===")
    for i, src in enumerate(result.get("sources", []), 1):
        # Each source may have a larger 'window' and an optional more precise span.
        win = src.get("window", ["?", "?"])
        if "precise_span" in src:
            ps = src["precise_span"]
            print(f"{i}")
            print(f"{i} {win[0]} → {win[1]}   |   Span {ps[0]} → {ps[1]}")
        else:
            print(f"{i} {win[0]} → {win[1]}")
    print()


if __name__ == "__main__":
    # Run the CLI when executed directly.
    main()
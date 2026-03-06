"""
MIA Worker B — Whisper ASR + Voice Emotion
===========================================
Run this on Laptop B:
    pip install -r requirements_worker_b.txt
    python workers/worker_asr_voice.py

Exposes:
    POST /transcribe       — audio (base64) → transcribed text
    POST /voice_emotion    — audio (base64) → emotion result
    GET  /health           — health check
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import base64
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from app.asr_pipeline import WhisperTranscriber
from app.voice_emotion import VoiceEmotionClassifier

app = FastAPI(title="MIA Worker B — ASR + Voice Emotion")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

print("Loading Whisper Transcriber...")
transcriber = WhisperTranscriber()
print("Loading Voice Emotion Classifier...")
voice_classifier = VoiceEmotionClassifier()
print("✅ Worker B ready.")


class AudioPayload(BaseModel):
    audio_b64: str          # base64-encoded float32 numpy array
    sample_rate: int = 16000
    dtype: str = "float32"  # "float32" or "int16"


def decode_audio(payload: AudioPayload) -> np.ndarray:
    """Decode base64 audio back to numpy array."""
    raw_bytes = base64.b64decode(payload.audio_b64)
    dtype = np.float32 if payload.dtype == "float32" else np.int16
    return np.frombuffer(raw_bytes, dtype=dtype)


@app.get("/health")
def health():
    return {"status": "ok", "worker": "B", "services": ["asr", "voice_emotion"]}


@app.post("/transcribe")
def transcribe(payload: AudioPayload):
    """
    Accepts base64-encoded audio, returns transcription.
    """
    try:
        audio = decode_audio(payload)
        result = transcriber.transcribe(audio, sample_rate=payload.sample_rate)
        return {"success": True, "text": result.get("text", "")}
    except Exception as e:
        return {"success": False, "text": "", "error": str(e)}


@app.post("/voice_emotion")
def voice_emotion(payload: AudioPayload):
    """
    Accepts base64-encoded audio, returns voice emotion classification.
    """
    try:
        audio = decode_audio(payload)
        result = voice_classifier.classify(audio, sample_rate=payload.sample_rate)
        return {"success": True, **result}
    except Exception as e:
        return {
            "success": False,
            "emotion": "neutral", "confidence": 0.0, "all_scores": [],
            "error": str(e)
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

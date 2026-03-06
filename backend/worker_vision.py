"""
MIA Worker C — Vision / FER Emotion
=====================================
Run this on Laptop C:
    pip install -r requirements_worker_c.txt
    python workers/worker_vision.py

Exposes:
    POST /vision_emotion   — JPEG frame (base64) → emotion + head pose + engagement
    GET  /health           — health check
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import base64
import numpy as np
import cv2
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from app.vision_pipeline import VisionEmotionClassifier

app = FastAPI(title="MIA Worker C — Vision Emotion")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

print("Loading Vision Emotion Classifier (FER)...")
vision_classifier = VisionEmotionClassifier()
print("✅ Worker C ready.")


class FramePayload(BaseModel):
    frame_b64: str   # base64-encoded JPEG image bytes


def decode_frame(payload: FramePayload) -> np.ndarray:
    """Decode base64 JPEG back to BGR numpy array."""
    jpg_bytes = base64.b64decode(payload.frame_b64)
    jpg_array = np.frombuffer(jpg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)
    return frame


@app.get("/health")
def health():
    return {"status": "ok", "worker": "C", "services": ["vision_emotion"]}


@app.post("/vision_emotion")
def vision_emotion(payload: FramePayload):
    """
    Accepts a base64-encoded JPEG frame, returns vision emotion analysis.
    """
    try:
        frame = decode_frame(payload)
        if frame is None:
            raise ValueError("Could not decode frame")

        result = vision_classifier.process_frame(frame, draw_landmarks=False)

        return {
            "success": True,
            "emotion": result["emotion"],
            "confidence": result["confidence"],
            "all_scores": result["all_scores"],
            "head_pose": result["head_pose"],
            "gaze": result["gaze"],
            "engagement": result["engagement"],
        }
    except Exception as e:
        return {
            "success": False,
            "emotion": "neutral", "confidence": 0.0, "all_scores": [],
            "head_pose": {"yaw": 0, "pitch": 0, "roll": 0},
            "gaze": "unknown", "engagement": 0.5,
            "error": str(e)
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)

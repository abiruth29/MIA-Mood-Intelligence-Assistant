"""
MIA Worker D — Text Emotion + LLM Response
===========================================
Run this on Laptop D:
    pip install -r requirements_worker_d.txt
    python workers/worker_llm_text.py

Exposes:
    POST /text_emotion     — text → emotion classification
    POST /llm_response     — text + emotion + context → MIA response
    GET  /health           — health check
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import asyncio

from app.text_emotion import TextEmotionClassifier
from app.llm_response import LLMResponseGenerator

app = FastAPI(title="MIA Worker D — Text Emotion + LLM")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

print("Loading Text Emotion Classifier...")
text_classifier = TextEmotionClassifier()
print("Loading LLM Response Generator...")
llm_generator = LLMResponseGenerator(provider="ollama", model="llama3.2")
print("✅ Worker D ready.")


class TextPayload(BaseModel):
    text: str


class LLMPayload(BaseModel):
    text: str
    emotion: str = "neutral"
    confidence: float = 0.5
    context: Optional[Dict[str, Any]] = None


@app.get("/health")
def health():
    return {"status": "ok", "worker": "D", "services": ["text_emotion", "llm_response"]}


@app.post("/text_emotion")
def text_emotion(payload: TextPayload):
    """
    Accepts a text string, returns text emotion classification.
    """
    try:
        result = text_classifier.classify(payload.text)
        return {"success": True, **result}
    except Exception as e:
        return {
            "success": False,
            "emotion": "neutral", "confidence": 0.0, "all_scores": [],
            "error": str(e)
        }


@app.post("/llm_response")
async def llm_response(payload: LLMPayload):
    """
    Accepts text + emotion context, returns MIA's empathetic response.
    """
    try:
        result = await llm_generator.generate_response(
            user_text=payload.text,
            emotion=payload.emotion,
            confidence=payload.confidence,
            context=payload.context or {}
        )
        return {"success": True, **result}
    except Exception as e:
        return {
            "success": False,
            "response": "",
            "emotion_acknowledged": False,
            "suggestions": [],
            "error": str(e)
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)

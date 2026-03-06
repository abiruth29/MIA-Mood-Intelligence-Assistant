"""
MIA — Distributed Main Backend (Laptop A)
==========================================
This is the coordinator. It runs on YOUR laptop and:
  - Captures audio/video locally
  - Sends audio to Worker B (ASR + Voice Emotion)
  - Sends video frames to Worker C (Vision Emotion)
  - Sends text to Worker D (Text Emotion + LLM)
  - Fuses all results and streams to the React frontend

Run:
    uvicorn main_distributed:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import uuid
import base64
import httpx
from collections import deque
from app.media_capture import VideoCapture, AudioCapture
from app.tts_engine import TTSEngine
from app.database import MIADatabase
from worker_config import WORKER_B_URL, WORKER_C_URL, WORKER_D_URL, WORKER_TIMEOUT

app = FastAPI(title="MIA Distributed Coordinator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Local components (Laptop A only) ─────────────────────────────────────────
video_capture = VideoCapture()
audio_capture = AudioCapture()

print("Initializing TTS Engine...")
tts_engine = TTSEngine()

print("Initializing Database...")
database = MIADatabase()

print(f"✅ Coordinator ready.")
print(f"   Worker B (ASR+Voice): {WORKER_B_URL}")
print(f"   Worker C (Vision):    {WORKER_C_URL}")
print(f"   Worker D (LLM+Text):  {WORKER_D_URL}")

# ── Fusion weights ────────────────────────────────────────────────────────────
AUDIO_WEIGHT = 0.3
TEXT_WEIGHT  = 0.3
VIDEO_WEIGHT = 0.4

ENABLE_LLM = True
ENABLE_TTS = True

# ── Remote worker helpers ─────────────────────────────────────────────────────

async def call_worker(client: httpx.AsyncClient, url: str, payload: dict, fallback: dict) -> dict:
    """
    POST to a remote worker. Returns fallback dict on any error so the
    main pipeline never crashes if a worker is down.
    """
    try:
        resp = await client.post(url, json=payload, timeout=WORKER_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"⚠️  Worker call failed [{url}]: {e}")
        return fallback


def encode_audio(audio_data) -> dict:
    """Encode numpy audio array to base64 payload for worker."""
    import numpy as np
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype("float32") / 32768.0
    raw_bytes = audio_data.astype("float32").tobytes()
    return {
        "audio_b64": base64.b64encode(raw_bytes).decode(),
        "sample_rate": 16000,
        "dtype": "float32"
    }


def normalize_label(label: str) -> str:
    label = label.lower()
    if label == "joy":    return "happiness"
    if label == "angry":  return "anger"
    if label == "sad":    return "sadness"
    if label == "happy":  return "happiness"
    return label


# ── REST analytics endpoints (unchanged) ──────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "MIA Distributed Coordinator is running"}

@app.get("/api/workers/health")
async def check_workers():
    """Ping all workers and return their health status."""
    results = {}
    async with httpx.AsyncClient() as client:
        for name, base_url in [("B_asr_voice", WORKER_B_URL),
                                ("C_vision",    WORKER_C_URL),
                                ("D_llm_text",  WORKER_D_URL)]:
            try:
                r = await client.get(f"{base_url}/health", timeout=3.0)
                results[name] = r.json()
            except Exception as e:
                results[name] = {"status": "unreachable", "error": str(e)}
    return results

@app.get("/api/analytics/emotions")
async def get_emotion_analytics(days: int = 7):
    return database.get_emotion_distribution(days=days)

@app.get("/api/analytics/sessions")
async def get_sessions(days: int = 30):
    return database.get_session_summary(days=days)

@app.get("/api/analytics/engagement")
async def get_engagement_stats(days: int = 7):
    return database.get_engagement_stats(days=days)

@app.get("/api/analytics/daily")
async def get_daily_analytics(date: str = None):
    return database.get_daily_summary(date)

@app.get("/api/analytics/daily/{date}")
async def get_daily_analytics_by_date(date: str):
    return database.get_daily_summary(date)

@app.get("/api/preferences")
async def get_preferences():
    return database.get_all_preferences()

@app.post("/api/preferences/{key}")
async def set_preference(key: str, value: str):
    database.set_preference(key, value)
    return {"status": "ok"}


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    session_id = str(uuid.uuid4())[:8]
    database.create_session(session_id, {"type": "distributed_websocket"})
    print(f"[{session_id}] Session started")

    # Shared httpx client for all worker calls in this session
    http_client = httpx.AsyncClient()

    # Temporal smoothing buffer (5-second window at ~2 Hz = 10 samples)
    emotion_history: deque = deque(maxlen=10)

    # ── Task: receive start/stop commands ─────────────────────────────────────
    async def receive_commands():
        try:
            while True:
                data = await websocket.receive_text()
                if data == "start_camera":
                    video_capture.start()
                    audio_capture.start()
                    await websocket.send_text(json.dumps({"status": "camera_started"}))
                elif data == "stop_camera":
                    video_capture.stop()
                    audio_capture.stop()
                    await websocket.send_text(json.dumps({"status": "camera_stopped"}))
        except WebSocketDisconnect:
            print(f"[{session_id}] Client disconnected (receiver)")
        except Exception as e:
            print(f"[{session_id}] Receiver error: {e}")

    # ── Task: stream video frames to frontend ─────────────────────────────────
    async def send_frames():
        try:
            while True:
                if video_capture.running:
                    frame_b64 = video_capture.get_jpeg_frame()
                    if frame_b64:
                        await websocket.send_text(json.dumps({
                            "type": "video_frame", "data": frame_b64
                        }))
                    await asyncio.sleep(0.033)  # ~30 FPS
                else:
                    await asyncio.sleep(0.1)
        except WebSocketDisconnect:
            print(f"[{session_id}] Client disconnected (sender)")
        except Exception as e:
            print(f"[{session_id}] Sender error: {e}")

    # ── Task: main processing loop ────────────────────────────────────────────
    async def process_audio():
        """
        Every 2 seconds:
          1. Grab audio + video frame
          2. Fan-out to all 3 workers IN PARALLEL
          3. Fuse results with temporal smoothing
          4. Push LLM + TTS back to frontend
        """
        try:
            while True:
                if not audio_capture.running:
                    await asyncio.sleep(0.5)
                    continue

                await asyncio.sleep(2.0)
                audio_data  = audio_capture.get_buffer()
                current_frame = video_capture.get_raw_frame()

                if audio_data is None or audio_data.size == 0:
                    continue

                # Encode audio once for both B workers
                audio_payload = encode_audio(audio_data)

                # Encode video frame for Worker C
                frame_payload = None
                if current_frame is not None and current_frame.size > 0:
                    import cv2
                    _, jpg_buf = cv2.imencode(".jpg", current_frame)
                    frame_payload = {
                        "frame_b64": base64.b64encode(jpg_buf.tobytes()).decode()
                    }

                # ── Fan-out: call all workers in parallel ──────────────────
                fallback_emotion = {"emotion": "neutral", "confidence": 0.0, "all_scores": []}
                fallback_vision  = {**fallback_emotion,
                                    "head_pose": {"yaw": 0, "pitch": 0, "roll": 0},
                                    "gaze": "unknown", "engagement": 0.5}
                fallback_text    = {**fallback_emotion}
                fallback_asr     = {"text": ""}

                tasks = [
                    call_worker(http_client,
                                f"{WORKER_B_URL}/transcribe",
                                audio_payload, fallback_asr),
                    call_worker(http_client,
                                f"{WORKER_B_URL}/voice_emotion",
                                audio_payload, fallback_emotion),
                    call_worker(http_client,
                                f"{WORKER_C_URL}/vision_emotion",
                                frame_payload or {}, fallback_vision)
                    if frame_payload else asyncio.coroutine(lambda: fallback_vision)(),
                ]

                asr_result, audio_emotion_result, video_emotion_result = await asyncio.gather(*tasks)

                text = asr_result.get("text", "").strip()

                # Send transcription to frontend immediately
                if text:
                    print(f"[{session_id}] Transcribed: {text}")
                    await websocket.send_text(json.dumps({"type": "transcription", "text": text}))

                # Text emotion (depends on transcription result)
                if text:
                    text_emotion_result = await call_worker(
                        http_client,
                        f"{WORKER_D_URL}/text_emotion",
                        {"text": text},
                        fallback_text
                    )
                else:
                    text_emotion_result = fallback_text

                # ── Tri-modal fusion ───────────────────────────────────────
                has_text  = bool(text)
                has_video = video_emotion_result.get("confidence", 0.0) > 0

                if has_text and has_video:
                    audio_w, text_w, video_w = AUDIO_WEIGHT, TEXT_WEIGHT, VIDEO_WEIGHT
                elif has_text:
                    audio_w, text_w, video_w = 0.4, 0.6, 0.0
                elif has_video:
                    audio_w, text_w, video_w = 0.35, 0.0, 0.65
                else:
                    audio_w, text_w, video_w = 1.0, 0.0, 0.0

                combined_scores = {}

                for item in audio_emotion_result.get("all_scores", []):
                    lbl   = normalize_label(item["label"])
                    score = float(item.get("score") or 0.0)
                    combined_scores[lbl] = combined_scores.get(lbl, 0.0) + score * audio_w

                if has_text:
                    for item in text_emotion_result.get("all_scores", []):
                        lbl   = normalize_label(item["label"])
                        score = float(item.get("score") or 0.0)
                        combined_scores[lbl] = combined_scores.get(lbl, 0.0) + score * text_w

                if has_video:
                    for item in video_emotion_result.get("all_scores", []):
                        lbl   = normalize_label(item["label"])
                        score = float(item.get("score") or 0.0)
                        combined_scores[lbl] = combined_scores.get(lbl, 0.0) + score * video_w

                # Temporal smoothing
                if combined_scores:
                    emotion_history.append(combined_scores)
                    smoothed_scores = {}
                    for hist in emotion_history:
                        for lbl, score in hist.items():
                            smoothed_scores[lbl] = smoothed_scores.get(lbl, 0.0) + score
                    for lbl in smoothed_scores:
                        smoothed_scores[lbl] /= len(emotion_history)

                    final_emotion    = max(smoothed_scores, key=smoothed_scores.get)
                    final_confidence = smoothed_scores[final_emotion]
                else:
                    final_emotion    = audio_emotion_result.get("emotion", "neutral")
                    final_confidence = audio_emotion_result.get("confidence", 0.0)
                    smoothed_scores  = {}

                head_pose  = video_emotion_result.get("head_pose", {"yaw": 0, "pitch": 0, "roll": 0})
                gaze       = video_emotion_result.get("gaze", "unknown")
                engagement = video_emotion_result.get("engagement", 0.5)

                sources = f"A:{audio_emotion_result.get('emotion','?')}"
                if has_text:  sources += f" T:{text_emotion_result.get('emotion','?')}"
                if has_video: sources += f" V:{video_emotion_result.get('emotion','?')}"
                print(f"[{session_id}] Tri-Modal: {final_emotion} ({final_confidence:.3f}) [{sources}]")

                # Send fused emotion to frontend
                await websocket.send_text(json.dumps({
                    "type": "emotion",
                    "emotion": final_emotion,
                    "confidence": round(final_confidence, 3),
                    "all_scores": [{"label": k, "score": round(v, 3)}
                                   for k, v in smoothed_scores.items()],
                    "modalities": {
                        "audio": audio_emotion_result.get("emotion"),
                        "text":  text_emotion_result.get("emotion") if has_text else None,
                        "video": video_emotion_result.get("emotion") if has_video else None,
                    },
                    "head_pose":  head_pose,
                    "gaze":       gaze,
                    "engagement": round(engagement, 2),
                }))

                # Log to DB
                try:
                    database.log_emotion_event(
                        session_id=session_id,
                        emotion=final_emotion, confidence=final_confidence,
                        audio_emotion=audio_emotion_result.get("emotion"),
                        text_emotion=text_emotion_result.get("emotion") if has_text else None,
                        video_emotion=video_emotion_result.get("emotion") if has_video else None,
                        engagement=engagement,
                    )
                except Exception as db_err:
                    print(f"[{session_id}] DB log error: {db_err}")

                # LLM response (Worker D)
                if ENABLE_LLM and has_text and len(text) > 3:
                    llm_result = await call_worker(
                        http_client,
                        f"{WORKER_D_URL}/llm_response",
                        {
                            "text": text,
                            "emotion": final_emotion,
                            "confidence": round(final_confidence, 3),
                            "context": {"engagement": engagement, "gaze": gaze},
                        },
                        {"response": "", "suggestions": []}
                    )

                    assistant_response = llm_result.get("response", "")
                    if assistant_response:
                        print(f"[{session_id}] MIA: {assistant_response[:80]}...")

                        await websocket.send_text(json.dumps({
                            "type": "llm_response",
                            "response": assistant_response,
                            "suggestions": llm_result.get("suggestions", []),
                        }))

                        # Save conversation
                        try:
                            database.save_conversation(
                                session_id=session_id,
                                user_text=text,
                                assistant_response=assistant_response,
                                emotion=final_emotion,
                                confidence=final_confidence,
                                modalities={
                                    "audio": audio_emotion_result.get("emotion"),
                                    "text":  text_emotion_result.get("emotion") if has_text else None,
                                    "video": video_emotion_result.get("emotion") if has_video else None,
                                },
                                engagement=engagement,
                                head_pose=head_pose,
                            )
                        except Exception as db_err:
                            print(f"[{session_id}] DB save error: {db_err}")

                        # TTS (local on Laptop A)
                        if ENABLE_TTS:
                            try:
                                tts_result = await tts_engine.synthesize(
                                    text=assistant_response,
                                    emotion=final_emotion,
                                    return_format="base64"
                                )
                                if tts_result.get("audio"):
                                    await websocket.send_text(json.dumps({
                                        "type": "tts_audio",
                                        "audio": tts_result["audio"],
                                        "format": tts_result.get("format", "mp3"),
                                        "voice": tts_result.get("voice", ""),
                                    }))
                            except Exception as tts_err:
                                print(f"[{session_id}] TTS error: {tts_err}")

        except WebSocketDisconnect:
            print(f"[{session_id}] Client disconnected (processor)")
        except Exception as e:
            import traceback
            print(f"[{session_id}] Processing error: {e}")
            traceback.print_exc()

    # ── Run all tasks concurrently ────────────────────────────────────────────
    try:
        done, pending = await asyncio.wait(
            [
                asyncio.create_task(receive_commands()),
                asyncio.create_task(send_frames()),
                asyncio.create_task(process_audio()),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
    except Exception as e:
        print(f"[{session_id}] WebSocket handler error: {e}")
    finally:
        video_capture.stop()
        audio_capture.stop()
        await http_client.aclose()
        try:
            database.end_session(session_id)
            print(f"[{session_id}] Session ended")
        except Exception as e:
            print(f"[{session_id}] Session end error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

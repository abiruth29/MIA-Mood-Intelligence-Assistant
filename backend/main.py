from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import uuid
from collections import deque
from app.media_capture import VideoCapture, AudioCapture
from app.asr_pipeline import WhisperTranscriber
from app.voice_emotion import VoiceEmotionClassifier
from app.text_emotion import TextEmotionClassifier
from app.vision_pipeline import VisionEmotionClassifier
from app.llm_response import LLMResponseGenerator
from app.tts_engine import TTSEngine
from app.database import MIADatabase

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

video_capture = VideoCapture()
audio_capture = AudioCapture()

# Initialize all components
print("Initializing Whisper Transcriber...")
transcriber = WhisperTranscriber()
print("Whisper Transcriber ready.")

print("Initializing Voice Emotion Classifier...")
emotion_classifier = VoiceEmotionClassifier()
print("Voice Emotion Classifier ready.")

print("Initializing Text Emotion Classifier...")
text_emotion_classifier = TextEmotionClassifier()
print("Text Emotion Classifier ready.")

print("Initializing Vision Emotion Classifier...")
vision_classifier = VisionEmotionClassifier()
print("Vision Emotion Classifier ready.")

print("Initializing LLM Response Generator...")
llm_generator = LLMResponseGenerator(provider="ollama", model="llama3.2")
print("LLM Response Generator ready.")

print("Initializing TTS Engine...")
tts_engine = TTSEngine()
print("TTS Engine ready.")

print("Initializing Database...")
database = MIADatabase()
print("Database ready.")

# Temporal smoothing buffer (5-second window at ~2Hz = 10 samples)
emotion_history = deque(maxlen=10)

# Tri-modal fusion weights
AUDIO_WEIGHT = 0.3
TEXT_WEIGHT = 0.3
VIDEO_WEIGHT = 0.4

# LLM response settings
ENABLE_LLM = True  # Set to False to disable LLM responses
ENABLE_TTS = True  # Set to False to disable text-to-speech

@app.get("/")
async def root():
    return {"message": "MIA Backend is running"}

# REST API endpoints for analytics
@app.get("/api/analytics/emotions")
async def get_emotion_analytics(days: int = 7):
    """Get emotion distribution analytics."""
    return database.get_emotion_distribution(days=days)

@app.get("/api/analytics/sessions")
async def get_sessions(days: int = 30):
    """Get session summaries."""
    return database.get_session_summary(days=days)

@app.get("/api/analytics/engagement")
async def get_engagement_stats(days: int = 7):
    """Get engagement statistics."""
    return database.get_engagement_stats(days=days)

@app.get("/api/analytics/daily")
async def get_daily_analytics(date: str = None):
    """Get daily analytics summary."""
    return database.get_daily_summary(date)

@app.get("/api/analytics/daily/{date}")
async def get_daily_analytics_by_date(date: str):
    """Get daily analytics summary for a specific date."""
    return database.get_daily_summary(date)

@app.get("/api/preferences")
async def get_preferences():
    """Get all user preferences."""
    return database.get_all_preferences()

@app.post("/api/preferences/{key}")
async def set_preference(key: str, value: str):
    """Set a user preference."""
    database.set_preference(key, value)
    return {"status": "ok"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Create a unique session ID for this connection
    session_id = str(uuid.uuid4())[:8]
    database.create_session(session_id, {"type": "websocket"})
    print(f"New session started: {session_id}")
    
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
            print("Client disconnected (receiver)")
        except Exception as e:
            print(f"Receiver error: {e}")

    async def send_frames():
        try:
            while True:
                if video_capture.running:
                    frame_base64 = video_capture.get_jpeg_frame()
                    if frame_base64:
                        await websocket.send_text(json.dumps({"type": "video_frame", "data": frame_base64}))
                    await asyncio.sleep(0.033) # ~30 FPS
                else:
                    await asyncio.sleep(0.1)
        except WebSocketDisconnect:
            print("Client disconnected (sender)")
        except Exception as e:
            print(f"Sender error: {e}")

    async def process_audio():
        """
        Tri-modal emotion processing loop.
        Combines: Audio (30%) + Text (30%) + Video (40%)
        With 5-second temporal smoothing.
        """
        try:
            while True:
                if audio_capture.running:
                    # Process every 2 seconds
                    await asyncio.sleep(2.0)
                    audio_data = audio_capture.get_buffer()
                    
                    # Get current video frame for vision analysis
                    current_frame = video_capture.get_raw_frame()
                    
                    if audio_data is not None and audio_data.size > 0:
                        # 1. Transcription
                        result = await asyncio.to_thread(transcriber.transcribe, audio_data)
                        text = result.get("text", "").strip()
                        
                        text_emotion_result = {"emotion": "neutral", "confidence": 0.0, "all_scores": []}
                        if text:
                            print(f"Transcribed: {text}")
                            await websocket.send_text(json.dumps({"type": "transcription", "text": text}))
                            
                            # 2. Text Emotion
                            text_emotion_result = await asyncio.to_thread(text_emotion_classifier.classify, text)

                        # 3. Audio Emotion
                        audio_emotion_result = await asyncio.to_thread(emotion_classifier.classify, audio_data)
                        
                        # 4. Video Emotion (from current frame)
                        video_emotion_result = {"emotion": "neutral", "confidence": 0.0, "all_scores": []}
                        head_pose = {"yaw": 0, "pitch": 0, "roll": 0}
                        gaze = "unknown"
                        engagement = 0.5
                        
                        if current_frame is not None and hasattr(current_frame, 'size') and current_frame.size > 0:
                            vision_result = await asyncio.to_thread(
                                vision_classifier.process_frame, current_frame, False
                            )
                            video_emotion_result = {
                                "emotion": vision_result["emotion"],
                                "confidence": vision_result["confidence"],
                                "all_scores": vision_result["all_scores"]
                            }
                            head_pose = vision_result["head_pose"]
                            gaze = vision_result["gaze"]
                            engagement = vision_result["engagement"]
                        
                        # 5. Tri-Modal Fusion
                        combined_scores = {}
                        
                        def normalize_label(l):
                            l = l.lower()
                            if l == 'joy': return 'happiness'
                            if l == 'angry': return 'anger'
                            if l == 'sad': return 'sadness'
                            if l == 'happy': return 'happiness'
                            return l
                        
                        # Determine effective weights based on what data is available
                        has_text = bool(text)
                        has_video = video_emotion_result["confidence"] > 0
                        
                        # Dynamic weight adjustment
                        if has_text and has_video:
                            # Full tri-modal: 30% audio, 30% text, 40% video
                            audio_w, text_w, video_w = AUDIO_WEIGHT, TEXT_WEIGHT, VIDEO_WEIGHT
                        elif has_text and not has_video:
                            # Bi-modal (audio + text): 40% audio, 60% text
                            audio_w, text_w, video_w = 0.4, 0.6, 0.0
                        elif has_video and not has_text:
                            # Bi-modal (audio + video): 35% audio, 65% video
                            audio_w, text_w, video_w = 0.35, 0.0, 0.65
                        else:
                            # Audio only
                            audio_w, text_w, video_w = 1.0, 0.0, 0.0

                        # Process Audio Scores
                        for item in audio_emotion_result.get("all_scores", []):
                            lbl = normalize_label(item["label"])
                            score = float(item["score"]) if item["score"] is not None else 0.0
                            combined_scores[lbl] = combined_scores.get(lbl, 0.0) + score * audio_w

                        # Process Text Scores
                        if has_text:
                            for item in text_emotion_result.get("all_scores", []):
                                lbl = normalize_label(item["label"])
                                score = float(item["score"]) if item["score"] is not None else 0.0
                                combined_scores[lbl] = combined_scores.get(lbl, 0.0) + score * text_w
                        
                        # Process Video Scores
                        if has_video:
                            for item in video_emotion_result.get("all_scores", []):
                                lbl = normalize_label(item["label"])
                                score = float(item["score"]) if item["score"] is not None else 0.0
                                combined_scores[lbl] = combined_scores.get(lbl, 0.0) + score * video_w
                        
                        # 6. Temporal Smoothing (5-second sliding window)
                        if combined_scores:
                            emotion_history.append(combined_scores)
                            
                            # Average across history window
                            smoothed_scores = {}
                            for hist_scores in emotion_history:
                                for lbl, score in hist_scores.items():
                                    smoothed_scores[lbl] = smoothed_scores.get(lbl, 0.0) + score
                            
                            # Normalize by history length
                            for lbl in smoothed_scores:
                                smoothed_scores[lbl] /= len(emotion_history)
                            
                            final_emotion = max(smoothed_scores, key=smoothed_scores.get)
                            final_confidence = smoothed_scores[final_emotion]
                            
                            sources = f"A:{audio_emotion_result['emotion']}"
                            if has_text:
                                sources += f" T:{text_emotion_result['emotion']}"
                            if has_video:
                                sources += f" V:{video_emotion_result['emotion']}"
                            
                            print(f"Tri-Modal Emotion: {final_emotion} ({final_confidence:.3f}) [{sources}]")
                            
                            # Send comprehensive result
                            await websocket.send_text(json.dumps({
                                "type": "emotion",
                                "emotion": final_emotion,
                                "confidence": round(final_confidence, 3),
                                "all_scores": [{"label": k, "score": round(v, 3)} for k, v in smoothed_scores.items()],
                                "modalities": {
                                    "audio": audio_emotion_result["emotion"],
                                    "text": text_emotion_result["emotion"] if has_text else None,
                                    "video": video_emotion_result["emotion"] if has_video else None
                                },
                                "head_pose": head_pose,
                                "gaze": gaze,
                                "engagement": round(engagement, 2)
                            }))
                            
                            # Log emotion event to database
                            try:
                                database.log_emotion_event(
                                    session_id=session_id,
                                    emotion=final_emotion,
                                    confidence=final_confidence,
                                    audio_emotion=audio_emotion_result["emotion"],
                                    text_emotion=text_emotion_result["emotion"] if has_text else None,
                                    video_emotion=video_emotion_result["emotion"] if has_video else None,
                                    engagement=engagement
                                )
                            except Exception as db_err:
                                print(f"DB logging error: {db_err}")
                            
                            # 7. Generate LLM Response (if text is available)
                            if ENABLE_LLM and has_text and len(text) > 3:
                                try:
                                    llm_result = await llm_generator.generate_response(
                                        user_text=text,
                                        emotion=final_emotion,
                                        confidence=final_confidence,
                                        context={"engagement": engagement, "gaze": gaze}
                                    )
                                    
                                    if llm_result.get("response"):
                                        assistant_response = llm_result["response"]
                                        print(f"MIA: {assistant_response[:100]}...")
                                        
                                        # Send LLM response to frontend
                                        await websocket.send_text(json.dumps({
                                            "type": "llm_response",
                                            "response": assistant_response,
                                            "suggestions": llm_result.get("suggestions", [])
                                        }))
                                        
                                        # Save conversation to database
                                        try:
                                            database.save_conversation(
                                                session_id=session_id,
                                                user_text=text,
                                                assistant_response=assistant_response,
                                                emotion=final_emotion,
                                                confidence=final_confidence,
                                                modalities={
                                                    "audio": audio_emotion_result["emotion"],
                                                    "text": text_emotion_result["emotion"] if has_text else None,
                                                    "video": video_emotion_result["emotion"] if has_video else None
                                                },
                                                engagement=engagement,
                                                head_pose=head_pose
                                            )
                                        except Exception as db_err:
                                            print(f"DB save error: {db_err}")
                                        
                                        # 8. Generate TTS audio (if enabled)
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
                                                        "voice": tts_result.get("voice", "")
                                                    }))
                                            except Exception as tts_err:
                                                print(f"TTS error: {tts_err}")
                                                
                                except Exception as llm_err:
                                    print(f"LLM error: {llm_err}")
                        else:
                            # Fallback to just audio if nothing else
                            emotion = audio_emotion_result.get("emotion", "neutral")
                            confidence = audio_emotion_result.get("confidence", 0.0)
                            print(f"Audio Emotion (fallback): {emotion} ({confidence})")
                            await websocket.send_text(json.dumps({
                                "type": "emotion",
                                "emotion": emotion,
                                "confidence": confidence,
                                "all_scores": audio_emotion_result.get("all_scores", [])
                            }))

                else:
                    await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            print("Client disconnected (audio)")
        except Exception as e:
            import traceback
            print(f"Audio processing error: {e}")
            traceback.print_exc()

    # Run tasks concurrently
    try:
        receiver_task = asyncio.create_task(receive_commands())
        sender_task = asyncio.create_task(send_frames())
        audio_task = asyncio.create_task(process_audio())
        
        # Wait for either to finish
        done, pending = await asyncio.wait(
            [receiver_task, sender_task, audio_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        for task in pending:
            task.cancel()
            
    except Exception as e:
        print(f"WebSocket handler error: {e}")
    finally:
        video_capture.stop()
        audio_capture.stop()
        # End the session
        try:
            database.end_session(session_id)
            print(f"Session ended: {session_id}")
        except Exception as e:
            print(f"Session end error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

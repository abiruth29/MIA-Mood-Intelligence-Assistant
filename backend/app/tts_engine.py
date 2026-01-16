"""
Text-to-Speech (TTS) System for MIA
Uses pyttsx3 (offline) or edge-tts (Microsoft Edge voices) for natural speech output.
"""

import asyncio
import edge_tts
import io
import base64
from typing import Optional, Dict, Any
import tempfile
import os

class TTSEngine:
    """
    Text-to-Speech engine with emotion-aware voice modulation.
    Uses Microsoft Edge TTS for high-quality, natural voices.
    """
    
    # Voice configurations for different emotional contexts
    # Note: edge-tts uses Hz for pitch (e.g., "-10Hz", "+5Hz") and percentage for rate
    EMOTION_VOICES = {
        # Calm, soothing voice for stress/anger
        "anger": {
            "voice": "en-US-JennyNeural",
            "rate": "-10%",  # Slower
            "pitch": "-10Hz",  # Lower pitch
            "style": "calm"
        },
        # Warm, gentle voice for sadness
        "sadness": {
            "voice": "en-US-AriaNeural",
            "rate": "-15%",
            "pitch": "-5Hz",
            "style": "empathetic"
        },
        # Reassuring voice for fear
        "fear": {
            "voice": "en-US-JennyNeural",
            "rate": "-5%",
            "pitch": "-5Hz",
            "style": "friendly"
        },
        # Upbeat voice for happiness
        "happiness": {
            "voice": "en-US-AriaNeural",
            "rate": "+5%",
            "pitch": "+10Hz",
            "style": "cheerful"
        },
        # Curious, engaged voice for surprise
        "surprise": {
            "voice": "en-US-GuyNeural",
            "rate": "+0%",
            "pitch": "+5Hz",
            "style": "excited"
        },
        # Neutral, clear voice for disgust
        "disgust": {
            "voice": "en-US-JennyNeural",
            "rate": "+0%",
            "pitch": "+0Hz",
            "style": "neutral"
        },
        # Default conversational voice
        "neutral": {
            "voice": "en-US-AriaNeural",
            "rate": "+0%",
            "pitch": "+0Hz",
            "style": "chat"
        }
    }
    
    # Available high-quality voices
    AVAILABLE_VOICES = [
        "en-US-AriaNeural",      # Female, versatile
        "en-US-JennyNeural",     # Female, warm
        "en-US-GuyNeural",       # Male, friendly
        "en-US-DavisNeural",     # Male, calm
        "en-GB-SoniaNeural",     # British Female
        "en-AU-NatashaNeural",   # Australian Female
    ]
    
    def __init__(self, default_voice: str = "en-US-AriaNeural"):
        """
        Initialize TTS engine.
        
        Args:
            default_voice: Default voice to use
        """
        self.default_voice = default_voice
        self.is_speaking = False
        print(f"TTS Engine initialized with voice: {default_voice}")
    
    async def synthesize(self, 
                        text: str, 
                        emotion: str = "neutral",
                        return_format: str = "base64") -> Dict[str, Any]:
        """
        Synthesize speech from text with emotion-appropriate voice.
        
        Args:
            text: Text to speak
            emotion: Current emotional context
            return_format: "base64" (for web), "bytes", or "file"
            
        Returns:
            Dict with audio data and metadata
        """
        if not text or not text.strip():
            return {"audio": None, "error": "No text provided"}
        
        # Get emotion-appropriate voice settings
        voice_config = self.EMOTION_VOICES.get(emotion.lower(), self.EMOTION_VOICES["neutral"])
        
        try:
            # Create communicate object with voice and settings
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice_config["voice"],
                rate=voice_config["rate"],
                pitch=voice_config["pitch"]
            )
            
            # Collect audio data
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            if return_format == "base64":
                audio_b64 = base64.b64encode(audio_data).decode("utf-8")
                return {
                    "audio": audio_b64,
                    "format": "mp3",
                    "voice": voice_config["voice"],
                    "emotion_style": voice_config["style"],
                    "text_length": len(text)
                }
            elif return_format == "bytes":
                return {
                    "audio": audio_data,
                    "format": "mp3",
                    "voice": voice_config["voice"]
                }
            elif return_format == "file":
                # Save to temp file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                temp_file.write(audio_data)
                temp_file.close()
                return {
                    "audio_path": temp_file.name,
                    "format": "mp3",
                    "voice": voice_config["voice"]
                }
            
        except Exception as e:
            print(f"TTS Error: {e}")
            return {"audio": None, "error": str(e)}
    
    async def synthesize_ssml(self, ssml: str) -> Dict[str, Any]:
        """
        Synthesize speech from SSML for advanced control.
        
        Args:
            ssml: SSML formatted text
            
        Returns:
            Dict with audio data
        """
        try:
            communicate = edge_tts.Communicate(ssml, voice=self.default_voice)
            
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            return {
                "audio": base64.b64encode(audio_data).decode("utf-8"),
                "format": "mp3"
            }
        except Exception as e:
            return {"audio": None, "error": str(e)}
    
    @staticmethod
    async def list_voices(language_filter: str = "en") -> list:
        """List available voices, optionally filtered by language."""
        voices = await edge_tts.list_voices()
        if language_filter:
            voices = [v for v in voices if v["Locale"].startswith(language_filter)]
        return voices
    
    def get_voice_for_emotion(self, emotion: str) -> Dict:
        """Get the voice configuration for a given emotion."""
        return self.EMOTION_VOICES.get(emotion.lower(), self.EMOTION_VOICES["neutral"])


class TTSQueue:
    """
    Queue-based TTS for handling multiple speech requests.
    Ensures responses don't overlap.
    """
    
    def __init__(self, engine: TTSEngine):
        self.engine = engine
        self.queue = asyncio.Queue()
        self.is_processing = False
        self._task = None
    
    async def add_to_queue(self, text: str, emotion: str = "neutral", priority: int = 0):
        """Add text to speech queue."""
        await self.queue.put({
            "text": text,
            "emotion": emotion,
            "priority": priority
        })
        
        if not self.is_processing:
            self._task = asyncio.create_task(self._process_queue())
    
    async def _process_queue(self):
        """Process queued speech requests."""
        self.is_processing = True
        
        while not self.queue.empty():
            item = await self.queue.get()
            result = await self.engine.synthesize(item["text"], item["emotion"])
            # The result would be sent via WebSocket in the main app
            yield result
        
        self.is_processing = False
    
    def clear_queue(self):
        """Clear all pending speech."""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break


# Simple offline fallback using pyttsx3
class OfflineTTS:
    """Offline TTS fallback using pyttsx3."""
    
    def __init__(self):
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.available = True
            print("Offline TTS (pyttsx3) initialized")
        except Exception as e:
            print(f"pyttsx3 not available: {e}")
            self.available = False
    
    def speak(self, text: str):
        """Speak text directly (blocking)."""
        if self.available:
            self.engine.say(text)
            self.engine.runAndWait()
    
    def speak_async(self, text: str):
        """Speak text in background thread."""
        if self.available:
            import threading
            thread = threading.Thread(target=self.speak, args=(text,))
            thread.start()
